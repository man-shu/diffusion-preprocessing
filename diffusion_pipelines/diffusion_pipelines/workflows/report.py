from niworkflows.interfaces.reportlets.masks import SimpleShowMaskRPT
from niworkflows.interfaces.reportlets.registration import (
    SimpleBeforeAfterRPT as SimpleBeforeAfter,
)
from nipype.interfaces.utility.wrappers import Function
from nipype import IdentityInterface, Node, Workflow, Merge
import os

TEMPLATE_ROOT = os.path.join(os.path.dirname(__file__), "report_template")
REPORT_TEMPLATE = os.path.join(TEMPLATE_ROOT, "report_template.html")


def _get_mean_bzero(dwi_file, bval):
    """Mean of the b=0 volumes of the input dwi file."""
    import os
    from nilearn.image import index_img, mean_img

    import numpy as np

    bvals = np.loadtxt(bval)
    # get the index of the b=0 volumes
    bzero_index = np.where(bvals == 0)[0]
    # get the mean image of the b=0 volumes
    mean_bzero_img = mean_img(index_img(dwi_file, bzero_index))
    # save the mean image
    out_file = os.path.join(os.getcwd(), "mean_bzero.nii.gz")
    mean_bzero_img.to_filename(out_file)

    return out_file


def create_html_report(
    calling_wf_name,
    report_wf_name,
    template_path,
    output_dir,
    bids_entities,
    plots,
):
    import os
    import string
    from nilearn.plotting.html_document import HTMLDocument
    import base64

    def _embed_svg(to_embed, template_path=template_path):
        with open(template_path) as f:
            template_text = f.read()
        string_template = string.Template(template_text)
        string_text = string_template.safe_substitute(**to_embed)
        f.close()

        return string_text

    def _get_html_text(subject_id, *args):
        to_embed = {"subject_id": subject_id}
        for plot in args:
            if plot is not None:
                with open(plot, "r", encoding="utf-8") as f:
                    svg_text = f.read()
                f.close()
                # get the plot name from the path
                plot_name = plot.split("/")[-2]
                to_embed[plot_name] = svg_text
        return _embed_svg(to_embed)

    subject_id = f"_subject_id_{bids_entities["subject"]}"
    html_text = _get_html_text(subject_id, *plots)
    out_file = os.path.join(
        output_dir,
        calling_wf_name,
        report_wf_name,
        subject_id,
        "report.html",
    )
    report_html = HTMLDocument(html_text).save_as_html(out_file)
    print(f"Report for {calling_wf_name} created at {out_file}")
    return out_file


# Define a dummy function that does nothing
def wait_func(*args, **kwargs):
    pass


def init_report_wf(calling_wf_name, output_dir, name="report"):
    """Create a workflow to generate a report for the diffusion preprocessing
    pipeline.

    Parameters
    ----------
    name : str, optional, by default "report"
        Name of the workflow
    output_dir : str, optional, by default "."
        Base directory to store the reports. The workflow will create a
        subdirectory called 'report' in this directory to store the reports.

    Returns
    -------
    workflow : nipype Workflow
        A nipype Workflow to generate the report for the diffusion
        preprocessing pipeline.
    """
    inputnode = Node(
        IdentityInterface(
            fields=[
                "dwi_initial",
                "bval",
                "eddy_corrected",
                "mask",
                "bet_mask",
                "dwi_rigid_registered",
                "template_t2_initial",
                "template_t2_masked",
                "bids_entities",
            ]
        ),
        name="report_inputnode",
    )
    outputnode = Node(
        IdentityInterface(fields=["out_file"]),
        name="report_outputnode",
    )
    # define a function to get the zero index of the input dwi file
    MeanBZero = Function(
        input_names=["dwi_file", "bval"],
        output_names=["out"],
        function=_get_mean_bzero,
    )
    # this node is used to get the zero index of the input dwi file
    get_intial_mean_bzero = Node(MeanBZero, name="get_intial_mean_bzero")
    # this node is used to get the zero index of the eddy corrected dwi file
    get_eddy_mean_bzero = get_intial_mean_bzero.clone("get_eddy_mean_bzero")
    # this node is used to get the zero index of the t2 template registered dwi file
    get_registered_mean_bzero = get_intial_mean_bzero.clone(
        "get_registered_mean_bzero"
    )

    # this node plots the before and after images of the eddy correction
    plot_before_after_eddy = Node(
        SimpleBeforeAfter(), name="plot_before_after_eddy"
    )
    # set labels for the before and after images
    plot_before_after_eddy.inputs.before_label = "Distorted"
    plot_before_after_eddy.inputs.after_label = "Eddy Corrected"
    # this node plots before and after images of masking T2 template
    plot_before_after_mask_t2 = Node(
        SimpleBeforeAfter(), name="plot_before_after_mask_t2"
    )
    # set labels for the before and after images
    plot_before_after_mask_t2.inputs.before_label = "T2 Template"
    plot_before_after_mask_t2.inputs.after_label = "Masked T2 Template"
    # this node plots the masked T2 template as before and the dwi registeresd
    # to it as after
    plot_before_after_t2_dwi = Node(
        SimpleBeforeAfter(), name="plot_before_after_t2_dwi"
    )
    # set labels for the before and after images
    plot_before_after_t2_dwi.inputs.before_label = "Masked T2 Template"
    plot_before_after_t2_dwi.inputs.after_label = "Registered DWI"
    # this node plots the extracted brain mask as outline on the initial dwi
    # image
    plot_bet = Node(SimpleShowMaskRPT(), name="plot_bet")
    # this node plots the transformed mask as an outline on transformed dwi
    # image
    plot_transformed = Node(SimpleShowMaskRPT(), name="plot_transformed")

    # Create a Merge node to combine the outputs of plot_bet,
    # plot_before_after_eddy, and plot_transformed
    merge_node = Node(Merge(5), name="merge_node")

    # embed plots in a html template
    CreateHTML = Function(
        input_names=[
            "calling_wf_name",
            "report_wf_name",
            "template_path",
            "output_dir",
            "bids_entities",
            "plots",
        ],
        output_names=["out_file"],
        function=create_html_report,
    )
    create_html = Node(CreateHTML, name="create_html")
    create_html.inputs.calling_wf_name = calling_wf_name
    create_html.inputs.report_wf_name = name
    create_html.inputs.template_path = REPORT_TEMPLATE
    create_html.inputs.output_dir = output_dir
    workflow = Workflow(name=name, base_dir=output_dir)
    workflow.connect(
        [
            # get the zero index of the input dwi file
            (
                inputnode,
                get_intial_mean_bzero,
                [
                    ("dwi_initial", "dwi_file"),
                    ("bval", "bval"),
                ],
            ),
            # get the zero index of the eddy corrected dwi file
            (
                inputnode,
                get_eddy_mean_bzero,
                [
                    ("eddy_corrected", "dwi_file"),
                    ("bval", "bval"),
                ],
            ),
            # get the zero index of the dwi file registered to the t2 template
            (
                inputnode,
                get_registered_mean_bzero,
                [
                    ("dwi_rigid_registered", "dwi_file"),
                    ("bval", "bval"),
                ],
            ),
            # plot the extracted brain mask as outline on the initial dwi image
            (
                inputnode,
                plot_bet,
                [
                    ("bet_mask", "mask_file"),
                    ("dwi_initial", "background_file"),
                ],
            ),
            # plot the initial dwi as before
            (
                get_intial_mean_bzero,
                plot_before_after_eddy,
                [
                    ("out", "before"),
                ],
            ),
            # plot the eddy corrected dwi as after
            (
                get_eddy_mean_bzero,
                plot_before_after_eddy,
                [
                    ("out", "after"),
                ],
            ),
            # plot the initial T2 template as before
            (
                inputnode,
                plot_before_after_mask_t2,
                [
                    ("template_t2_initial", "before"),
                ],
            ),
            # plot the masked T2 template as after
            (
                inputnode,
                plot_before_after_mask_t2,
                [
                    ("template_t2_masked", "after"),
                ],
            ),
            # plot the masked T2 template as before and transformed dwi as
            # after
            (
                inputnode,
                plot_before_after_t2_dwi,
                [
                    ("template_t2_masked", "before"),
                ],
            ),
            (
                get_registered_mean_bzero,
                plot_before_after_t2_dwi,
                [("out", "after")],
            ),
            # plot the transformed mask as an outline on transformed dwi image
            (
                inputnode,
                plot_transformed,
                [
                    ("dwi_rigid_registered", "background_file"),
                    ("mask", "mask_file"),
                ],
            ),
            # merge the outputs of plot_bet, plot_before_after_eddy,
            # plot_before_after_mask_t2, plot_transformed
            (
                plot_bet,
                merge_node,
                [
                    ("out_report", "in1"),
                ],
            ),
            (
                plot_before_after_eddy,
                merge_node,
                [
                    ("out_report", "in2"),
                ],
            ),
            (
                plot_before_after_mask_t2,
                merge_node,
                [
                    ("out_report", "in3"),
                ],
            ),
            (
                plot_before_after_t2_dwi,
                merge_node,
                [("out_report", "in4")],
            ),
            (
                plot_transformed,
                merge_node,
                [
                    ("out_report", "in5"),
                ],
            ),
            # input the bids_entities
            (
                inputnode,
                create_html,
                [
                    ("bids_entities", "bids_entities"),
                ],
            ),
            # create the html report
            (
                merge_node,
                create_html,
                [
                    ("out", "plots"),
                ],
            ),
            # output the html report
            (create_html, outputnode, [("out_file", "out_file")]),
        ]
    )
    return workflow
