from niworkflows.interfaces.reportlets.masks import SimpleShowMaskRPT
from niworkflows.interfaces.reportlets.registration import (
    SimpleBeforeAfterRPT as SimpleBeforeAfter,
)
from nipype.interfaces.utility.wrappers import Function
from nipype import IdentityInterface, Node, Workflow
import os

TEMPLATE_ROOT = os.path.join(os.path.dirname(__file__), "report_template")
REPORT_TEMPLATE = os.path.join(TEMPLATE_ROOT, "report_template.html")


def _get_dwi_zero(dwi_file):
    """Get the zero index of the input dwi file.

    Not exactly zero index, but the 26th volume of the input dwi file.
    This is because the actual zero index is the fixed image to which the
    remaining images are moved to. So to see the changes in the eddy correction
    + coregistration, we need to see the an image that is moved.
    """
    import os
    from nilearn.image import index_img

    zero_index_img = index_img(dwi_file, 26)
    out_file = os.path.join(os.getcwd(), "zero_index.nii.gz")
    zero_index_img.to_filename(out_file)

    return out_file


def create_html_report(
    calling_wf_name,
    report_wf_name,
    plot_bet,
    plot_before_after_eddy,
    plot_transformed,
    template_path,
    output_root,
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

    def _get_html_text(*args):
        to_embed = {}
        for plot in args:
            if plot is not None:
                svg_path = os.path.join(plot, "report.svg")
                with open(svg_path, "rb") as f:
                    svg_text = str(base64.b64encode(f.read()), "utf-8")
                to_embed[plot] = svg_text
        return _embed_svg(to_embed)

    plot_dir = [plot_bet, plot_before_after_eddy, plot_transformed]
    html_text = _get_html_text(*plot_dir)
    out_file = os.path.join(
        output_root, calling_wf_name, report_wf_name, "report.html"
    )
    report_html = HTMLDocument(html_text).save_as_html(out_file)
    return out_file


def init_report_wf(calling_wf_name, output_root, name="reporter"):
    """Create a workflow to generate a report for the diffusion preprocessing
    pipeline.

    Parameters
    ----------
    name : str, optional, by default "reporter"
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
                "eddy_corrected",
                "mask",
                "bet_mask",
                "dwi_rigid_registered",
            ]
        ),
        name="reporter_inputnode",
    )
    # define a function to get the zero index of the input dwi file
    DWIZero = Function(
        input_names=["dwi_file"], output_names=["out"], function=_get_dwi_zero
    )
    # this node is used to get the zero index of the input dwi file
    get_intial_zero = Node(DWIZero, name="get_intial_zero")
    # this node is used to get the zero index of the eddy corrected dwi file
    get_eddy_zero = get_intial_zero.clone("get_eddy_zero")
    # this node plots the before and after images of the eddy correction
    plot_before_after_eddy = Node(
        SimpleBeforeAfter(), name="plot_before_after_eddy"
    )
    # set labels for the before and after images
    plot_before_after_eddy.inputs.before_label = "Distorted"
    plot_before_after_eddy.inputs.after_label = "Eddy Corrected"
    # this node plots the extracted brain mask as outline on the initial dwi
    # image
    plot_bet = Node(SimpleShowMaskRPT(), name="plot_bet")
    # this node plots the transformed mask as an outline on transformed dwi
    # image
    plot_transformed = Node(SimpleShowMaskRPT(), name="plot_transformed")

    # embed plots in a html template
    CreateHTML = Function(
        input_names=[
            "calling_wf_name",
            "report_wf_name",
            "plot_bet",
            "plot_before_after_eddy",
            "plot_transformed",
            "template_path",
            "output_root",
        ],
        output_names=["out_file"],
        function=create_html_report,
    )
    create_html = Node(CreateHTML, name="create_html")
    create_html.inputs.calling_wf_name = calling_wf_name
    create_html.inputs.report_wf_name = name
    create_html.inputs.plot_bet = plot_bet.name
    create_html.inputs.plot_before_after_eddy = plot_before_after_eddy.name
    create_html.inputs.plot_transformed = plot_transformed.name
    create_html.inputs.template_path = REPORT_TEMPLATE
    create_html.inputs.output_root = output_root

    workflow = Workflow(name=name)
    workflow.connect(
        [
            # get the zero index of the input dwi file
            (inputnode, get_intial_zero, [("dwi_initial", "dwi_file")]),
            # get the zero index of the eddy corrected dwi file
            (inputnode, get_eddy_zero, [("eddy_corrected", "dwi_file")]),
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
                get_intial_zero,
                plot_before_after_eddy,
                [
                    ("out", "before"),
                ],
            ),
            # plot the eddy corrected dwi as after
            (
                get_eddy_zero,
                plot_before_after_eddy,
                [
                    ("out", "after"),
                ],
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
            # create the html report
            # (
            #     inputnode,
            #     create_html,
            #     [
            #         ("calling_wf_name", "calling_wf_name"),
            #         (workflow.name, "report_wf_name"),
            #         (plot_bet.name, "plot_bet"),
            #         (plot_before_after_eddy.name, "plot_before_after_eddy"),
            #         (plot_transformed.name, "plot_transformed"),
            #     ],
            # ),
        ]
    )
    workflow.add_nodes([create_html])
    return workflow
