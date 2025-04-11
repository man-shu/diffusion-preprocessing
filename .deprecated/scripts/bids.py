from bids.layout import BIDSLayout


layout = BIDSLayout(
    "/data/parietal/store3/work/haggarwa/diffusion/data/WAND-bids",
)


from nipype.pipeline import Node, MapNode, Workflow
from nipype.interfaces.utility import Function
from nipype.interfaces.io import BIDSDataGrabber


def printMe(paths):
    print("\n\nanalyzing " + str(paths) + "\n\n")


analyzeDWI = Node(
    Function(function=printMe, input_names=["paths"], output_names=[]),
    name="analyzeDWI",
)


bg_all = Node(BIDSDataGrabber(), name="bids-grabber")
bg_all.inputs.base_dir = (
    "/data/parietal/store3/work/haggarwa/diffusion/data/WAND-bids"
)
bg_all.inputs.output_query = {"dwis": dict(suffix="dwi")}
bg_all.iterables = ("subject", layout.get_subjects()[:2])
wf = Workflow(name="bids_demo")
wf.connect(bg_all, "dwis", analyzeDWI, "paths")
wf.run()
