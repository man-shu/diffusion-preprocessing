import nibabel
import nimesh
import numpy
import tempfile
import os
import numpy as np

from scipy import ndimage
from joblib import Parallel, delayed


def grad_descend(
    start_pos, gradient, dist=2, weight=[1, 1, 1], step_size=0.1, eps=1e-4
):
    """Walks a determinated distance following the gradient field"""

    weight = np.abs(weight)
    x, y, z = pos_act = start_pos
    walked_distance = 0
    step_length = np.inf

    while walked_distance < dist and step_length > eps:
        x, y, z = pos_act

        direction = ndimage.map_coordinates(
            gradient, [[0, 1, 2], [x, x, x], [y, y, y], [z, z, z]], order=1
        )

        step = step_size * direction

        step_length = np.linalg.norm(np.multiply(step, weight))
        pos_new = pos_act + step

        walked_distance += step_length

        pos_act = pos_new

    return pos_act


def shrink_surface(surface_file, reference, distance, outfile):
    """Shrinks the surfaces some mm in the wmmask followind the sig_dis"""

    surface = nimesh.io.load(surface_file)
    # Load white matter mask and its metadata
    wm_mask = nibabel.load(reference)
    voxel_size = wm_mask.header.get_zooms()[:3]

    # Compute a signed distance map inside of the white matter
    tmp_file = tempfile.NamedTemporaryFile(suffix=".nii.gz")

    command_string = "wb_command -create-signed-distance-volume {} {} {}"
    os.system(command_string.format(surface_file, reference, tmp_file.name))

    signed_distance_si = nibabel.load(tmp_file.name)
    signed_distance = signed_distance_si.get_fdata()

    # Transform the points to voxels
    mm_to_voxel_affine = numpy.linalg.inv(wm_mask.affine)
    vertices_voxel = nibabel.affines.apply_affine(
        mm_to_voxel_affine, surface.vertices
    )

    # Push vertices in the white matter following the gradient
    gradient = numpy.array(numpy.gradient(-signed_distance))
    shrinked_vertices_voxel = Parallel(n_jobs=-1)(
        delayed(grad_descend)(p, gradient, distance, voxel_size)
        for p in vertices_voxel
    )
    # Bring back to mm
    shrinked_vertices = nibabel.affines.apply_affine(
        wm_mask.affine, shrinked_vertices_voxel
    )

    # Save the shrinked surface
    shrinked_surface = nimesh.Mesh(shrinked_vertices, surface.triangles)
    nimesh.io.save(outfile, shrinked_surface)


surface_file = "/Users/himanshu/Desktop/diffusion/surface_recon_output_20241012-154737/surface_recon/freesurfer_surf_2_native/mapflow/_freesurfer_surf_2_native0/rh.white_converted.surf.gii"
reference = "/Users/himanshu/Desktop/diffusion/surface_recon_output_20241012-154737/surface_recon/mri_convert/brain_out.nii"
distance = 10
outfile = "/Users/himanshu/Desktop/diffusion/try_shrink_surface_output/rh.white_converted_shrinked.surf.gii"
os.makedirs(os.path.dirname(outfile), exist_ok=True)
shrink_surface(surface_file, reference, distance, outfile)
