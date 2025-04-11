import numpy as np
import nibabel as nib
import os


def create_nifti_image(
    shape, affine, filename, brain_center, brain_radius, skull_thickness
):
    data = np.zeros(shape)
    if len(shape) == 3:
        x, y, z = np.indices(shape)
    else:
        x, y, z, _ = np.indices(shape)

    # Create brain-like structure
    brain_mask = (x - brain_center[0]) ** 2 + (y - brain_center[1]) ** 2 + (
        z - brain_center[2]
    ) ** 2 <= brain_radius**2
    data[brain_mask] = 1

    # Create skull-like structure
    skull_mask = (
        (x - brain_center[0]) ** 2
        + (y - brain_center[1]) ** 2
        + (z - brain_center[2]) ** 2
        <= (brain_radius + skull_thickness) ** 2
    ) & ~brain_mask
    data[skull_mask] = 2

    img = nib.Nifti1Image(data, affine)
    nib.save(img, filename)


def create_bval_bvec(bval_filename, bvec_filename, num_gradients):
    bvals = np.random.randint(0, 3000, num_gradients)
    bvecs = np.random.rand(3, num_gradients)
    bvecs /= np.linalg.norm(bvecs, axis=0)

    np.savetxt(bval_filename, bvals, fmt="%d")
    np.savetxt(bvec_filename, bvecs, fmt="%.6f")


def create_roi(shape, affine, filename, center, radius):
    data = np.zeros(shape)
    x, y, z = np.indices(shape)
    mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (
        z - center[2]
    ) ** 2 <= radius**2
    data[mask] = 1
    img = nib.Nifti1Image(data, affine)
    nib.save(img, filename)


def create_template(shape, affine, filename, brain_center, brain_radius):
    data = np.zeros(shape)
    x, y, z = np.indices(shape)

    # Create brain-like structure
    brain_mask = (x - brain_center[0]) ** 2 + (y - brain_center[1]) ** 2 + (
        z - brain_center[2]
    ) ** 2 <= brain_radius**2
    data[brain_mask] = 1

    img = nib.Nifti1Image(data, affine)
    nib.save(img, filename)


output_dir = "/Users/himanshu/Desktop/diffusion/synthetic_data"
os.makedirs(output_dir, exist_ok=True)

# Create synthetic T1 and T2 images with brain and skull
affine = np.eye(4)
brain_center = (16, 16, 16)
brain_radius = 10
skull_thickness = 2

create_nifti_image(
    (32, 32, 32),
    affine,
    os.path.join(output_dir, "synthetic_T1.nii"),
    brain_center,
    brain_radius,
    skull_thickness,
)
create_nifti_image(
    (32, 32, 32),
    affine,
    os.path.join(output_dir, "synthetic_T2.nii"),
    brain_center,
    brain_radius,
    skull_thickness,
)

# Create synthetic DWI image
create_nifti_image(
    (32, 32, 32, 5),
    affine,
    os.path.join(output_dir, "synthetic_dwi.nii.gz"),
    brain_center,
    brain_radius,
    skull_thickness,
)

# Create synthetic bval and bvec files
create_bval_bvec(
    os.path.join(output_dir, "synthetic_dwi.bval"),
    os.path.join(output_dir, "synthetic_dwi.bvec"),
    5,
)

# Create synthetic ROIs
roi_dir = os.path.join(output_dir, "rois")
os.makedirs(roi_dir, exist_ok=True)
create_roi(
    (32, 32, 32),
    affine,
    os.path.join(roi_dir, "roi1.nii.gz"),
    center=(16, 16, 16),
    radius=3,
)
create_roi(
    (32, 32, 32),
    affine,
    os.path.join(roi_dir, "roi2.nii.gz"),
    center=(10, 10, 10),
    radius=3,
)
create_roi(
    (32, 32, 32),
    affine,
    os.path.join(roi_dir, "roi3.nii.gz"),
    center=(22, 22, 22),
    radius=3,
)

# Create synthetic template data with different affine
template_dir = os.path.join(output_dir, "template")
os.makedirs(template_dir, exist_ok=True)
template_affine = np.array(
    [[2, 0, 0, 50], [0, 2, 0, 50], [0, 0, 2, 50], [0, 0, 0, 1]]
)
create_template(
    (32, 32, 32),
    template_affine,
    os.path.join(template_dir, "synthetic_template_T1.nii"),
    brain_center,
    brain_radius,
)
create_template(
    (32, 32, 32),
    template_affine,
    os.path.join(template_dir, "synthetic_template_T2.nii"),
    brain_center,
    brain_radius,
)
create_template(
    (32, 32, 32),
    template_affine,
    os.path.join(template_dir, "synthetic_template_mask.nii"),
    brain_center,
    brain_radius,
)
