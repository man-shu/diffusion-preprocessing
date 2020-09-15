#!/bin/env python

from glob import glob
import nibabel as nib
import pandas as pd
import re
from os import path


tracts = list(sorted(glob('results/*/pbx2_cp/_session_1_subject_id_*_visit_1/_probtrackx2_bypairs*/fdt_paths.nii.gz')))
seeds = list(sorted(glob('../dwi_rois/1mm/*bin.nii.gz')))

diffusion_measuremets = {
    'FA': 'results/{subject}/dti/_session_1_subject_id_{subject}_visit_1/dtifit__FA.nii.gz',
    'MD': 'results/{subject}/dti/_session_1_subject_id_{subject}_visit_1/dtifit__MD.nii.gz'
}

combined_seeds = [
    (
        path.basename(re.search('^(combined_)*(.*)_1mm.*$', t1).groups()[1]),
        path.basename(re.search('^(combined_)*(.*)_1mm.*$', t2).groups()[1])
    )
    for i, t1 in enumerate(seeds)
    for j, t2 in enumerate(seeds)
    if i < j
]

results = []
print(len(tracts))
for tract in tracts:
    try:
        subject, tract_number = re.search('^results/([0-9]+)/.*bypairs([0-9]+)/', tract).groups()
        tract_image = nib.load(tract)
        tract_data = tract_image.get_fdata()
        tract_data /= tract_data.sum()
        res = {
            'subject': subject,
            'ROI1': combined_seeds[int(tract_number)][0],
            'ROI2': combined_seeds[int(tract_number)][1]
        }
        for k, v in diffusion_measuremets.items():
            fname = v.format(subject=subject)
            res[k] = (nib.load(fname).get_fdata() * tract_data).sum()
        results.append(res)
        print(res)
    except Exception as e:
        print(f"Problem {tract}: {e}")

results = pd.DataFrame(results).to_csv('per_tract_diffusion_measurements.csv')
