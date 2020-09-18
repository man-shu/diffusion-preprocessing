import pandas as pd
import nibabel as nib
from glob import glob
import re
import numpy as np
from tqdm import tqdm


NUMBER_OF_PARTICLES_PER_SEED = 5000


fdt_network_files = glob('results/*/pbx2/*/fdt_network_matrix')
fdt_dot1_files = glob('results/*/pbx2/*/fdt_matrix1.dot')
waytotal_files = glob('results/*/pbx2/*/waytotal')


connection_names = glob('results/*/seeds/*/*/*')
connection_names = pd.DataFrame(
    [
        re.search('.*/([0-9]{4})/.*/([^/]+)$', n).groups() + (n,)
        for n in connection_names
    ], columns=('subject_id', 'label_name', 'file_name')
).sort_values(['subject_id', 'label_name'])
connection_labels = list(
    connection_names
    .query(f'subject_id == "{connection_names.subject_id.unique()[1]}"')
    .label_name
    .str.replace('_1mmiso_roi_1mm_bin_trans.nii.gz', '')
    .str.replace('_roi_1mm_bin_trans.nii.gz', '')
    .unique()
)


networks = []
for fdt_dot, waytotal in tqdm(zip(fdt_dot1_files, waytotal_files)):
    subject = re.match(r'[^/]+/([0-9]+)/.*', fdt_dot).groups()[0]
    waytotals = [int(n.strip()) for n in open(waytotal).readlines()]
    label_names = list(sorted(connection_names.query('subject_id == @subject').label_name.unique()))
    folder = f'results/{subject}/seeds/_session_1_subject_id_{subject}_visit_1/'
    label_fnames = [
        glob(folder + '/*/' + ln)[0]
        for ln in label_names
    ]
    voxel_counts = [0] + [
        (nib.load(fn).get_fdata() > 0).sum()
        for fn in label_fnames
    ]
    label_voxel_sizes = [(nib.load(fn).get_fdata() > 0).sum() for fn in label_fnames]
    fdt1 = pd.read_csv(
        fdt_dot,
        header=None, sep='\s+',
    )
    fdt1.columns = ['src', 'dst', 'counts']

    if fdt1['src'].max() > sum(voxel_counts):
        print(f"Warning {subject} Error on voxel counts: {fdt1['src'].max()} {sum(voxel_counts)}")

    fdt1['src'] = pd.cut(fdt1['src'],np.cumsum(voxel_counts), labels=label_names)
    fdt1['dst'] = pd.cut(fdt1['dst'],np.cumsum(voxel_counts), labels=label_names)
    network = (
        fdt1.groupby(['src', 'dst']).sum() /
        (fdt1.groupby(['src', 'dst']).count() * NUMBER_OF_PARTICLES_PER_SEED * 2) +
        fdt1.groupby(['dst', 'src']).sum() /
        (fdt1.groupby(['dst', 'src']).count() * NUMBER_OF_PARTICLES_PER_SEED * 2)
    ).fillna(0)
    nn = network.reset_index().query('src < dst')
    #nn['ROI1'] = nn.src.apply(lambda s: re.search('^(combined_)*(.*)_1mm.*$', s).groups()[1])
    #nn['ROI2'] = nn.dst.apply(lambda s: re.search('^(combined_)*(.*)_1mm.*$', s).groups()[1])
    nn['connection'] = nn.apply(
        lambda t: re.sub('_1mm.*$', '', t.src) + '___' + re.sub('_1mm.*$', '', t.dst),
        axis=1
    )
    nn = nn[['connection', 'counts']].set_index('connection').T
    print(f"Max connectivity {subject}: {nn.T.max()}")
    nn['subject_id'] = subject
    nn = nn.set_index('subject_id')
    networks.append(nn)

all_network_connections = pd.concat(networks).sort_index().dropna()
all_network_connections.to_csv('matfun_probabilistic_connections.csv')

