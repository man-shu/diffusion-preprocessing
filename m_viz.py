import pandas as pd
import os
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np


def plot_graph(df):
    import matplotlib.pyplot as plt

    row_names = np.array(df.index)

    # Create an array of column names
    column_names = np.array(df.columns)

    # Get the number of rows and columns
    num_rows, num_columns = df.shape

    # Create a figure and subplots for each column
    fig, axs = plt.subplots(nrows=1, ncols=num_columns, figsize=(10, 5), sharey=True)


    # Iterate over each column
    for i, column_name in enumerate(column_names):
        # Get the values for the current column
        values = df[column_name].values

        # Generate corresponding x-values
        x = np.full(len(values), i + 1)

        # Create a scatter plot for the current column
        axs[i].scatter(x, values)

        # Set the x-axis tick labels
        axs[i].set_xticks([i + 1])
        axs[i].set_xticklabels([column_name])
        
        # Set the y-axis tick labels
        
        #axs[i].set_yticks(range(1, num_rows + 1))
        #axs[i].set_yticklabels(row_names)


    # Set the common y-axis label
    fig.text(0.04, 0.5, 'Connectivity Values', va='center', rotation='vertical')

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.5)

    plt.savefig('/home/parietal/dwasserm/research/data/LargeBrainNets/mathfun/scripts/my_graph')
    





st_df = pd.read_csv('/data/parietal/store/work/zmohamed/mathfun/mathfun_probabilistic_connections.csv')

st_ordered_columns = ['subject_id','combined_BN_L_Hipp_roi___combined_BN_L_PHG_roi', 'combined_BN_L_Hipp_roi___combined_BN_L_PPC_targets_roi', 'combined_BN_L_Hipp_roi___combined_BN_R_Hipp_roi', 'combined_BN_L_Hipp_roi___combined_BN_R_PHG_roi', 'combined_BN_L_Hipp_roi___combined_BN_R_PPC_targets'
		   , 'combined_BN_L_PHG_roi___combined_BN_L_PPC_targets_roi', 'combined_BN_L_PHG_roi___combined_BN_R_Hipp_roi', 'combined_BN_L_PHG_roi___combined_BN_R_PHG_roi', 'combined_BN_L_PHG_roi___combined_BN_R_PPC_targets', 
	   'combined_BN_L_PPC_targets_roi___combined_BN_R_Hipp_roi', 'combined_BN_L_PPC_targets_roi___combined_BN_R_PHG_roi' , 'combined_BN_L_PPC_targets_roi___combined_BN_R_PPC_targets', 
       'combined_BN_R_Hipp_roi___combined_BN_R_PHG_roi', 'combined_BN_R_Hipp_roi___combined_BN_R_PPC_targets', 'combined_BN_R_PHG_roi___combined_BN_R_PPC_targets']

st_df = st_df.reindex(columns=st_ordered_columns)
st_df.set_index('subject_id', inplace=True)

#print(st_df)


st_df.to_csv('/data/parietal/store/work/zmohamed/mathfun/st_df.csv')

my_df = pd.read_csv('/data/parietal/store/work/zmohamed/mathfun/output/multi_index_conn_out/multi_index_conn.csv')

#my_df.columns = my_df.columns.str.replace('.nii.gz', '')

my_ordered_columns = ['subject_id','combined_BN_L_PHG_roi_1mm_bin.nii.gz_combined_BN_L_Hipp_roi_1mm_bin.nii.gz', 'combined_BN_L_PPC_targets_roi_1mm_bin.nii.gz_combined_BN_L_Hipp_roi_1mm_bin.nii.gz', 'combined_BN_R_Hipp_roi_1mm_bin.nii.gz_combined_BN_L_Hipp_roi_1mm_bin.nii.gz', 'combined_BN_R_PHG_roi_1mm_bin.nii.gz_combined_BN_L_Hipp_roi_1mm_bin.nii.gz', 'combined_BN_R_PPC_targets_1mmiso_roi_1mm_bin.nii.gz_combined_BN_L_Hipp_roi_1mm_bin.nii.gz'
		   , 'combined_BN_L_PPC_targets_roi_1mm_bin.nii.gz_combined_BN_L_PHG_roi_1mm_bin.nii.gz', 'combined_BN_R_Hipp_roi_1mm_bin.nii.gz_combined_BN_L_PHG_roi_1mm_bin.nii.gz', 'combined_BN_R_PHG_roi_1mm_bin.nii.gz_combined_BN_L_PHG_roi_1mm_bin.nii.gz', 
	   'combined_BN_R_PPC_targets_1mmiso_roi_1mm_bin.nii.gz_combined_BN_L_PHG_roi_1mm_bin.nii.gz', 'combined_BN_R_Hipp_roi_1mm_bin.nii.gz_combined_BN_L_PPC_targets_roi_1mm_bin.nii.gz' , 'combined_BN_R_PHG_roi_1mm_bin.nii.gz_combined_BN_L_PPC_targets_roi_1mm_bin.nii.gz', 
       'combined_BN_R_PPC_targets_1mmiso_roi_1mm_bin.nii.gz_combined_BN_L_PPC_targets_roi_1mm_bin.nii.gz', 'combined_BN_R_PHG_roi_1mm_bin.nii.gz_combined_BN_R_Hipp_roi_1mm_bin.nii.gz', 'combined_BN_R_PPC_targets_1mmiso_roi_1mm_bin.nii.gz_combined_BN_R_Hipp_roi_1mm_bin.nii.gz', 'combined_BN_R_PPC_targets_1mmiso_roi_1mm_bin.nii.gz_combined_BN_R_PHG_roi_1mm_bin.nii.gz']


my_df = my_df[my_ordered_columns]
my_df.set_index('subject_id', inplace=True)


my_df.to_csv('/data/parietal/store/work/zmohamed/mathfun/my_df.csv')


new_column_names = ['combined_BN_L_Hipp_roi___combined_BN_L_PHG_roi', 'combined_BN_L_Hipp_roi___combined_BN_L_PPC_targets_roi', 'combined_BN_L_Hipp_roi___combined_BN_R_Hipp_roi', 'combined_BN_L_Hipp_roi___combined_BN_R_PHG_roi', 'combined_BN_L_Hipp_roi___combined_BN_R_PPC_targets'
		   , 'combined_BN_L_PHG_roi___combined_BN_L_PPC_targets_roi', 'combined_BN_L_PHG_roi___combined_BN_R_Hipp_roi', 'combined_BN_L_PHG_roi___combined_BN_R_PHG_roi', 'combined_BN_L_PHG_roi___combined_BN_R_PPC_targets', 
	   'combined_BN_L_PPC_targets_roi___combined_BN_R_Hipp_roi', 'combined_BN_L_PPC_targets_roi___combined_BN_R_PHG_roi' , 'combined_BN_L_PPC_targets_roi___combined_BN_R_PPC_targets', 
       'combined_BN_R_Hipp_roi___combined_BN_R_PHG_roi', 'combined_BN_R_Hipp_roi___combined_BN_R_PPC_targets', 'combined_BN_R_PHG_roi___combined_BN_R_PPC_targets']

my_df.columns = new_column_names


common_indexes = st_df.index.intersection(my_df.index)

print(len(common_indexes))

# Extract the common subjects
df1_common = st_df.loc[common_indexes]
df2_common = my_df.loc[common_indexes]

normalized_df1=(df1_common-df1_common.mean())/df1_common.std()
normalized_df2=(df2_common-df2_common.mean())/df2_common.std()



plt.plot(normalized_df1['combined_BN_L_Hipp_roi___combined_BN_L_PHG_roi'])
#plt.savefig('/home/parietal/dwasserm/research/data/LargeBrainNets/mathfun/scripts/st_Hipp_PHG_graph')
plt.plot(normalized_df2['combined_BN_L_Hipp_roi___combined_BN_L_PHG_roi'])
plt.savefig('/home/parietal/dwasserm/research/data/LargeBrainNets/mathfun/scripts/my_Hipp_PHG_graph')



#plot_graph(normalized_df1)
plot_graph(normalized_df2)


correlation_matrix = normalized_df1.corrwith(normalized_df2)

# Print the correlation matrix
print(correlation_matrix)

'''

values1 = df1_common.values
values2 = df2_common.values

# Compute the correlation between values for each column and each row
correlation_matrix = np.zeros((len(df1_common.columns), len(df1_common.index)))

for i in range(len(df1_common.columns)):
    for j in range(len(df1_common.index)):
        correlation_matrix[i, j] = np.corrcoef(values1[j, :], values2[j, :])[0, 1]

print(correlation_matrix)

# Reshape the correlation matrix to a 1D array

correlation_values = correlation_matrix.flatten()

# Print the correlation values
#print(correlation_values)

'''

