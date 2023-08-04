#!/bin/bash
#
#SBATCH --job-name=m_niflow_job
#SBATCH --output=res_test_job_%A_%a.txt
#
#SBATCH --ntasks=1
#SBATCH --partition=normal
#SBATCH --mem-per-cpu=8
#SBATCH --error error_%A_%a.out
#
#SBATCH --array=0-73

srun python m_niflow_multiproc.py --config mg_test.cfg --id_to_process $SLURM_ARRAY_TASK_ID

