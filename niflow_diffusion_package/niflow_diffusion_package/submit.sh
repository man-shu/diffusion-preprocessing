#!/bin/bash
#
#SBATCH --job-name=m_niflow_job
#
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --mem-per-cpu=320
#SBATCH --error error_%A_%a.out
#SBATCH --exclusive
#SBATCH --wait-all-nodes=1
#SBATCH --spread-job


#
#SBATCH --array=16-20

srun python m_niflow_multiproc.py --config mg_test.cfg --id_to_process $SLURM_ARRAY_TASK_ID

