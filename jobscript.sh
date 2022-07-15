#!/bin/bash
#SBATCH --account=opengptx-elm
#SBATCH --job-name="dlProfTest"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=develbooster
#SBATCH --gres=gpu:1
#SBATCH --output=test-out.%j
#SBATCH --error=test-err.%j
#SBATCH --time=00:10:00

echo -e "Start `date +"%F %T"` | $SLURM_JOB_ID $SLURM_JOB_NAME | `hostname` | `pwd` \n" #log start time etc.

srun apptainer exec --nv pytorch-21.11-py3.sif dlprof --mode=pytorch --reports=summary --iter_start=200 --iter_stop=400 python mnist_train.py

echo -e "End `date +"%F %T"` | $SLURM_JOB_ID $SLURM_JOB_NAME | `hostname` | `pwd` \n" #log start time etc.
set -e 