#!/bin/bash -l
#SBATCH --partition=gpuk80
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH -t 48:00:00
#SBATCH --exclude=gpu013

module load cuda/10.1
module load anaconda

# this works on MARCC, work on Lustre /scratch
cd $HOME/data/ravi/pytorch-speech-transformer
conda activate /work-zfs/avenka14/ravi/conda_envs/seq2seq_3.7

python3.7 gru_conversion.py

conda deactivate
