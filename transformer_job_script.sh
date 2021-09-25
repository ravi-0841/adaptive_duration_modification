#!/bin/bash -l
#SBATCH --partition=gpuk80
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH -t 30:00:00
#SBATCH --exclude=gpu019

module load cuda/10.1
module load anaconda

# this works on MARCC, work on Lustre /scratch
cd $HOME/data/ravi/adaptive_duration_modification
conda activate /work-zfs/avenka14/ravi/conda_envs/seq2seq_3.7

python3.7 main_transformer.py --encoder_layers $2 --decoder_layers $2

conda deactivate
