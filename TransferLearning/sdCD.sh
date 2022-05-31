#!/bin/sh
#SBATCH --time=5:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=sdCD
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu_v100
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=eric.le@huskers.unl.edu
#SBATCH --error=/work/netthinker/ele3/Research/results/sdCD.err
#SBATCH --output=/work/netthinker/ele3/Research/results/sdCD.out

module load anaconda
conda activate research
python -u $@