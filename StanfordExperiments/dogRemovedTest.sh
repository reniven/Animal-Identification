#!/bin/sh
#SBATCH --time=168:00:00
#SBATCH --mem=375GB
#SBATCH --job-name=Image
#SBATCH --licenses=common
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu_16gb'
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=eric.le@huskers.unl.edu
#SBATCH --error=/work/netthinker/ele3/Research/StanfordExperiments/results/dogRemoved1.err
#SBATCH --output=/work/netthinker/ele3/Research/StanfordExperiments/results/dogRemoved1.out

module load anaconda
conda activate /common/netthinker/ele3/research
python -u $@