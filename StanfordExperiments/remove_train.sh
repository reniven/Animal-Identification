#!/bin/sh
#SBATCH --time=168:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=subsetSmall
#SBATCH --licenses=common
#SBATCH --nodes=1
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=eric.le@huskers.unl.edu
#SBATCH --error=/work/netthinker/ele3/Research/StanfordExperiments/results/remove.err
#SBATCH --output=/work/netthinker/ele3/Research/StanfordExperiments/results/remove.out

module load anaconda
conda activate /common/netthinker/ele3/research
python -u $@