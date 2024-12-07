#!/bin/bash

#SBATCH --job-name="contrastive"
#SBATCH --output=slurmjobs/log/%j.out
#SBATCH --error=slurmjobs/log/%j.err
#SBATCH -p healthyml
#SBATCH -q healthyml-main 
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50gb
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=3-00:00 

echo "#!/bin/bash"
mkdir -p slurmjobs/log
mkdir -p experimental_results
python contrastive.py --level_of_specificity=2