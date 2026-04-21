#!/bin/bash
#SBATCH --job-name=eval_regina
#SBATCH --partition=ai             
#SBATCH --gres=gpu:1               
#SBATCH --ntasks=1                
#SBATCH --cpus-per-task=32         
#SBATCH --mem=128G                  
#SBATCH --time=12:00:00            
#SBATCH --output=logs/evaluate_regina.log

module load singularity
export WORKSPACE="/home/c_ai4db/c_ai4sci_scratch/vcell"
export REAL_WORKSPACE=$(readlink -f $WORKSPACE)

export SINGULARITYENV_PYTHONPATH=$REAL_WORKSPACE
cd $REAL_WORKSPACE

singularity exec --nv --pwd $REAL_WORKSPACE -B $REAL_WORKSPACE utils/vcell.sif python3 evaluate_h5ad.py
