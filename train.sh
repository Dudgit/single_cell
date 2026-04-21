#!/bin/bash
#SBATCH --job-name=gears_norman
#SBATCH --partition=ai             
#SBATCH --gres=gpu:1               
#SBATCH --ntasks=1                
#SBATCH --cpus-per-task=32         
#SBATCH --mem=256G                  
#SBATCH --time=12:00:00            
#SBATCH --output=logs/train_GEARS.log

module load singularity
export WORKSPACE="/home/c_ai4db/c_ai4sci_scratch/vcell"
export REAL_WORKSPACE=$(readlink -f $WORKSPACE)

export SINGULARITYENV_PYTHONPATH=$REAL_WORKSPACE
cd $REAL_WORKSPACE

singularity exec --nv --pwd $REAL_WORKSPACE -B $REAL_WORKSPACE utils/vcell.sif python3 train_models.py
