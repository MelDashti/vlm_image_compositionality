#!/bin/bash
#SBATCH --job-name=lde_closed_gpu
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=lde_closed_gpu_%j.out
#SBATCH --error=lde_closed_gpu_%j.err

echo "========================================"
echo "LDE Closed-World on GPU (Original Settings)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

source /etc/profile.d/modules.sh
module load miniconda3/25.7
module load nvidia/cuda-11.8
eval "$(conda shell.bash hook)"
conda activate dino-gde
cd ~/analysis/vlm_image_compositionality

python classification.py \
  --dataset 'mit-states' \
  --model_architecture 'dinov2_vitl14' \
  --model_pretraining 'facebook' \
  --experiment_name 'LDE' \
  --modality_IW 'image'

echo "End time: $(date)"
