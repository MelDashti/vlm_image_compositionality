#!/bin/bash
#SBATCH --job-name=lde_open
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=lde_open_%j.out
#SBATCH --error=lde_open_%j.err

echo "========================================"
echo "LDE Open-World Classification"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

source /etc/profile.d/modules.sh
module load miniconda3/25.7
eval "$(conda shell.bash hook)"
conda activate dino-gde
cd ~/analysis/vlm_image_compositionality

echo "Running LDE OPEN-WORLD classification..."
python classification.py \
  --dataset 'mit-states' \
  --model_architecture 'dinov2_vitl14' \
  --model_pretraining 'facebook' \
  --experiment_name 'LDE' \
  --modality_IW 'image' \
  --open_world

echo ""
echo "End time: $(date)"
echo "========================================"
