#!/bin/bash
#SBATCH --job-name=gde_classify
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=gde_classify_%j.out
#SBATCH --error=gde_classify_%j.err

echo "========================================"
echo "GDE Classification Experiment"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Initialize environment
source /etc/profile.d/modules.sh
module load miniconda3/25.7
eval "$(conda shell.bash hook)"
conda activate dino-gde

# Navigate to project
cd ~/analysis/vlm_image_compositionality

echo "Current directory: $(pwd)"
echo "Python: $(which python)"
echo ""

# Run GDE classification
echo "Running GDE classification experiment..."
python classification.py \
  --dataset 'mit-states' \
  --model_architecture 'dinov2_vitl14' \
  --model_pretraining 'facebook' \
  --experiment_name 'GDE' \
  --modality_IW 'image' \
  --open_world

echo ""
echo "End time: $(date)"
echo "========================================"
