#!/bin/bash
#SBATCH --job-name=dino_extract
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=dino_extract_%j.out
#SBATCH --error=dino_extract_%j.err

echo "========================================"
echo "DINOv2 Embedding Extraction"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Initialize module system
source /etc/profile.d/modules.sh

# Load modules
module load miniconda3/25.7
module load nvidia/cuda-11.8

# Activate environment
eval "$(conda shell.bash hook)"
conda activate dino-gde

# Navigate to project directory
cd ~/analysis/vlm_image_compositionality

echo "Current directory: $(pwd)"
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Run embedding extraction
echo "Starting DINOv2 embedding extraction..."
python -m datasets.compute_embeddings_dinov2 mit-states --model_size vit_large --batch_size 64

echo ""
echo "End time: $(date)"
echo "========================================"
