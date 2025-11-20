#!/bin/bash
#SBATCH --job-name=dino_utzap
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/s3767876/analysis/vlm_image_compositionality/outputs/embeddings/dino_utzappos_%j.out
#SBATCH --error=/home/s3767876/analysis/vlm_image_compositionality/outputs/embeddings/dino_utzappos_%j.err

echo "========================================"
echo "DINOv2 Embedding Extraction - UT-Zappos"
echo "Model: dinov2_vitl14"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

source /etc/profile.d/modules.sh
module load miniconda3/25.7
module load nvidia/cuda-11.8
eval "$(conda shell.bash hook)"
conda activate dino-gde

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

cd /home/s3767876/analysis/vlm_image_compositionality

echo "Extracting DINOv2 embeddings for UT-Zappos..."
python -m datasets.compute_embeddings_dinov2 ut-zappos

echo ""
echo "End time: $(date)"
echo "========================================"
