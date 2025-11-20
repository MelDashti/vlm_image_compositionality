#!/bin/bash
#SBATCH --job-name=dinov1_mit
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/s3767876/analysis/vlm_image_compositionality/outputs/embeddings/dinov1_mitstates_%j.out
#SBATCH --error=/home/s3767876/analysis/vlm_image_compositionality/outputs/embeddings/dinov1_mitstates_%j.err

echo "========================================"
echo "DINOv1 Embedding Extraction - MIT-States"
echo "Model: dino_vitb16 (ViT-Base/16)"
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

echo "Extracting DINOv1 embeddings for MIT-States..."
python -m datasets.compute_embeddings_dinov1 mit-states --model_name dino_vitb16 --batch_size 64

echo ""
echo "End time: $(date)"
echo "========================================"
