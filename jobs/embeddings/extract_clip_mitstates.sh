#!/bin/bash
#SBATCH --job-name=clip_embed
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/s3767876/analysis/vlm_image_compositionality/outputs/embeddings/clip_mitstates_%j.out
#SBATCH --error=/home/s3767876/analysis/vlm_image_compositionality/outputs/embeddings/clip_mitstates_%j.err

echo "========================================"
echo "CLIP Embedding Extraction - MIT-States"
echo "Model: ViT-L-14 (OpenAI)"
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

export HF_HUB_OFFLINE=1
echo "Extracting CLIP embeddings..."
python -m datasets.compute_embeddings 'mit-states' 'ViT-L-14' 'openai'

echo ""
echo "End time: $(date)"
echo "========================================"
