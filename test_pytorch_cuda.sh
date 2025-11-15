#!/bin/bash
#SBATCH --job-name=pytorch_test
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=pytorch_test_%j.out

module load miniconda3/25.7
module load nvidia/cuda-11.8

eval "$(conda shell.bash hook)"
conda activate dino-gde

echo "=== Testing PyTorch + CUDA ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    # Test actual GPU computation
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.matmul(x, x)
    print('✓ Successfully ran tensor operation on GPU!')
"
