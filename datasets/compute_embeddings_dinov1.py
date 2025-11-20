"""
Extract DINOv1 embeddings for images in a dataset.
Based on compute_embeddings_dinov2.py but adapted for DINOv1.

Usage: python -m datasets.compute_embeddings_dinov1 <dataset_name>
Example: python -m datasets.compute_embeddings_dinov1 mit-states
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse
import os
from tqdm import tqdm
from pathlib import Path

from datasets.read_datasets import DATASET_PATHS
from datasets.composition_dataset import CompositionDataset


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_dinov1_model(model_name='dino_vitb16', device='cuda'):
    """
    Load DINOv1 model from torch hub.
    
    Available models:
    - dino_vits16: ViT-Small/16
    - dino_vits8: ViT-Small/8  
    - dino_vitb16: ViT-Base/16 (recommended)
    - dino_vitb8: ViT-Base/8
    """
    print(f"Loading DINOv1 model: {model_name}...")
    
    try:
        model = torch.hub.load('facebookresearch/dino:main', model_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load DINOv1 model from torch hub. "
            f"Please check your internet connection and try again.\n"
            f"Error: {e}"
        )

    model = model.to(device)
    model.eval()

    print(f"OK Model loaded successfully on {device}")
    return model, model_name


def get_dinov1_transforms():
    """
    Get standard DINOv1 preprocessing transforms.
    
    DINOv1 uses ImageNet normalization and 224x224 input size.
    """
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def extract_embeddings(dataset_name, model_name='dino_vitb16', batch_size=64, device=None):
    """
    Extract DINOv1 embeddings for all images in the dataset.
    
    Args:
        dataset_name: Name of dataset ('mit-states', 'ut-zappos')
        model_name: DINOv1 model to use
        batch_size: Batch size for extraction
        device: Device to use ('cuda' or 'cpu')
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 70)
    print("DINOv1 EMBEDDING EXTRACTION")
    print("=" * 70)
    print(f"Using device: {device}")
    
    # Load model
    model, model_name = get_dinov1_model(model_name, device)
    
    # Load dataset
    dataset_path = DATASET_PATHS[dataset_name]
    print(f"Loading dataset from: {dataset_path}")
    
    transform = get_dinov1_transforms()
    dataset = CompositionDataset(
        dataset_path,
        phase='all',
        transform=transform
    )
    
    print(f"Dataset: {dataset_name}")
    print(f"Total images: {len(dataset.data)}")
    print()
    
    # Get all image paths
    all_images, all_attrs, all_objs = zip(*dataset.data)
    all_pairs = list(zip(all_attrs, all_objs))
    
    print("Extracting embeddings...")
    print(f"Batch size: {batch_size}")
    print(f"Total images: {len(all_images)}")
    print(f"Batches: {len(all_images) // batch_size + 1}")
    print()
    
    # Extract embeddings
    all_embeddings = []
    successful_images = []
    failed_images = []
    
    with torch.no_grad():
        for img_chunk in tqdm(chunks(list(range(len(all_images))), batch_size),
                             total=len(all_images) // batch_size + 1,
                             desc='Extracting embeddings'):
            
            batch_imgs = []
            batch_indices = []
            
            for idx in img_chunk:
                img_path = all_images[idx]
                try:
                    img = dataset.loader(img_path)  # Load PIL image
                    img_tensor = transform(img)
                    batch_imgs.append(img_tensor)
                    batch_indices.append(idx)
                except Exception as e:
                    failed_images.append((img_path, str(e)))
                    continue
            
            if not batch_imgs:
                continue
            
            # Process batch
            imgs = torch.stack(batch_imgs).to(device)
            embs = model(imgs)  # DINOv1 returns embeddings directly
            
            # L2 normalize embeddings (for GDE compatibility)
            embs = embs / torch.norm(embs, p=2, dim=-1, keepdim=True)
            
            all_embeddings.append(embs.cpu())
            
            # Track successful images
            for idx in batch_indices:
                successful_images.append(all_images[idx])
    
    # Report failures
    if failed_images:
        print(f"\n⚠ WARNING: {len(failed_images)} images failed to load:")
        for img_path, error in failed_images[:10]:  # Show first 10
            print(f"  - {img_path}: {error}")
        if len(failed_images) > 10:
            print(f"  ... and {len(failed_images) - 10} more")
        print()
    
    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_images = tuple(successful_images)
    all_pairs = tuple([all_pairs[i] for i in range(len(all_pairs)) if i < len(successful_images)])
    
    print(f"\nOK Extracted {len(all_images)} embeddings")
    print(f"Embedding shape: {all_embeddings.shape}")
    print(f"Embedding dim: {all_embeddings.shape[1]}")
    
    # Verify L2 normalization
    norms = torch.norm(all_embeddings, p=2, dim=-1)
    print(f"Embedding norms (should be ~1.0): mean={norms.mean():.4f}, std={norms.std():.6f}")
    
    # Save embeddings
    output_filename = f"IMGemb_{model_name}_facebook.pt"
    output_path = os.path.join(dataset_path, output_filename)
    
    torch.save({
        'image_ids': all_images,
        'embeddings': all_embeddings,
        'pairs': all_pairs
    }, output_path)
    
    # Report file size
    file_size = os.path.getsize(output_path) / (1024 ** 2)  # MB
    print(f"\nOK Saved embeddings to: {output_path}")
    print(f"File size: {file_size:.1f} MB")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract DINOv1 embeddings')
    parser.add_argument('dataset', type=str, help='Dataset name (mit-states, ut-zappos)')
    parser.add_argument('--model', type=str, default='dino_vitb16',
                       help='DINOv1 model (dino_vits16, dino_vitb16, etc.)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    extract_embeddings(
        dataset_name=args.dataset,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device
    )
