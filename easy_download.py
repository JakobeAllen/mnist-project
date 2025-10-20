"""
Easy MNIST Setup using PyTorch
Downloads and organizes MNIST automatically using torchvision
"""
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
from pathlib import Path
import numpy as np

def download_mnist_pytorch():
    """Download MNIST using PyTorch and organize into folders"""
    print("Downloading MNIST using PyTorch...")
    
    # Download MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Download both train and test sets
    train_dataset = torchvision.datasets.MNIST(
        root='./pytorch_mnist', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./pytorch_mnist', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    print(f"✅ Downloaded {len(train_dataset)} training images")
    print(f"✅ Downloaded {len(test_dataset)} test images")
    
    # Create output directory structure
    output_dir = Path('mnist_data')
    output_dir.mkdir(exist_ok=True)
    
    for digit in range(10):
        (output_dir / str(digit)).mkdir(exist_ok=True)
    
    # Organize training images
    print("Organizing training images...")
    for idx, (image, label) in enumerate(train_dataset):
        # Convert tensor to PIL Image
        image_pil = transforms.ToPILImage()(image)
        
        # Save to appropriate folder
        filename = f"train_{idx:05d}.png"
        filepath = output_dir / str(label) / filename
        image_pil.save(filepath)
        
        if (idx + 1) % 5000 == 0:
            print(f"  Saved {idx + 1} training images...")
    
    # Organize test images
    print("Organizing test images...")
    for idx, (image, label) in enumerate(test_dataset):
        # Convert tensor to PIL Image
        image_pil = transforms.ToPILImage()(image)
        
        # Save to appropriate folder
        filename = f"test_{idx:05d}.png"
        filepath = output_dir / str(label) / filename
        image_pil.save(filepath)
        
        if (idx + 1) % 2000 == 0:
            print(f"  Saved {idx + 1} test images...")
    
    # Print final statistics
    print("\n" + "=" * 50)
    print("✨ MNIST Dataset Ready!")
    print("=" * 50)
    print("Dataset Statistics:")
    total_images = 0
    for digit in range(10):
        count = len(list((output_dir / str(digit)).glob('*.png')))
        print(f"  Digit {digit}: {count:,} images")
        total_images += count
    
    print(f"\nTotal: {total_images:,} images")
    print("\nFolder structure created:")
    print("mnist_data/")
    for digit in range(10):
        print(f"├── {digit}/")
    
    print("\n🚀 Ready to run experiments!")
    print("Next step: python main.py")
    
    return output_dir

if __name__ == '__main__':
    download_mnist_pytorch()