"""
Simple MNIST Dataset Downloader
Downloads MNIST using PyTorch and organizes into folders by digit
"""
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

def download_and_organize_mnist():
    print("Downloading MNIST dataset...")
    
    # Download MNIST
    train_dataset = torchvision.datasets.MNIST(
        root='./pytorch_mnist', 
        train=True, 
        download=True, 
        transform=transforms.ToTensor()
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./pytorch_mnist', 
        train=False, 
        download=True, 
        transform=transforms.ToTensor()
    )
    
    print(f"Downloaded {len(train_dataset)} training + {len(test_dataset)} test images")
    
    # Create output directory
    output_dir = Path('mnist_data')
    output_dir.mkdir(exist_ok=True)
    
    for digit in range(10):
        (output_dir / str(digit)).mkdir(exist_ok=True)
    
    # Save training images
    print("Saving training images...")
    for idx, (image, label) in enumerate(train_dataset):
        image_pil = transforms.ToPILImage()(image)
        filepath = output_dir / str(label) / f"train_{idx:05d}.png"
        image_pil.save(filepath)
        if (idx + 1) % 10000 == 0:
            print(f"  {idx + 1} images saved...")
    
    # Save test images
    print("Saving test images...")
    for idx, (image, label) in enumerate(test_dataset):
        image_pil = transforms.ToPILImage()(image)
        filepath = output_dir / str(label) / f"test_{idx:05d}.png"
        image_pil.save(filepath)
        if (idx + 1) % 2000 == 0:
            print(f"  {idx + 1} images saved...")
    
    print("\n" + "="*50)
    print("MNIST dataset ready in mnist_data/ folder!")
    print("Run: python main.py")
    print("="*50)

if __name__ == '__main__':
    download_and_organize_mnist()
