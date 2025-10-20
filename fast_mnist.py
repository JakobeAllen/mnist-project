"""
Fast MNIST Loader - Uses NumPy arrays instead of individual PNG files
This is MUCH faster than loading 70,000 separate image files!
"""
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pickle

def create_fast_mnist():
    """Create a single NumPy file with all MNIST data for fast loading"""
    print("Creating fast-loading MNIST dataset...")
    
    # Download MNIST using PyTorch
    train_dataset = torchvision.datasets.MNIST(
        root='./pytorch_mnist', 
        train=True, 
        download=True
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./pytorch_mnist', 
        train=False, 
        download=True
    )
    
    # Convert to NumPy arrays
    print("Converting to NumPy arrays...")
    
    # Training data
    X_train = train_dataset.data.numpy()
    y_train = train_dataset.targets.numpy()
    
    # Test data
    X_test = test_dataset.data.numpy()
    y_test = test_dataset.targets.numpy()
    
    # Combine all data
    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    
    print(f"Total images: {len(X_all)}")
    print(f"Image shape: {X_all[0].shape}")
    
    # Save as a single file for fast loading
    mnist_data = {
        'images': X_all,
        'labels': y_all
    }
    
    print("Saving to mnist_fast.npz...")
    np.savez_compressed('mnist_fast.npz', **mnist_data)
    
    print("\n" + "=" * 50)
    print("✅ Fast MNIST dataset created!")
    print("=" * 50)
    print(f"File: mnist_fast.npz")
    print(f"Size: {X_all.nbytes / 1024 / 1024:.2f} MB")
    print(f"Images: {len(X_all):,}")
    print(f"\nLoading time comparison:")
    print(f"  PNG files: ~3-5 minutes")
    print(f"  NPZ file:  ~1-2 seconds ⚡")
    
    return mnist_data

class FastMNISTLoader:
    """Fast MNIST loader using pre-saved NumPy arrays"""
    
    def __init__(self, train_ratio=0.8, seed=42):
        self.train_ratio = train_ratio
        np.random.seed(seed)
        
    def load_data(self):
        """Load MNIST from pre-saved NPZ file (FAST!)"""
        print("Loading MNIST from mnist_fast.npz...")
        
        data = np.load('mnist_fast.npz')
        images = data['images']
        labels = data['labels']
        
        # Shuffle
        indices = np.random.permutation(len(images))
        images = images[indices]
        labels = labels[indices]
        
        # Split
        split_idx = int(len(images) * self.train_ratio)
        
        X_train = images[:split_idx]
        y_train = labels[:split_idx]
        X_test = images[split_idx:]
        y_test = labels[split_idx:]
        
        print(f"✅ Loaded {len(images)} total images in ~1 second!")
        print(f"Training set: {len(X_train)} images")
        print(f"Test set: {len(X_test)} images")
        
        return X_train, y_train, X_test, y_test
    
    @staticmethod
    def normalize(images, method='0-1'):
        if method == '0-1':
            return images.astype(np.float32) / 255.0
        elif method == '-1-1':
            return (images.astype(np.float32) / 255.0) * 2 - 1
        return images
    
    @staticmethod
    def flatten(images):
        return images.reshape(len(images), -1)

if __name__ == '__main__':
    # Create the fast-loading dataset
    create_fast_mnist()
    
    # Test loading speed
    print("\nTesting loading speed...")
    import time
    
    start = time.time()
    loader = FastMNISTLoader()
    X_train, y_train, X_test, y_test = loader.load_data()
    end = time.time()
    
    print(f"\n⚡ Loading time: {end - start:.2f} seconds")
    print(f"Image shape: {X_train[0].shape}")
    print(f"Label range: {y_train.min()} to {y_train.max()}")