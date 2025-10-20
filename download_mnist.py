"""
MNIST Dataset Downloader and Organizer
Downloads MNIST and organizes it into folders by digit (0-9)
"""
import os
import gzip
import numpy as np
from PIL import Image
import urllib.request
from pathlib import Path

def download_mnist_files():
    """Download MNIST dataset files if they don't exist"""
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]
    
    data_dir = Path('mnist_raw')
    data_dir.mkdir(exist_ok=True)
    
    for file in files:
        file_path = data_dir / file
        if not file_path.exists():
            print(f"Downloading {file}...")
            urllib.request.urlretrieve(base_url + file, file_path)
    
    return data_dir

def load_mnist_data(data_dir):
    """Load MNIST data from downloaded files"""
    
    def load_images(filename):
        with gzip.open(filename, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            num_images = int.from_bytes(f.read(4), 'big')
            num_rows = int.from_bytes(f.read(4), 'big')
            num_cols = int.from_bytes(f.read(4), 'big')
            
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num_images, num_rows, num_cols)
            return images
    
    def load_labels(filename):
        with gzip.open(filename, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            num_labels = int.from_bytes(f.read(4), 'big')
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels
    
    train_images = load_images(data_dir / 'train-images-idx3-ubyte.gz')
    train_labels = load_labels(data_dir / 'train-labels-idx1-ubyte.gz')
    test_images = load_images(data_dir / 't10k-images-idx3-ubyte.gz')
    test_labels = load_labels(data_dir / 't10k-labels-idx1-ubyte.gz')
    
    return train_images, train_labels, test_images, test_labels

def organize_mnist_by_digit(images, labels, output_dir, prefix="img"):
    """Organize MNIST images into folders by digit"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create digit folders
    for digit in range(10):
        (output_path / str(digit)).mkdir(exist_ok=True)
    
    # Save images into appropriate folders
    for i, (image, label) in enumerate(zip(images, labels)):
        digit_folder = output_path / str(label)
        filename = f"{prefix}_{i:05d}.png"
        filepath = digit_folder / filename
        
        # Convert to PIL Image and save
        img = Image.fromarray(image, mode='L')
        img.save(filepath)
        
        if (i + 1) % 5000 == 0:
            print(f"Saved {i + 1} images...")

def main():
    print("MNIST Dataset Downloader and Organizer")
    print("=" * 50)
    
    # Download MNIST files
    print("Step 1: Downloading MNIST dataset...")
    data_dir = download_mnist_files()
    print("✅ Download complete!")
    
    # Load MNIST data
    print("\nStep 2: Loading MNIST data...")
    train_images, train_labels, test_images, test_labels = load_mnist_data(data_dir)
    print(f"✅ Loaded {len(train_images)} training images and {len(test_images)} test images")
    
    # Organize into folders
    print("\nStep 3: Organizing images by digit...")
    
    # Combine train and test for the project requirement
    all_images = np.concatenate([train_images, test_images])
    all_labels = np.concatenate([train_labels, test_labels])
    
    organize_mnist_by_digit(all_images, all_labels, 'mnist_data', 'mnist')
    
    print("✅ Organization complete!")
    
    # Print statistics
    print("\nDataset Statistics:")
    for digit in range(10):
        count = len(list(Path('mnist_data').glob(f'{digit}/*.png')))
        print(f"  Digit {digit}: {count} images")
    
    print("\n" + "=" * 50)
    print("✨ MNIST dataset is ready!")
    print("Update DATA_DIR in main.py to: './mnist_data'")
    print("Then run: python main.py")
    print("=" * 50)

if __name__ == '__main__':
    main()