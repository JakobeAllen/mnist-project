import os

# Define all file contents
files = {
    'requirements.txt': '''numpy>=1.21.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
seaborn>=0.11.0
tqdm>=4.62.0''',
    
    'data_loader.py': '''import os
import numpy as np
from PIL import Image
from pathlib import Path
import random

class MNISTLoader:
    def __init__(self, data_dir, train_ratio=0.8, seed=42):
        self.data_dir = Path(data_dir)
        self.train_ratio = train_ratio
        random.seed(seed)
        np.random.seed(seed)
        
    def load_data(self):
        images = []
        labels = []
        
        for digit in range(10):
            digit_path = self.data_dir / str(digit)
            if not digit_path.exists():
                print(f"Warning: Folder {digit_path} not found")
                continue
                
            image_files = list(digit_path.glob('*.png')) + list(digit_path.glob('*.jpg'))
            
            for img_file in image_files:
                try:
                    img = Image.open(img_file).convert('L')
                    img_array = np.array(img, dtype=np.float32)
                    images.append(img_array)
                    labels.append(digit)
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
        
        images = np.array(images)
        labels = np.array(labels)
        
        indices = np.random.permutation(len(images))
        images = images[indices]
        labels = labels[indices]
        
        split_idx = int(len(images) * self.train_ratio)
        
        X_train = images[:split_idx]
        y_train = labels[:split_idx]
        X_test = images[split_idx:]
        y_test = labels[split_idx:]
        
        print(f"Loaded {len(images)} total images")
        print(f"Training set: {len(X_train)} images")
        print(f"Test set: {len(X_test)} images")
        
        return X_train, y_train, X_test, y_test
    
    @staticmethod
    def normalize(images, method='0-1'):
        if method == '0-1':
            return images / 255.0
        elif method == '-1-1':
            return (images / 255.0) * 2 - 1
        return images
    
    @staticmethod
    def flatten(images):
        return images.reshape(images.shape[0], -1)
'''
}

# Create files
base_dir = r'c:\Users\theal\Downloads\mnist-project'

for filename, content in files.items():
    filepath = os.path.join(base_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created: {filename}")

print("\n✓ Basic files created! Now you need to add the remaining files.")
print("Please download the complete code from a GitHub repository or")
print("manually create the remaining files: utils.py, knn.py, naive_bayes.py, linear_classifier.py, mlp.py, cnn.py, main.py")
