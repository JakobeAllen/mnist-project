"""
MNIST Project Auto-Setup Script
Run this file to automatically create all project files!
"""

import os

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# All file contents stored as a dictionary
FILE_CONTENTS = {
    
    'requirements.txt': """numpy>=1.21.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
seaborn>=0.11.0
tqdm>=4.62.0
""",

    'data_loader.py': """import os
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
""",

    'utils.py': """import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)
    return accuracy, report

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm

def visualize_weights(weights, title='Weight Visualization', save_path=None):
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    
    for i, ax in enumerate(axes.flat):
        if i < weights.shape[0]:
            w = weights[i].reshape(28, 28)
            im = ax.imshow(w, cmap='RdBu', vmin=-np.abs(w).max(), vmax=np.abs(w).max())
            ax.set_title(f'Digit {i}')
            ax.axis('off')
    
    plt.colorbar(im, ax=axes.ravel().tolist())
    plt.suptitle(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_probability_maps(prob_maps, title='Probability Maps', save_path=None):
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    
    for i, ax in enumerate(axes.flat):
        if i < prob_maps.shape[0]:
            pm = prob_maps[i].reshape(28, 28)
            ax.imshow(pm, cmap='hot', vmin=0, vmax=1)
            ax.set_title(f'Digit {i}')
            ax.axis('off')
    
    plt.suptitle(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
""",

    'knn.py': """import numpy as np
from tqdm import tqdm

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict_single(self, x):
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        unique, counts = np.unique(k_nearest_labels, return_counts=True)
        return unique[np.argmax(counts)]
    
    def predict(self, X_test):
        predictions = []
        for x in tqdm(X_test, desc=f'KNN (k={self.k})'):
            predictions.append(self.predict_single(x))
        return np.array(predictions)

def run_knn_experiments(X_train, y_train, X_test, y_test, k_values=[1, 3, 5]):
    from utils import compute_metrics, plot_confusion_matrix
    
    results = {}
    
    for k in k_values:
        print(f"\\n{'='*50}")
        print(f"Running KNN with k={k}")
        print(f"{'='*50}")
        
        knn = KNN(k=k)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        
        accuracy, report = compute_metrics(y_test, predictions)
        
        print(f"\\nAccuracy: {accuracy:.4f}")
        print("\\nClassification Report:")
        print(report)
        
        plot_confusion_matrix(y_test, predictions, 
                            title=f'KNN Confusion Matrix (k={k})',
                            save_path=f'knn_confusion_k{k}.png')
        
        results[k] = {
            'accuracy': accuracy,
            'predictions': predictions,
            'report': report
        }
    
    return results
""",

    'naive_bayes.py': """import numpy as np

class NaiveBayes:
    def __init__(self, threshold=0.5, smoothing=1e-10):
        self.threshold = threshold
        self.smoothing = smoothing
        self.class_priors = None
        self.feature_probs = None
        self.classes = None
    
    def binarize(self, X):
        return (X > self.threshold).astype(np.float32)
    
    def fit(self, X_train, y_train):
        X_binary = self.binarize(X_train)
        
        self.classes = np.unique(y_train)
        n_classes = len(self.classes)
        n_features = X_binary.shape[1]
        
        self.feature_probs = np.zeros((n_classes, n_features))
        self.class_priors = np.zeros(n_classes)
        
        for idx, c in enumerate(self.classes):
            X_c = X_binary[y_train == c]
            self.class_priors[idx] = X_c.shape[0] / X_binary.shape[0]
            self.feature_probs[idx] = (np.sum(X_c, axis=0) + self.smoothing) / (X_c.shape[0] + 2 * self.smoothing)
    
    def predict(self, X_test):
        X_binary = self.binarize(X_test)
        predictions = []
        
        for x in X_binary:
            log_probs = np.log(self.class_priors)
            
            for idx, c in enumerate(self.classes):
                feature_log_prob = x * np.log(self.feature_probs[idx] + 1e-10) + (1 - x) * np.log(1 - self.feature_probs[idx] + 1e-10)
                log_probs[idx] += np.sum(feature_log_prob)
            
            predictions.append(self.classes[np.argmax(log_probs)])
        
        return np.array(predictions)
    
    def get_probability_maps(self):
        return self.feature_probs

def run_naive_bayes_experiment(X_train, y_train, X_test, y_test):
    from utils import compute_metrics, plot_confusion_matrix, visualize_probability_maps
    
    print(f"\\n{'='*50}")
    print("Running Naïve Bayes Classifier")
    print(f"{'='*50}")
    
    nb = NaiveBayes(threshold=0.5, smoothing=1e-10)
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)
    
    accuracy, report = compute_metrics(y_test, predictions)
    
    print(f"\\nAccuracy: {accuracy:.4f}")
    print("\\nClassification Report:")
    print(report)
    
    plot_confusion_matrix(y_test, predictions,
                         title='Naïve Bayes Confusion Matrix',
                         save_path='naive_bayes_confusion.png')
    
    prob_maps = nb.get_probability_maps()
    visualize_probability_maps(prob_maps,
                              title='Naïve Bayes Probability Maps',
                              save_path='naive_bayes_prob_maps.png')
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'report': report,
        'prob_maps': prob_maps
    }
""",

    'linear_classifier.py': """import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class LinearClassifierNumPy:
    def __init__(self, input_dim=784, output_dim=10, lr=0.01, epochs=100):
        self.W = np.random.randn(output_dim, input_dim) * 0.01
        self.b = np.zeros(output_dim)
        self.lr = lr
        self.epochs = epochs
        self.loss_history = []
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        return X @ self.W.T + self.b
    
    def compute_loss(self, y_pred, y_true):
        n = y_true.shape[0]
        y_one_hot = np.zeros_like(y_pred)
        y_one_hot[np.arange(n), y_true.astype(int)] = 1
        probs = self.softmax(y_pred)
        loss = -np.sum(y_one_hot * np.log(probs + 1e-10)) / n
        return loss
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        n = X_train.shape[0]
        
        for epoch in range(self.epochs):
            logits = self.forward(X_train)
            probs = self.softmax(logits)
            loss = self.compute_loss(logits, y_train)
            self.loss_history.append(loss)
            
            y_one_hot = np.zeros_like(probs)
            y_one_hot[np.arange(n), y_train.astype(int)] = 1
            
            d_logits = (probs - y_one_hot) / n
            d_W = d_logits.T @ X_train
            d_b = np.sum(d_logits, axis=0)
            
            self.W -= self.lr * d_W
            self.b -= self.lr * d_b
            
            if (epoch + 1) % 10 == 0:
                train_acc = np.mean(self.predict(X_train) == y_train)
                val_acc = np.mean(self.predict(X_val) == y_val) if X_val is not None else 0
                print(f'Epoch {epoch+1}/{self.epochs}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    
    def predict(self, X):
        logits = self.forward(X)
        return np.argmax(logits, axis=1)

class LinearClassifierPyTorch(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super(LinearClassifierPyTorch, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

def train_pytorch_linear(model, X_train, y_train, X_val, y_val, epochs=100, lr=0.01, batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train_t.size(0))
        epoch_loss = 0
        
        for i in range(0, X_train_t.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_X = X_train_t[indices]
            batch_y = y_train_t[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / (X_train_t.size(0) / batch_size)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                train_outputs = model(X_train_t)
                train_acc = (train_outputs.argmax(1) == y_train_t).float().mean().item()
                val_outputs = model(X_val_t)
                val_acc = (val_outputs.argmax(1) == y_val_t).float().mean().item()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    
    return model, loss_history

def run_linear_experiments(X_train, y_train, X_test, y_test, use_pytorch=False):
    from utils import compute_metrics, plot_confusion_matrix, visualize_weights
    
    print(f"\\n{'='*50}")
    print(f"Running Linear Classifier ({'PyTorch' if use_pytorch else 'NumPy'})")
    print(f"{'='*50}")
    
    if use_pytorch:
        model = LinearClassifierPyTorch(input_dim=784, output_dim=10)
        model, loss_history = train_pytorch_linear(model, X_train, y_train, X_test, y_test, epochs=100, lr=0.1)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(device)
            predictions = model(X_test_t).argmax(1).cpu().numpy()
        weights = model.linear.weight.detach().cpu().numpy()
    else:
        model = LinearClassifierNumPy(lr=0.1, epochs=100)
        model.fit(X_train, y_train, X_test, y_test)
        predictions = model.predict(X_test)
        weights = model.W
        loss_history = model.loss_history
    
    accuracy, report = compute_metrics(y_test, predictions)
    
    print(f"\\nAccuracy: {accuracy:.4f}")
    print("\\nClassification Report:")
    print(report)
    
    plot_confusion_matrix(y_test, predictions, title='Linear Classifier Confusion Matrix', save_path='linear_confusion.png')
    visualize_weights(weights, title='Linear Classifier Weights', save_path='linear_weights.png')
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'report': report,
        'weights': weights,
        'loss_history': loss_history
    }
""",

    'mlp.py': """import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[256, 128], output_dim=10, dropout=0.2):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def train_mlp(model, X_train, y_train, X_val, y_val, epochs=50, lr=0.001, batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    val_acc_history = []
    
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train_t.size(0))
        epoch_loss = 0
        
        for i in range(0, X_train_t.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_X = X_train_t[indices]
            batch_y = y_train_t[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / (X_train_t.size(0) / batch_size)
        loss_history.append(avg_loss)
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_acc = (val_outputs.argmax(1) == y_val_t).float().mean().item()
            val_acc_history.append(val_acc)
            train_outputs = model(X_train_t)
            train_acc = (train_outputs.argmax(1) == y_train_t).float().mean().item()
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    
    return model, loss_history, val_acc_history

def run_mlp_experiment(X_train, y_train, X_test, y_test, hidden_dims=[256, 128], epochs=50):
    from utils import compute_metrics, plot_confusion_matrix
    
    print(f"\\n{'='*50}")
    print(f"Running MLP (hidden layers: {hidden_dims})")
    print(f"{'='*50}")
    
    model = MLP(input_dim=784, hidden_dims=hidden_dims, output_dim=10)
    model, loss_history, val_acc_history = train_mlp(model, X_train, y_train, X_test, y_test, epochs=epochs)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(device)
        predictions = model(X_test_t).argmax(1).cpu().numpy()
    
    accuracy, report = compute_metrics(y_test, predictions)
    
    print(f"\\nAccuracy: {accuracy:.4f}")
    print("\\nClassification Report:")
    print(report)
    
    plot_confusion_matrix(y_test, predictions, title='MLP Confusion Matrix', save_path='mlp_confusion.png')
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'report': report,
        'loss_history': loss_history,
        'val_acc_history': val_acc_history
    }
""",

    'cnn.py': """import torch
import torch.nn as nn
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_cnn(model, X_train, y_train, X_val, y_val, epochs=30, lr=0.001, batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    val_acc_history = []
    
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train_t.size(0))
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, X_train_t.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_X = X_train_t[indices]
            batch_y = y_train_t[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_acc = (val_outputs.argmax(1) == y_val_t).float().mean().item()
            val_acc_history.append(val_acc)
            train_outputs = model(X_train_t)
            train_acc = (train_outputs.argmax(1) == y_train_t).float().mean().item()
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    
    return model, loss_history, val_acc_history

def run_cnn_experiment(X_train, y_train, X_test, y_test, epochs=30):
    from utils import compute_metrics, plot_confusion_matrix
    
    print(f"\\n{'='*50}")
    print("Running CNN")
    print(f"{'='*50}")
    
    model = CNN(num_classes=10)
    model, loss_history, val_acc_history = train_cnn(model, X_train, y_train, X_test, y_test, epochs=epochs)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(device)
        predictions = model(X_test_t).argmax(1).cpu().numpy()
    
    accuracy, report = compute_metrics(y_test, predictions)
    
    print(f"\\nAccuracy: {accuracy:.4f}")
    print("\\nClassification Report:")
    print(report)
    
    plot_confusion_matrix(y_test, predictions, title='CNN Confusion Matrix', save_path='cnn_confusion.png')
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'report': report,
        'loss_history': loss_history,
        'val_acc_history': val_acc_history,
        'model': model
    }
""",

    'main.py': """import numpy as np
import matplotlib.pyplot as plt
from data_loader import MNISTLoader
from knn import run_knn_experiments
from naive_bayes import run_naive_bayes_experiment
from linear_classifier import run_linear_experiments
from mlp import run_mlp_experiment
from cnn import run_cnn_experiment
import json

def main():
    DATA_DIR = './mnist_data'  # UPDATE THIS PATH!
    TRAIN_RATIO = 0.8
    
    print("="*70)
    print("MNIST Classification Project")
    print("="*70)
    
    print("\\nLoading MNIST dataset...")
    loader = MNISTLoader(DATA_DIR, train_ratio=TRAIN_RATIO, seed=42)
    X_train, y_train, X_test, y_test = loader.load_data()
    
    X_train_norm = loader.normalize(X_train, method='0-1')
    X_test_norm = loader.normalize(X_test, method='0-1')
    X_train_flat = loader.flatten(X_train_norm)
    X_test_flat = loader.flatten(X_test_norm)
    
    all_results = {}
    
    # KNN
    print("\\n" + "="*70)
    print("EXPERIMENT 1: K-Nearest Neighbors")
    print("="*70)
    knn_results = run_knn_experiments(X_train_flat, y_train, X_test_flat, y_test, k_values=[1, 3, 5])
    all_results['knn'] = {k: v['accuracy'] for k, v in knn_results.items()}
    
    # Naive Bayes
    print("\\n" + "="*70)
    print("EXPERIMENT 2: Naïve Bayes")
    print("="*70)
    nb_results = run_naive_bayes_experiment(X_train_flat, y_train, X_test_flat, y_test)
    all_results['naive_bayes'] = nb_results['accuracy']
    
    # Linear (NumPy)
    print("\\n" + "="*70)
    print("EXPERIMENT 3: Linear Classifier (NumPy)")
    print("="*70)
    linear_numpy_results = run_linear_experiments(X_train_flat, y_train, X_test_flat, y_test, use_pytorch=False)
    all_results['linear_numpy'] = linear_numpy_results['accuracy']
    
    # Linear (PyTorch)
    print("\\n" + "="*70)
    print("EXPERIMENT 4: Linear Classifier (PyTorch)")
    print("="*70)
    linear_pytorch_results = run_linear_experiments(X_train_flat, y_train, X_test_flat, y_test, use_pytorch=True)
    all_results['linear_pytorch'] = linear_pytorch_results['accuracy']
    
    # MLP
    print("\\n" + "="*70)
    print("EXPERIMENT 5: Multilayer Perceptron")
    print("="*70)
    mlp_results = run_mlp_experiment(X_train_flat, y_train, X_test_flat, y_test, hidden_dims=[256, 128], epochs=50)
    all_results['mlp'] = mlp_results['accuracy']
    
    # CNN
    print("\\n" + "="*70)
    print("EXPERIMENT 6: Convolutional Neural Network")
    print("="*70)
    cnn_results = run_cnn_experiment(X_train_flat, y_train, X_test_flat, y_test, epochs=30)
    all_results['cnn'] = cnn_results['accuracy']
    
    # Summary
    print("\\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print("\\nTest Accuracies:")
    print(f"  KNN (k=1):              {all_results['knn'][1]:.4f}")
    print(f"  KNN (k=3):              {all_results['knn'][3]:.4f}")
    print(f"  KNN (k=5):              {all_results['knn'][5]:.4f}")
    print(f"  Naïve Bayes:            {all_results['naive_bayes']:.4f}")
    print(f"  Linear (NumPy):         {all_results['linear_numpy']:.4f}")
    print(f"  Linear (PyTorch):       {all_results['linear_pytorch']:.4f}")
    print(f"  MLP:                    {all_results['mlp']:.4f}")
    print(f"  CNN:                    {all_results['cnn']:.4f}")
    
    with open('results_summary.json', 'w') as f:
        json.dump(all_results, f, indent=4)
    
    plot_results_comparison(all_results)
    
    print("\\n" + "="*70)
    print("All experiments completed!")
    print("="*70)

def plot_results_comparison(results):
    methods = ['KNN\\n(k=1)', 'KNN\\n(k=3)', 'KNN\\n(k=5)', 
               'Naïve\\nBayes', 'Linear\\n(NumPy)', 'Linear\\n(PyTorch)', 'MLP', 'CNN']
    accuracies = [
        results['knn'][1], results['knn'][3], results['knn'][5],
        results['naive_bayes'], results['linear_numpy'], 
        results['linear_pytorch'], results['mlp'], results['cnn']
    ]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(methods, accuracies, color='skyblue', edgecolor='navy')
    max_idx = np.argmax(accuracies)
    bars[max_idx].set_color('gold')
    bars[max_idx].set_edgecolor('darkgoldenrod')
    
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('MNIST Classification: Method Comparison', fontsize=14, fontweight='bold')
    plt.ylim([0, 1])
    plt.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
"""
}

def create_files():
    """Create all project files"""
    print("🚀 Creating MNIST Project Files...")
    print(f"📁 Base directory: {BASE_DIR}\n")
    
    created_files = []
    failed_files = []
    
    for filename, content in FILE_CONTENTS.items():
        filepath = os.path.join(BASE_DIR, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            created_files.append(filename)
            print(f"✅ Created: {filename}")
        except Exception as e:
            failed_files.append((filename, str(e)))
            print(f"❌ Failed: {filename} - {e}")
    
    print(f"\n{'='*70}")
    print(f"✨ Successfully created {len(created_files)} files!")
    
    if failed_files:
        print(f"⚠️  Failed to create {len(failed_files)} files:")
        for fname, error in failed_files:
            print(f"   - {fname}: {error}")
    
    print(f"\n{'='*70}")
    print("📋 NEXT STEPS:")
    print("1. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("\n2. Organize your MNIST data into folders (0-9)")
    print("   Example: mnist_data/0/, mnist_data/1/, ... mnist_data/9/")
    print("\n3. Update DATA_DIR in main.py to point to your data folder")
    print("\n4. Run the project:")
    print("   python main.py")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    create_files()
