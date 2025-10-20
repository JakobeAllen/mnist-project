"""
MNIST Project Demo with Synthetic Data
Tests all classifiers with a small synthetic dataset
"""
import numpy as np
import matplotlib.pyplot as plt
from knn import KNN
from naive_bayes import NaiveBayes
from linear_classifier import LinearClassifierNumPy, train_pytorch_linear
from mlp import MLP, train_mlp
from cnn import CNN, train_cnn
import torch

def create_synthetic_mnist():
    """Create a small synthetic MNIST-like dataset for testing"""
    print("Creating synthetic MNIST dataset for demo...")
    
    # Create 1000 synthetic 28x28 images
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic patterns for each digit
    images = []
    labels = []
    
    for digit in range(10):
        for _ in range(n_samples // 10):
            # Create a simple pattern for each digit
            img = np.zeros((28, 28))
            
            # Add some digit-specific patterns
            if digit == 0:  # Circle-like pattern
                center = (14, 14)
                for i in range(28):
                    for j in range(28):
                        dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                        if 8 < dist < 12:
                            img[i, j] = 1
            elif digit == 1:  # Vertical line
                img[5:23, 12:16] = 1
            elif digit == 2:  # Horizontal lines
                img[8:12, 5:23] = 1
                img[18:22, 5:23] = 1
            # Add more patterns for other digits...
            else:
                # Random pattern with some structure
                img[np.random.randint(0, 28, 50), np.random.randint(0, 28, 50)] = 1
            
            # Add noise
            img += np.random.normal(0, 0.1, (28, 28))
            img = np.clip(img, 0, 1)
            
            images.append(img)
            labels.append(digit)
    
    return np.array(images), np.array(labels)

def demo_classifiers():
    """Demo all classifiers with synthetic data"""
    print("=" * 60)
    print("MNIST PROJECT DEMO - SYNTHETIC DATA")
    print("=" * 60)
    
    # Create synthetic data
    X, y = create_synthetic_mnist()
    print(f"Created {len(X)} synthetic images")
    
    # Split into train/test
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    # Flatten for some algorithms
    X_train_flat = X_train.reshape(n_train, -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    
    results = {}
    
    # 1. KNN Demo
    print("\n" + "=" * 40)
    print("DEMO 1: K-Nearest Neighbors")
    print("=" * 40)
    
    knn = KNN(k=3)
    knn.fit(X_train_flat, y_train)
    # Test on small subset for speed
    test_subset = X_test_flat[:50]
    y_subset = y_test[:50]
    knn_pred = [knn.predict_single(x) for x in test_subset]
    knn_acc = np.mean(np.array(knn_pred) == y_subset)
    results['KNN'] = knn_acc
    print(f"KNN Accuracy (k=3): {knn_acc:.4f}")
    
    # 2. Naive Bayes Demo
    print("\n" + "=" * 40)
    print("DEMO 2: Naïve Bayes")
    print("=" * 40)
    
    nb = NaiveBayes()
    nb.fit(X_train_flat, y_train)
    nb_pred = nb.predict(X_test_flat)
    nb_acc = np.mean(nb_pred == y_test)
    results['Naive Bayes'] = nb_acc
    print(f"Naïve Bayes Accuracy: {nb_acc:.4f}")
    
    # 3. Linear Classifier Demo (NumPy)
    print("\n" + "=" * 40)
    print("DEMO 3: Linear Classifier (NumPy)")
    print("=" * 40)
    
    linear_np = LinearClassifierNumPy(input_dim=784, output_dim=10)
    linear_np.fit(X_train_flat, y_train)
    linear_pred = linear_np.predict(X_test_flat)
    linear_acc = np.mean(linear_pred == y_test)
    results['Linear (NumPy)'] = linear_acc
    print(f"Linear Classifier Accuracy: {linear_acc:.4f}")
    
    # 4. MLP Demo
    print("\n" + "=" * 40)
    print("DEMO 4: Multilayer Perceptron")
    print("=" * 40)
    
    mlp = MLP(input_dim=784, hidden_dims=[128, 64], output_dim=10)
    train_mlp(mlp, X_train_flat, y_train, X_test_flat, y_test, epochs=20)
    
    mlp.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_flat)
        mlp_outputs = mlp(X_test_tensor)
        mlp_pred = torch.argmax(mlp_outputs, dim=1).numpy()
    
    mlp_acc = np.mean(mlp_pred == y_test)
    results['MLP'] = mlp_acc
    print(f"MLP Accuracy: {mlp_acc:.4f}")
    
    # 5. CNN Demo
    print("\n" + "=" * 40)
    print("DEMO 5: Convolutional Neural Network")
    print("=" * 40)
    
    cnn = CNN(num_classes=10)
    train_cnn(cnn, X_train_flat, y_train, X_test_flat, y_test, epochs=10)
    
    cnn.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_flat)
        cnn_outputs = cnn(X_test_tensor)
        cnn_pred = torch.argmax(cnn_outputs, dim=1).numpy()
    
    cnn_acc = np.mean(cnn_pred == y_test)
    results['CNN'] = cnn_acc
    print(f"CNN Accuracy: {cnn_acc:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("DEMO RESULTS SUMMARY")
    print("=" * 60)
    print("Classifier Accuracies on Synthetic Data:")
    for method, acc in results.items():
        print(f"  {method:15}: {acc:.4f}")
    
    # Plot comparison
    methods = list(results.keys())
    accuracies = list(results.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, accuracies, color='lightblue', edgecolor='navy')
    plt.ylabel('Accuracy')
    plt.title('MNIST Classifiers Demo - Synthetic Data')
    plt.ylim([0, 1])
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('demo_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Demo completed! Check 'demo_results.png' for visualization.")
    print("💡 Ready to run the full project with real MNIST data:")
    print("   1. Run: python download_mnist.py")
    print("   2. Run: python main.py")

if __name__ == '__main__':
    demo_classifiers()