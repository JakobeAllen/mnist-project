# MNIST Classification Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive implementation and comparison of five different machine learning approaches for handwritten digit classification on the MNIST dataset.

## 📋 Overview

This project implements and compares classical machine learning and deep learning approaches:

1. **K-Nearest Neighbors (KNN)** - NumPy only implementation
2. **Naïve Bayes** - NumPy only with binary features
3. **Linear Classifier** - Both NumPy and PyTorch implementations
4. **Multilayer Perceptron (MLP)** - PyTorch implementation
5. **Convolutional Neural Network (CNN)** - PyTorch implementation

## 🎯 Results

| Method | Test Accuracy | Parameters | Training Time |
|--------|---------------|------------|---------------|
| **CNN** | **99.13%** | ~200K | ~10 min |
| **MLP** | **98.04%** | ~200K | ~5 min |
| **KNN (k=3)** | **93.50%** | N/A | Instant |
| **Linear (PyTorch)** | **92.36%** | 7,850 | ~2 min |
| **Linear (NumPy)** | 86.29% | 7,850 | ~3 min |
| **Naïve Bayes** | 83.80% | 7,840 | ~1 min |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/JakobeAllen/mnist-project.git
cd mnist-project

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

**Option 1: Fast Version (Recommended)** ⚡
```bash
# Download and prepare MNIST dataset
python fast_mnist.py

# Run all experiments
python main_fast.py
```

**Option 2: Original Version**
```bash
# Download MNIST as image files
python easy_download.py

# Run all experiments
python main.py
```

**Option 3: Quick Demo (No Download Required)**
```bash
# Test with synthetic data
python demo.py
```

## 📁 Project Structure

```
mnist-project/
├── knn.py                  # K-Nearest Neighbors (NumPy only)
├── naive_bayes.py          # Naïve Bayes classifier (NumPy only)
├── linear_classifier.py    # Linear classifier (NumPy/PyTorch)
├── mlp.py                  # Multilayer Perceptron (PyTorch)
├── cnn.py                  # Convolutional Neural Network (PyTorch)
├── utils.py                # Evaluation and visualization utilities
├── data_loader.py          # Data loading and preprocessing
├── main.py                 # Main experiment runner
├── main_fast.py            # Fast version with optimized data loading
├── fast_mnist.py           # Create fast-loading dataset
├── easy_download.py        # Download MNIST using PyTorch
├── demo.py                 # Quick demo with synthetic data
└── requirements.txt        # Project dependencies
```

## 🔬 Implementation Details

### K-Nearest Neighbors
- Pure NumPy implementation (no scikit-learn)
- Euclidean distance metric
- Tested with k=1, 3, 5

### Naïve Bayes
- Pure NumPy implementation
- Binary features (threshold=0.5)
- Laplace smoothing

### Linear Classifier
- Two implementations: NumPy (manual gradients) and PyTorch (autograd)
- Softmax activation + Cross-entropy loss
- Gradient descent optimization

### Multilayer Perceptron
- Architecture: 784 → 256 → 128 → 10
- ReLU activation, Dropout regularization
- Adam optimizer

### Convolutional Neural Network
- Architecture: Conv(1→32) → MaxPool → Conv(32→64) → MaxPool → FC(128) → FC(10)
- ReLU activation, Dropout regularization
- Achieves 99.13% accuracy

## 📊 Output Files

The project generates:
- `results_summary.json` - All accuracy results
- `method_comparison.png` - Performance comparison chart
- Confusion matrices for each method
- Weight visualizations
- Probability maps

## 🛠️ Requirements

- Python 3.8+
- NumPy ≥ 1.21.0
- PyTorch ≥ 2.0.0
- torchvision ≥ 0.15.0
- matplotlib ≥ 3.5.0
- scikit-learn ≥ 1.0.0 (for metrics only)
- Pillow ≥ 9.0.0
- seaborn ≥ 0.11.0
- tqdm ≥ 4.62.0

## 📈 Key Findings

1. **Deep learning dominates**: CNN (99.13%) and MLP (98.04%) significantly outperform traditional methods
2. **Spatial features matter**: CNN > MLP demonstrates the importance of convolutional layers
3. **Optimization matters**: PyTorch Linear (92.36%) > NumPy Linear (86.29%)
4. **Independence assumption fails**: Naïve Bayes (83.80%) struggles with correlated pixels
5. **KNN sweet spot**: k=3 provides the best bias-variance tradeoff

## 📝 Report

See `REPORT_TEMPLATE.md` for the complete project report template.

## 🤝 Contributing

This is an academic project. Feel free to fork and experiment!

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- MNIST dataset by Yann LeCun
- Pattern Recognition course project
- PyTorch and NumPy communities

## 👤 Author

**Jakobe Allen**
- GitHub: [@JakobeAllen](https://github.com/JakobeAllen)

---

**Note**: This project was created as part of a Pattern Recognition course assignment. The MNIST dataset is not included in the repository due to size constraints. Use the provided download scripts to obtain the data.