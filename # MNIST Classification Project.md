# MNIST Classification Project

This project implements and compares five different machine learning approaches for handwritten digit classification on the MNIST dataset, fulfilling all requirements for Pattern Recognition Project 1.

## 📋 Project Overview

This implementation includes all required classifiers:
1. **K-Nearest Neighbors (KNN)** - NumPy only implementation
2. **Naïve Bayes** - NumPy only with binary features
3. **Linear Classifier** - Both NumPy and PyTorch implementations
4. **Multilayer Perceptron (MLP)** - PyTorch implementation
5. **Convolutional Neural Network (CNN)** - PyTorch implementation

# MNIST Classification Project

This project implements and compares five different machine learning approaches for handwritten digit classification on the MNIST dataset, fulfilling all requirements for Pattern Recognition Project 1.

## 📋 Project Overview

This implementation includes all required classifiers:
1. **K-Nearest Neighbors (KNN)** - NumPy only implementation
2. **Naïve Bayes** - NumPy only with binary features
3. **Linear Classifier** - Both NumPy and PyTorch implementations
4. **Multilayer Perceptron (MLP)** - PyTorch implementation
5. **Convolutional Neural Network (CNN)** - PyTorch implementation

## 🗂️ Project Structure

```
mnist-project/
├── data_loader.py          # Data loading and preprocessing
├── download_mnist.py       # Automatic MNIST dataset downloader
├── knn.py                  # K-Nearest Neighbors (NumPy only)
├── naive_bayes.py          # Naïve Bayes classifier (NumPy only)
├── linear_classifier.py    # Linear classifier (NumPy/PyTorch)
├── mlp.py                  # Multilayer Perceptron (PyTorch)
├── cnn.py                  # Convolutional Neural Network (PyTorch)
├── utils.py                # Utility functions
├── main.py                 # Main experiment runner
├── demo.py                 # Quick demo with synthetic data
├── requirements.txt        # Dependencies
├── setup_project.py        # Project auto-setup script
└── REPORT_TEMPLATE.md      # Report template for assignment
```

## 🚀 Quick Start

### Option 1: Automatic Setup (Recommended)
```bash
# 1. Download and set up the complete MNIST dataset
python download_mnist.py

# 2. Run all experiments
python main.py
```

### Option 2: Quick Demo (No Download Required)
```bash
# Test all classifiers with synthetic data
python demo.py
```

### Option 3: Manual Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Organize your MNIST data:
```
mnist_data/
├── 0/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── 1/
│   └── ...
└── 9/
    └── ...
```

3. Update `DATA_DIR` in `main.py` to point to your data directory.

4. Run experiments:
```bash
python main.py
```

## 💻 Usage Examples

### Run All Experiments
```bash
python main.py
```

### Run Individual Experiments
```python
from data_loader import MNISTLoader
from knn import run_knn_experiments

loader = MNISTLoader('path/to/data')
X_train, y_train, X_test, y_test = loader.load_data()
X_train_flat = loader.flatten(loader.normalize(X_train))
X_test_flat = loader.flatten(loader.normalize(X_test))

results = run_knn_experiments(X_train_flat, y_train, X_test_flat, y_test)
```

## 🔬 Implemented Classifiers

### 1. K-Nearest Neighbors (KNN) - NumPy Only
- **Implementation**: Pure NumPy, no scikit-learn
- **Distance**: Euclidean distance in 784-dimensional space
- **Parameters**: k = 1, 3, 5 (as required)
- **Features**: Flattened 784-dimensional vectors
- **Output**: Confusion matrix, accuracy for each k

### 2. Naïve Bayes - NumPy Only
- **Implementation**: Pure NumPy with manual probability calculations
- **Features**: Binary features (pixel > 0.5 threshold)
- **Smoothing**: Laplace smoothing to handle zero probabilities
- **Assumption**: Pixel independence (naïve assumption)
- **Output**: Confusion matrix, probability maps visualization

### 3. Linear Classifier - Dual Implementation
- **NumPy Version**: Manual gradient descent implementation
- **PyTorch Version**: Using nn.Linear with autograd
- **Loss**: Cross-entropy loss
- **Optimization**: Gradient descent
- **Output**: Weight visualization as digit-like images

### 4. Multilayer Perceptron (MLP) - PyTorch
- **Architecture**: 784 → 256 → 128 → 10
- **Activation**: ReLU between hidden layers
- **Regularization**: Dropout (0.2)
- **Optimization**: Adam optimizer
- **Output**: Training curves, confusion matrix

### 5. Convolutional Neural Network (CNN) - PyTorch
- **Architecture**: 
  - Conv2d(1→32, 3×3) → ReLU → MaxPool(2×2)
  - Conv2d(32→64, 3×3) → ReLU → MaxPool(2×2)
  - Linear(3136→128) → ReLU → Dropout(0.5)
  - Linear(128→10)
- **Input**: 28×28 2D images (maintains spatial structure)
- **Output**: Feature map visualizations, confusion matrix

## 📊 Output Files

After running experiments, the following files will be generated:

- `results_summary.json` - All accuracy results
- `method_comparison.png` - Performance comparison chart
- `knn_confusion_k1.png`, `knn_confusion_k3.png`, `knn_confusion_k5.png`
- `naive_bayes_confusion.png`
- `naive_bayes_probability_maps.png` - Digit probability visualizations
- `linear_numpy_confusion.png`, `linear_pytorch_confusion.png`
- `linear_weights_visualization.png` - Learned weights as images
- `mlp_confusion.png`
- `cnn_confusion.png`
- `cnn_feature_maps.png` - Learned convolutional features

## 📋 Requirements Compliance

✅ **Dataset**: MNIST handwritten digits (~60,000 images)
✅ **Data Partitioning**: Custom 80/20 train/test split (not using standard split)
✅ **Preprocessing**: Normalization to [0,1], flattening for traditional ML
✅ **KNN**: NumPy only, Euclidean distance, k=1,3,5
✅ **Naïve Bayes**: NumPy only, binary features, Bayes rule
✅ **Linear**: Both NumPy and PyTorch implementations
✅ **MLP**: PyTorch, hidden layers with ReLU
✅ **CNN**: PyTorch, 2+ conv layers with pooling

## 🛠️ Dependencies

- Python 3.8+
- NumPy ≥ 1.21.0
- PyTorch ≥ 2.0.0
- torchvision ≥ 0.15.0
- matplotlib ≥ 3.5.0
- scikit-learn ≥ 1.0.0 (for metrics only)
- Pillow ≥ 9.0.0
- seaborn ≥ 0.11.0
- tqdm ≥ 4.62.0

## 📖 Report Writing

Use `REPORT_TEMPLATE.md` as a starting point for your project report. The template includes:

- Abstract and project description
- AI techniques explanation
- Dataset description
- Implementation tools
- Results analysis section
- References and AI assistance acknowledgment
- Appendix with code structure

## 🎯 Key Features

- **Modular Design**: Each classifier in separate file
- **Comprehensive Evaluation**: Confusion matrices, accuracy metrics
- **Visualization**: Weight matrices, probability maps, feature maps
- **Reproducible**: Fixed random seeds for consistent results
- **Educational**: Clear comments and documentation
- **Compliant**: Meets all project requirements

## 🔧 Troubleshooting

### Common Issues

1. **Missing Data**: Run `python download_mnist.py` first
2. **Memory Error**: Reduce batch size in neural network training
3. **Slow KNN**: Use subset of data for testing (modify in main.py)
4. **CUDA Errors**: Code automatically falls back to CPU

### Performance Tips

- For faster KNN: Reduce test set size in `main.py`
- For better CNN: Increase epochs (may take longer)
- For quick testing: Use `demo.py` with synthetic data

## 📚 Learning Objectives

This project demonstrates:

1. **Classical ML**: KNN and Naïve Bayes fundamentals
2. **Linear Models**: Understanding of linear decision boundaries
3. **Deep Learning**: MLP and CNN architectures
4. **Implementation Skills**: NumPy vs PyTorch differences
5. **Evaluation**: Proper ML evaluation methodology
6. **Visualization**: Interpreting learned representations

---

## 🏃‍♂️ Getting Started NOW

```bash
# Quick start - everything automated:
python download_mnist.py  # Downloads and organizes MNIST
python main.py           # Runs all 6 experiments

# OR test without download:
python demo.py          # Tests with synthetic data
```

**Ready to submit?** Use `REPORT_TEMPLATE.md` for your written report! 📝

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Organize your MNIST data:
```
mnist_data/
├── 0/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── 1/
│   └── ...
└── 9/
    └── ...
```

3. Update `DATA_DIR` in `main.py` to point to your data directory.

## Usage

Run all experiments:
```bash
python main.py
```

Run individual experiments:
```python
from data_loader import MNISTLoader
from knn import run_knn_experiments

loader = MNISTLoader('path/to/data')
X_train, y_train, X_test, y_test = loader.load_data()
X_train_flat = loader.flatten(loader.normalize(X_train))
X_test_flat = loader.flatten(loader.normalize(X_test))

results = run_knn_experiments(X_train_flat, y_train, X_test_flat, y_test)
```

## Implemented Classifiers

1. **K-Nearest Neighbors (KNN)** - NumPy only
   - Euclidean distance
   - Tested with k=1, 3, 5

2. **Naïve Bayes** - NumPy only
   - Binary features (threshold=0.5)
   - Laplace smoothing

3. **Linear Classifier** - NumPy and PyTorch
   - Softmax + cross-entropy loss
   - Gradient descent optimization

4. **Multilayer Perceptron (MLP)** - PyTorch
   - Architecture: 784 → 256 → 128 → 10
   - ReLU activation, dropout

5. **Convolutional Neural Network (CNN)** - PyTorch
   - Conv layers with MaxPooling
   - Architecture: Conv32 → Conv64 → FC128 → FC10

## Output

- Confusion matrices for each method
- Weight visualizations (Linear, CNN)
- Probability maps (Naïve Bayes)
- Performance comparison chart
- `results_summary.json` with all accuracies

## Requirements

- Python 3.8+
- NumPy
- PyTorch
- Matplotlib
- Scikit-learn (for metrics only)
- Pillow
