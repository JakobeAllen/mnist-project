# MNIST Classification Project Report Template

## Title
**Pattern Recognition Project 1: MNIST Handwritten Digit Classification using Multiple Machine Learning Approaches**

## Abstract
This project implements and compares five different machine learning classifiers for handwritten digit recognition on the MNIST dataset. The implemented approaches include K-Nearest Neighbors (KNN), Naïve Bayes, Linear Classifier, Multilayer Perceptron (MLP), and Convolutional Neural Network (CNN). Each classifier was implemented according to specific requirements, with some using only NumPy and others utilizing PyTorch. The project demonstrates the progression from classical machine learning techniques to modern deep learning approaches, analyzing their relative performance and characteristics.

## 1. Project Description

### Problem Statement
The goal of this project is to classify handwritten digits (0-9) from the MNIST dataset using various machine learning approaches. This is a fundamental computer vision task that serves as a benchmark for comparing different classification algorithms.

### Motivations and Applications
- **Educational Value**: Understanding the evolution from classical ML to deep learning
- **Computer Vision**: Foundation for optical character recognition (OCR) systems
- **Real-world Applications**: 
  - Postal mail sorting systems
  - Bank check processing
  - Digital form processing
  - Mobile app digit recognition

## 2. AI Techniques

### 2.1 K-Nearest Neighbors (KNN)
- **Principle**: Non-parametric, instance-based learning
- **Distance Metric**: Euclidean distance in 784-dimensional space
- **Implementation**: Pure NumPy, tested with k=1, 3, 5
- **Strengths**: Simple, no training phase, effective for small datasets
- **Weaknesses**: Computationally expensive for large datasets, sensitive to curse of dimensionality

### 2.2 Naïve Bayes
- **Principle**: Probabilistic classifier based on Bayes' theorem with independence assumption
- **Feature Engineering**: Binary features (pixel > 0.5 threshold)
- **Implementation**: Pure NumPy with Laplace smoothing
- **Strengths**: Fast training and prediction, works well with limited data
- **Weaknesses**: Strong independence assumption, sensitive to feature correlation

### 2.3 Linear Classifier
- **Principle**: Linear decision boundaries with softmax activation
- **Loss Function**: Cross-entropy loss with L2 regularization
- **Implementation**: Both NumPy (manual gradients) and PyTorch (autograd)
- **Strengths**: Fast training, interpretable weights, good baseline
- **Weaknesses**: Limited to linearly separable problems

### 2.4 Multilayer Perceptron (MLP)
- **Architecture**: 784 → 256 → 128 → 10 with ReLU activation
- **Implementation**: PyTorch with dropout regularization
- **Optimization**: Adam optimizer with learning rate scheduling
- **Strengths**: Can learn non-linear patterns, universal approximator
- **Weaknesses**: Prone to overfitting, requires hyperparameter tuning

### 2.5 Convolutional Neural Network (CNN)
- **Architecture**: Conv(1→32) → MaxPool → Conv(32→64) → MaxPool → FC(128) → FC(10)
- **Implementation**: PyTorch with ReLU activation and dropout
- **Strengths**: Translation invariant, hierarchical feature learning
- **Weaknesses**: More complex, requires more data and computation

## 3. Datasets

### MNIST Dataset
- **Size**: 70,000 images (60,000 training + 10,000 testing)
- **Image Dimensions**: 28×28 grayscale pixels
- **Classes**: 10 digits (0-9)
- **Data Partitioning**: Custom 80/20 train/test split as required
- **Preprocessing**: 
  - Normalization to [0,1] range
  - Flattening to 784-dimensional vectors for traditional ML
  - Maintaining 2D structure for CNN

## 4. Implementation Tools

### Programming Language and Libraries
- **Python 3.8+**: Main programming language
- **NumPy**: Matrix operations and mathematical computations
- **PyTorch**: Deep learning framework for MLP and CNN
- **Matplotlib**: Visualization and plotting
- **Scikit-learn**: Evaluation metrics (confusion matrix, classification report)
- **Pillow**: Image processing and loading
- **tqdm**: Progress bars for training visualization

### Development Environment
- **IDE**: Visual Studio Code
- **Version Control**: Git (if applicable)
- **Hardware**: CPU-based training (GPU optional for CNN)

## 5. Experimental Results

### Performance Comparison
*(Fill in actual results after running experiments)*

| Method | Test Accuracy | Training Time | Parameters |
|--------|---------------|---------------|------------|
| KNN (k=1) | 0.XXXX | Instant | N/A |
| KNN (k=3) | 0.XXXX | Instant | N/A |
| KNN (k=5) | 0.XXXX | Instant | N/A |
| Naïve Bayes | 0.XXXX | XX sec | XX |
| Linear (NumPy) | 0.XXXX | XX sec | 7,850 |
| Linear (PyTorch) | 0.XXXX | XX sec | 7,850 |
| MLP | 0.XXXX | XX sec | XXX,XXX |
| CNN | 0.XXXX | XX sec | XXX,XXX |

### Key Findings
*(Update with actual observations)*
- CNN achieved the highest accuracy due to spatial feature learning
- Linear classifiers provided good baseline performance
- KNN showed reasonable performance but high computational cost
- Naïve Bayes struggled with pixel dependencies

### Visualization Analysis
- **Weight Visualization**: Linear classifier weights resemble digit shapes
- **Confusion Matrices**: Show common misclassifications (e.g., 4/9, 3/8)
- **Probability Maps**: Naïve Bayes pixel importance patterns
- **Learning Curves**: Training convergence behavior

## 6. Hyperparameter Analysis

### KNN
- **k=1**: Highest variance, sensitive to noise
- **k=3**: Good balance between bias and variance
- **k=5**: Smoother decision boundaries, potentially underfit

### Neural Networks
- **Learning Rate**: 0.001 for MLP/CNN, 0.01 for linear
- **Batch Size**: 128 for efficiency vs. convergence trade-off
- **Dropout**: 0.2-0.5 to prevent overfitting
- **Epochs**: 20-50 based on convergence monitoring

## 7. Failure Mode Analysis

### Common Misclassifications
*(Analyze confusion matrices)*
- 4 ↔ 9: Similar upper portions
- 3 ↔ 8: Curved segments
- 5 ↔ 6: Lower loop confusion
- 1 ↔ 7: Thin vertical lines

### Model-Specific Issues
- **KNN**: Sensitive to outliers and noisy samples
- **Naïve Bayes**: Struggles with correlated pixels
- **Linear**: Cannot handle non-linear digit variations
- **Deep Models**: May overfit with limited data

## 8. References and AI Assistance

### References
1. LeCun, Y., et al. (1998). "Gradient-based learning applied to document recognition." *Proceedings of the IEEE*.
2. Bishop, C. M. (2006). "Pattern Recognition and Machine Learning." Springer.
3. Goodfellow, I., et al. (2016). "Deep Learning." MIT Press.

### AI Assistance
- **ChatGPT**: Used for debugging PyTorch implementation issues
- **GitHub Copilot**: Assisted with boilerplate code generation
- **Documentation**: PyTorch official documentation for CNN architecture design

## Appendix

### A. Code Structure
```
mnist-project/
├── data_loader.py          # Custom MNIST data loading
├── knn.py                  # K-Nearest Neighbors implementation
├── naive_bayes.py          # Naïve Bayes implementation
├── linear_classifier.py    # Linear classifier (NumPy/PyTorch)
├── mlp.py                  # Multilayer Perceptron
├── cnn.py                  # Convolutional Neural Network
├── utils.py                # Evaluation and visualization utilities
├── main.py                 # Main experiment runner
└── download_mnist.py       # Automatic dataset downloader
```

### B. Sample Code Excerpts

#### KNN Implementation (NumPy only)
```python
def predict_single(self, x):
    distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
    k_indices = np.argsort(distances)[:self.k]
    k_nearest_labels = self.y_train[k_indices]
    return np.bincount(k_nearest_labels).argmax()
```

#### CNN Architecture
```python
self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
self.pool = nn.MaxPool2d(2, 2)
self.fc1 = nn.Linear(64 * 7 * 7, 128)
self.fc2 = nn.Linear(128, num_classes)
```

### C. Sample Outputs
*(Include confusion matrices, weight visualizations, and accuracy plots)*

### D. GitHub Repository
*(If applicable, include link to code repository)*

---

**Note**: This template should be filled with actual experimental results after running the complete MNIST classification experiments.