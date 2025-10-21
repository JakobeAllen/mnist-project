# MNIST Handwritten Digit Classification
## Pattern Recognition Project 1

**Author:** Jakobe Allen  
**GitHub:** [@JakobeAllen](https://github.com/JakobeAllen/mnist-project)

---

## Abstract

This project implements and compares five different machine learning classifiers for handwritten digit recognition using the MNIST dataset. The implemented approaches span from classical machine learning (K-Nearest Neighbors, Naïve Bayes) to modern deep learning (Linear Classifier, Multilayer Perceptron, Convolutional Neural Network). Each classifier was implemented according to specific requirements: KNN and Naïve Bayes use pure NumPy without scikit-learn, while deep learning models leverage PyTorch. Results demonstrate the progression in accuracy from traditional methods (83.80% - 93.50%) to deep learning approaches (98.04% - 99.13%), with the CNN achieving the best performance. The project includes comprehensive evaluation through confusion matrices, weight visualizations, and comparative analysis across all methods.

---

## 1. Project Description

### Problem Statement
The goal of this project is to classify handwritten digits (0-9) from the MNIST dataset using various machine learning approaches. This fundamental computer vision task serves as a benchmark for comparing different classification algorithms, from classical statistical methods to modern deep learning architectures.

### Motivations and Applications
- **Educational Foundation:** Understanding the evolution from classical machine learning to deep learning
- **Real-World Applications:**
  - Automated postal mail sorting systems
  - Bank check processing and verification
  - Digital form processing and data entry
  - Mobile applications for digit recognition
  - Optical Character Recognition (OCR) systems

### Project Goals
1. Implement five distinct classifiers with varying complexity
2. Compare performance across classical ML and deep learning approaches
3. Analyze failure modes through confusion matrices
4. Visualize learned representations (weights, probability maps)
5. Understand the trade-offs between model complexity and accuracy

---

## 2. AI Techniques

### 2.1 K-Nearest Neighbors (KNN)
**Implementation:** Pure NumPy without scikit-learn

**Core Concepts:**
- **Instance-based learning:** No explicit training phase; stores all training examples
- **Distance metric:** Euclidean distance in 784-dimensional feature space
- **Classification:** Majority vote among k nearest neighbors
- **Tested values:** k = 1, 3, 5

**Algorithm:**
```
For each test image:
1. Compute Euclidean distance to all training images
2. Find k nearest neighbors
3. Assign the most frequent class label among neighbors
```

**Results:** 92.95% - 93.50% accuracy (k=3 performed best)

### 2.2 Naïve Bayes
**Implementation:** Pure NumPy with manual probability calculations

**Core Concepts:**
- **Probabilistic classifier:** Based on Bayes' theorem
- **Independence assumption:** Assumes pixel values are independent given the class
- **Feature engineering:** Binary features (pixel > 0.5 threshold)
- **Smoothing:** Laplace smoothing to handle zero probabilities

**Algorithm:**
```
Training:
1. Binarize training images (threshold = 0.5)
2. Estimate P(feature=1|class) for each pixel and class
3. Calculate prior probabilities P(class)

Testing:
1. Apply Bayes' rule with independence assumption
2. Select class with highest posterior probability
```

**Results:** 83.80% accuracy (limited by independence assumption)

### 2.3 Linear Classifier
**Implementation:** Dual approach - NumPy (manual gradients) and PyTorch (autograd)

**Core Concepts:**
- **Linear decision boundaries:** y = Wx + b
- **Softmax activation:** Converts logits to probabilities
- **Cross-entropy loss:** Measures prediction quality
- **Optimization:** Gradient descent

**NumPy Implementation:**
- Manual forward pass computation
- Manual gradient calculation and backpropagation
- Custom weight update rules

**PyTorch Implementation:**
- nn.Linear layer with automatic differentiation
- Adam optimizer for better convergence

**Results:** 
- NumPy: 86.29% accuracy
- PyTorch: 92.36% accuracy (better optimization)

### 2.4 Multilayer Perceptron (MLP)
**Implementation:** PyTorch with custom architecture

**Architecture:**
- Input layer: 784 neurons (28×28 flattened image)
- Hidden layer 1: 256 neurons with ReLU activation
- Hidden layer 2: 128 neurons with ReLU activation
- Output layer: 10 neurons (one per digit class)
- Dropout: 0.2 for regularization

**Training:**
- Optimizer: Adam with learning rate 0.001
- Loss function: Cross-entropy
- Batch size: 128
- Epochs: 20

**Results:** 98.04% accuracy

### 2.5 Convolutional Neural Network (CNN)
**Implementation:** PyTorch with spatial feature learning

**Architecture:**
```
Input (1×28×28)
→ Conv2D(1→32, 3×3, padding=1) → ReLU → MaxPool(2×2)
→ Conv2D(32→64, 3×3, padding=1) → ReLU → MaxPool(2×2)
→ Flatten → Linear(3136→128) → ReLU → Dropout(0.5)
→ Linear(128→10) → Softmax
```

**Key Features:**
- Maintains 2D spatial structure of images
- Hierarchical feature learning through convolutional layers
- Translation invariance through pooling
- Parameter sharing reduces overfitting

**Training:**
- Optimizer: Adam with learning rate 0.001
- Loss function: Cross-entropy
- Batch size: 128
- Epochs: 10

**Results:** 99.13% accuracy (best performance)

---

## 3. Dataset

### MNIST Handwritten Digit Dataset

**Source:** Yann LeCun's MNIST database  
**Size:** 70,000 grayscale images
- Training: 60,000 images
- Testing: 10,000 images

**Image Properties:**
- Dimensions: 28×28 pixels
- Color: Grayscale (0-255 intensity values)
- Classes: 10 digits (0-9)
- Distribution: Approximately balanced across classes

**Data Partitioning:**
Per project requirements, custom 80/20 train-test splits were created:
- Training set: 56,000 images (80%)
- Testing set: 14,000 images (20%)
- Random seed: 42 (for reproducibility)

**Preprocessing:**
1. **Normalization:** Pixel values scaled to [0,1] range by dividing by 255
2. **Flattening:** Images reshaped to 784-dimensional vectors for KNN, Naïve Bayes, Linear, and MLP
3. **2D Structure:** Maintained 28×28 shape for CNN to preserve spatial information

---

## 4. Implementation Tools

### Programming Environment
- **Language:** Python 3.13
- **IDE:** Visual Studio Code
- **Version Control:** Git/GitHub

### Core Libraries
- **NumPy 1.21.0+** - Matrix operations and mathematical computations for KNN and Naïve Bayes
- **PyTorch 2.0.0+** - Deep learning framework for Linear, MLP, and CNN
- **torchvision 0.15.0+** - MNIST dataset loading utilities

### Visualization and Analysis
- **matplotlib 3.5.0+** - Plotting confusion matrices and performance charts
- **seaborn 0.11.0+** - Enhanced visualization aesthetics
- **scikit-learn 1.0.0+** - Evaluation metrics (confusion matrix, classification report)

### Utilities
- **Pillow 9.0.0+** - Image loading and processing
- **tqdm 4.62.0+** - Progress bars for training visualization

### Hardware
- **CPU-based training** (GPU optional for CNN acceleration)
- Training time: ~20 minutes total for all experiments

---

## 5. Results and Analysis

### Performance Comparison

| Method | Test Accuracy | Parameters | Training Time | Implementation |
|--------|---------------|------------|---------------|----------------|
| **CNN** | **99.13%** | ~200,000 | ~10 min | PyTorch |
| **MLP** | **98.04%** | ~200,000 | ~5 min | PyTorch |
| **KNN (k=3)** | **93.50%** | N/A | Instant | NumPy only |
| **KNN (k=1)** | 92.95% | N/A | Instant | NumPy only |
| **KNN (k=5)** | 93.35% | N/A | Instant | NumPy only |
| **Linear (PyTorch)** | 92.36% | 7,850 | ~2 min | PyTorch |
| **Linear (NumPy)** | 86.29% | 7,850 | ~3 min | NumPy only |
| **Naïve Bayes** | 83.80% | 7,840 | ~1 min | NumPy only |

### Key Findings

1. **Deep Learning Dominance:** CNN (99.13%) and MLP (98.04%) significantly outperform traditional methods, demonstrating the power of non-linear feature learning.

2. **Spatial Features Matter:** CNN outperforms MLP despite similar parameter counts, showing that convolutional layers effectively capture spatial patterns in images.

3. **Optimization Impact:** PyTorch Linear (92.36%) substantially outperforms NumPy Linear (86.29%) with identical architectures, highlighting the importance of sophisticated optimizers (Adam vs. basic gradient descent).

4. **Independence Assumption Limitation:** Naïve Bayes (83.80%) struggles because pixels in handwritten digits are highly correlated, violating the independence assumption.

5. **KNN Bias-Variance Tradeoff:** k=3 provides the best balance (93.50%), with k=1 overfitting to noise and k=5 slightly undersmoothing decision boundaries.

### Failure Mode Analysis

**Common Misclassifications (from confusion matrices):**
- **4 ↔ 9:** Similar upper loop structures
- **3 ↔ 8:** Both contain curved segments
- **5 ↔ 6:** Lower loop confusion
- **7 ↔ 1:** Thin vertical strokes
- **2 ↔ 7:** Similar diagonal elements

**Model-Specific Challenges:**
- **Naïve Bayes:** Struggles with digits sharing similar pixel patterns (3, 5, 8)
- **Linear Classifier:** Cannot handle non-linear variations in digit writing styles
- **KNN:** Sensitive to outliers and unusual writing styles
- **Deep Models:** Occasionally confused by poorly written or ambiguous digits

### Visualizations Generated

1. **Confusion Matrices:** One for each classifier showing true vs. predicted labels
2. **Weight Visualization:** Linear classifier weights displayed as 10 digit-like images (one per class)
3. **Probability Maps:** Naïve Bayes pixel probabilities visualized as heatmaps
4. **Performance Comparison Chart:** Bar chart comparing all methods' accuracies
5. **Training Curves:** Loss and accuracy progression for neural networks

---

## 6. Project Structure

```
mnist-project/
├── knn.py                  # K-Nearest Neighbors (NumPy only)
├── naive_bayes.py          # Naïve Bayes classifier (NumPy only)
├── linear_classifier.py    # Linear classifier (NumPy + PyTorch)
├── mlp.py                  # Multilayer Perceptron (PyTorch)
├── cnn.py                  # Convolutional Neural Network (PyTorch)
├── utils.py                # Evaluation and visualization utilities
├── data_loader.py          # Custom MNIST data loading
├── main.py                 # Main experiment runner
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

### Key Code Components

**Data Loading (`data_loader.py`):**
- Custom MNIST loader from image files
- Random 80/20 train-test splitting
- Normalization and flattening utilities

**Evaluation Utilities (`utils.py`):**
- Confusion matrix generation and visualization
- Classification report generation
- Weight visualization functions
- Probability map rendering

**Main Runner (`main.py`):**
- Coordinates all experiments
- Generates comparative visualizations
- Saves results to JSON

---

## 7. Running the Project

### Installation

```bash
# Clone the repository
git clone https://github.com/JakobeAllen/mnist-project.git
cd mnist-project

# Install dependencies
pip install -r requirements.txt
```

### Execution

```bash
# Download MNIST dataset (if not already available)
# Place MNIST image files in mnist_data/ folder organized by digit (0-9)

# Run all experiments
python main.py
```

### Expected Runtime
- Data loading: < 1 second
- KNN experiments: ~2 minutes (for all k values)
- Naïve Bayes: ~1 minute
- Linear classifiers: ~5 minutes (both versions)
- MLP: ~5 minutes
- CNN: ~10 minutes
- **Total: ~20 minutes**

### Generated Outputs
- `results_summary.json` - All accuracy results
- `method_comparison.png` - Performance bar chart
- `*_confusion.png` - Confusion matrix for each classifier
- `linear_weights.png` - Visualized weight matrices
- `naive_bayes_prob_maps.png` - Probability heatmaps

---

## 8. References and AI Assistance

### Academic References

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). "Gradient-based learning applied to document recognition." *Proceedings of the IEEE*, 86(11), 2278-2324.

2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

4. Duda, R. O., Hart, P. E., & Stork, D. G. (2001). *Pattern Classification* (2nd ed.). Wiley-Interscience.

### Online Resources

5. PyTorch Documentation. "MNIST Dataset Tutorial." Retrieved from https://pytorch.org/tutorials/

6. NumPy Documentation. "Array Programming." Retrieved from https://numpy.org/doc/

### AI Assistance Statement

This project utilized AI tools in the following ways:

**GitHub Copilot:**
- Code completion suggestions for boilerplate implementations
- Syntax assistance for PyTorch model definitions
- Debugging support for NumPy array operations

**ChatGPT/Claude:**
- Clarification of project requirements and best practices
- Guidance on PyTorch DataLoader usage
- Explanation of confusion matrix interpretation
- Suggestions for code organization and modularization

**Human Contributions:**
- All algorithm implementations were hand-coded per requirements
- Architecture decisions and hyperparameter selection
- Experimental design and result analysis
- Report writing and documentation

**Compliance:** All code follows project requirements:
- KNN and Naïve Bayes use pure NumPy (no scikit-learn)
- Custom data partitioning implemented
- All required visualizations generated
- Modular, well-commented code structure

---

## 9. Conclusions

This project successfully demonstrates the progression from classical machine learning to deep learning approaches for image classification. The results clearly show that:

1. **Complexity vs. Performance:** More sophisticated models generally achieve better accuracy, but require more computational resources and training time.

2. **Feature Learning:** Deep learning models (MLP, CNN) automatically learn hierarchical features, eliminating the need for manual feature engineering required in traditional methods.

3. **Spatial Awareness:** CNNs excel at image tasks by preserving and exploiting spatial relationships through convolutional operations.

4. **Trade-offs:** Simple models like KNN provide decent accuracy (93.50%) with zero training time, making them viable for resource-constrained scenarios.

5. **Implementation Matters:** Identical architectures can achieve different results based on optimization strategies (NumPy vs. PyTorch Linear classifiers).

### Future Improvements

- Data augmentation to improve robustness
- Hyperparameter tuning with grid search
- Ensemble methods combining multiple classifiers
- Transfer learning from pre-trained models
- Analysis of computational efficiency and inference speed

---

## Appendix

### A. Sample Code Excerpts

**KNN Distance Calculation (NumPy only):**
```python
def predict_single(self, x):
    distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
    k_indices = np.argsort(distances)[:self.k]
    k_nearest_labels = self.y_train[k_indices]
    unique, counts = np.unique(k_nearest_labels, return_counts=True)
    return unique[np.argmax(counts)]
```

**CNN Architecture (PyTorch):**
```python
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
```

### B. Repository Information

**GitHub Repository:** https://github.com/JakobeAllen/mnist-project

**Dataset:** MNIST should be downloaded and organized in `mnist_data/` folder with subfolders 0-9

**License:** MIT License

**Author:** Jakobe Allen

---

**Note:** This project was developed as part of a Pattern Recognition course (Project 1). All implementations comply with the specified requirements, including NumPy-only implementations for classical methods and proper use of PyTorch for deep learning approaches.