Project 1 (Pattern Recognition, Individual) by Jakobe Allen - README

Abstract

This project compares five machine learning models for handwritten digit recognition using the MNIST dataset. Two methods (KNN and Naive Bayes) were implemented using NumPy, while the other three models (Linear, MLP, and CNN) are built using PyTorch. When working on this project, you can see how accuracy improves as model complexity increases. Lastly, the results of this project will show all these methods being shown with a very high accuracy.

1. Project Description

The task for this project was to classify 28 by 28 grayscale images of written digits (0-9). Each classifier tries to correctly identify the digit from the image. This project will show how machine learning models can learn how to read images.

Motivation and Applications:
This type of project is usually used in real life to read numbers on checks, forms, and even mail. This shows how machine learning models and AI can read images and make accurate predictions from them which is very interesting.


2. AI Techniques

K-Nearest Neighbors (KNN):
- How it works: Finds the k closest training images and votes for the answer
- Accuracy: 93.30% (k=5)
- Speed: Slow 

Naive Bayes:
- How it works: Turns pixels into black & white and calculates probabilities
- Accuracy: 84.22%
- Speed: Very fast

Linear Classifier:
- How it works: Draws straight lines to separate different digits
- Accuracy: 83.48% (NumPy), 85.76% (PyTorch)
- Uses: L2 loss (this measures errors by squaring differences)
  
Multilayer Perceptron (MLP):
- How it works: Multiple layers that learn complex patterns
- Layers: 784 → 256 → 128 → 10
- Accuracy: 98.28%

Convolutional Neural Network (CNN):
 - How it works: Uses filters to detect edges, curves, and shapes in images
 - Layers: Two convolutional layers + pooling + fully connected
 - Accuracy: 99.16% (The best one!)

3. Dataset

The project uses the MNIST handwritten digit dataset:
Total images - 70,000 images
Training set - 56,000 images (80%)
Test set - 14,000 images (20%)
Image size - 28 by 28 pixels each
Classes - Digits from 0-9

Also, all of the requirements were done below for the project:

1. Split data 80/20 randomly 
2. Converted pixel values from 0-255 to 0-1
3. Reshape 28 by 28 images into 784 length lists (for KNN, Naive Bayes, Linear, MLP)
4. Maintain 28 by 28 shape for CNN


4. Implementation Tools

Python 3.13 - Programming language
NumPy - Math operations for KNN and Naive Bayes
PyTorch - Deep learning framework for Linear, MLP, and CNN
Pillow - Loading images from files
Matplotlib - Creating charts and graphs
VS Code - Code editor


5. Results

This link is what output I got when running my code:
https://drive.google.com/file/d/1lpEwM2nRSfRj3ZBy_8s41PJ0j7DUT-Bl/view?usp=sharing


Images from VScode:
https://drive.google.com/file/d/1QAgX3X6BJ3e8JJfXLu7YXeP8Y5tMBYpS/view?usp=sharing
https://drive.google.com/file/d/12RrtJNT6hZB8MhZO5W6HIpUZNzQbA7qo/view?usp=sharing


Output Accuracy Table:
+-------------------+----------+--------------+
| Model             | Accuracy | Type         |
+-------------------+----------+--------------+
| CNN               | 99.16%   | Deep Learning|
| MLP               | 98.28%   | Deep Learning|
| KNN (k=5)         | 93.30%   | Classical ML |
| KNN (k=1)         | 93.20%   | Classical ML |
| KNN (k=3)         | 92.95%   | Classical ML |
| Linear (PyTorch)  | 85.76%   | Deep Learning|
| Naive Bayes       | 84.22%   | Classical ML |
| Linear (NumPy)    | 83.48%   | Classical ML |
+-------------------+----------+--------------+

Observations:
1. CNN wins because it can see shapes and patterns in images
2. MLP is behind CNN because it learns complex patterns but doesn't use image structure
3. KNN works well while being simple
4. Linear classifiers are very limited and can only draw straight boundaries
5. Naive Bayes is the lowest because it ignores how pixels relate to each other

Average run time: 10-40 minutes in total for all experiments to load

6. Project Structure

mnist-project/
├── knn.py                  # K-Nearest Neighbors (NumPy only)
├── naive_bayes.py          # Naive Bayes (NumPy only)
├── linear_classifier.py    # Linear Classifier (NumPy + PyTorch)
├── mlp.py                  # Multilayer Perceptron (PyTorch)
├── cnn.py                  # Convolutional Neural Network (PyTorch)
├── data_loader.py          # Loads and splits MNIST images
├── utils.py                # Evaluation and visualization
├── main.py                 # Runs all experiments
├── setup_mnist.py          # Downloads MNIST dataset
├── requirements.txt        # Required Python packages
└── README.txt              # The file you are reading right now :)

Output Files Created:
- Results_summary.json (All accuracy numbers)
- Method_comparison.png (Bar chart comparing methods)
- Knn_confusion_k1.png- (Confusion matrix for KNN k=1)
- knn_confusion_k3.png (Confusion matrix for KNN k=3)
- knn_confusion_k5.png  (Confusion matrix for KNN k=5)
- Naive_bayes_confusion.png (Confusion matrix for Naive Bayes)
- linear_confusion.png (Confusion matrix for Linear)
- mlp_confusion.png (Confusion matrix for MLP)
- cnn_confusion.png (Confusion matrix for CNN)

There also could be other output files created after running the “python main.py” in the terminal. This is just the majority of the output files that could be created.

How to Run:

Step 1 (Install packages):
pip install -r requirements.txt

Step 2 (Download MNIST Dataset):
python setup_mnist.py

Step 3 (Run all the experiments):
python main.py

7. References and AI Assistance

References:
- PyTorch Official Documentation: pytorch.org/docs
- NumPy Official Documentation: numpy.org/doc
- MNIST Dataset: Yann LeCun's Database

AI Tools Used:
- GitHub Copilot: Some code completion and syntax help
- ChatGPT/Claude: Debugging help, project structure guidance, and helped with ideas for this report

Work by me:
- All algorithms implemented manually following project requirements
- Model setup
- Testing the models (There was a lot of testing)

8. Conclusion

In conclusion, this project shows that deep learning models provide better performance than classical machine learning models when working with image data. CNN achieved the best accuracy because it kept the images' shapes and patterns. Also, KNN still works well when you have less computing power.

9. Github

I made a public github for my code: https://github.com/JakobeAllen/mnist-project


