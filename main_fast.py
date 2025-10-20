"""
FAST Main - Uses NumPy arrays instead of PNG files
This version loads data in ~1 second instead of 3-5 minutes!
"""
import numpy as np
import matplotlib.pyplot as plt
from fast_mnist import FastMNISTLoader
from knn import run_knn_experiments
from naive_bayes import run_naive_bayes_experiment
from linear_classifier import run_linear_experiments
from mlp import run_mlp_experiment
from cnn import run_cnn_experiment
import json
import time

def main():
    print("="*70)
    print("MNIST Classification Project - FAST VERSION")
    print("="*70)
    
    # Load MNIST data (FAST!)
    print("\nLoading MNIST dataset...")
    start_time = time.time()
    loader = FastMNISTLoader(train_ratio=0.8, seed=42)
    X_train, y_train, X_test, y_test = loader.load_data()
    load_time = time.time() - start_time
    print(f"⚡ Data loaded in {load_time:.2f} seconds!\n")
    
    # Prepare data
    X_train_norm = loader.normalize(X_train, method='0-1')
    X_test_norm = loader.normalize(X_test, method='0-1')
    X_train_flat = loader.flatten(X_train_norm)
    X_test_flat = loader.flatten(X_test_norm)
    
    all_results = {}
    
    # KNN - WARNING: This is SLOW with 56,000 training samples!
    print("\n" + "="*70)
    print("EXPERIMENT 1: K-Nearest Neighbors")
    print("NOTE: KNN is slow with large datasets. Using subset for demo...")
    print("="*70)
    
    # Use smaller subset for KNN to make it feasible
    subset_size = 5000  # Use 5000 training samples instead of 56,000
    print(f"Using {subset_size} training samples for KNN (instead of {len(X_train_flat)})")
    
    knn_results = run_knn_experiments(
        X_train_flat[:subset_size], 
        y_train[:subset_size], 
        X_test_flat[:2000],  # Also limit test set
        y_test[:2000], 
        k_values=[1, 3, 5]
    )
    all_results['knn'] = {k: v['accuracy'] for k, v in knn_results.items()}
    
    # Naive Bayes
    print("\n" + "="*70)
    print("EXPERIMENT 2: Naïve Bayes")
    print("="*70)
    nb_results = run_naive_bayes_experiment(X_train_flat, y_train, X_test_flat, y_test)
    all_results['naive_bayes'] = nb_results['accuracy']
    
    # Linear (NumPy)
    print("\n" + "="*70)
    print("EXPERIMENT 3: Linear Classifier (NumPy)")
    print("="*70)
    linear_numpy_results = run_linear_experiments(X_train_flat, y_train, X_test_flat, y_test, use_pytorch=False)
    all_results['linear_numpy'] = linear_numpy_results['accuracy']
    
    # Linear (PyTorch)
    print("\n" + "="*70)
    print("EXPERIMENT 4: Linear Classifier (PyTorch)")
    print("="*70)
    linear_pytorch_results = run_linear_experiments(X_train_flat, y_train, X_test_flat, y_test, use_pytorch=True)
    all_results['linear_pytorch'] = linear_pytorch_results['accuracy']
    
    # MLP
    print("\n" + "="*70)
    print("EXPERIMENT 5: Multilayer Perceptron")
    print("="*70)
    mlp_results = run_mlp_experiment(X_train_flat, y_train, X_test_flat, y_test, hidden_dims=[256, 128], epochs=20)
    all_results['mlp'] = mlp_results['accuracy']
    
    # CNN
    print("\n" + "="*70)
    print("EXPERIMENT 6: Convolutional Neural Network")
    print("="*70)
    cnn_results = run_cnn_experiment(X_train_flat, y_train, X_test_flat, y_test, epochs=10)
    all_results['cnn'] = cnn_results['accuracy']
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print("\nTest Accuracies:")
    print(f"  KNN (k=1):              {all_results['knn'][1]:.4f}")
    print(f"  KNN (k=3):              {all_results['knn'][3]:.4f}")
    print(f"  KNN (k=5):              {all_results['knn'][5]:.4f}")
    print(f"  Naïve Bayes:            {all_results['naive_bayes']:.4f}")
    print(f"  Linear (NumPy):         {all_results['linear_numpy']:.4f}")
    print(f"  Linear (PyTorch):       {all_results['linear_pytorch']:.4f}")
    print(f"  MLP:                    {all_results['mlp']:.4f}")
    print(f"  CNN:                    {all_results['cnn']:.4f}")
    
    # Save results
    with open('results_summary.json', 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # Plot comparison
    plot_results_comparison(all_results)
    
    print("\n" + "="*70)
    print("All experiments completed!")
    print("Results saved to: results_summary.json")
    print("Comparison plot: method_comparison.png")
    print("="*70)

def plot_results_comparison(results):
    methods = ['KNN\n(k=1)', 'KNN\n(k=3)', 'KNN\n(k=5)', 
               'Naïve\nBayes', 'Linear\n(NumPy)', 'Linear\n(PyTorch)', 'MLP', 'CNN']
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
    print("✅ Saved comparison plot: method_comparison.png")

if __name__ == '__main__':
    total_start = time.time()
    main()
    total_time = time.time() - total_start
    print(f"\n⏱️  Total runtime: {total_time/60:.1f} minutes")