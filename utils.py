import numpy as np
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
