import numpy as np
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
        # Calculate distance from x to all training images
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        # Find the k closest images
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        # Picks the most common label
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
        print(f"\n{'='*50}")
        print(f"Running KNN with k={k}")
        print(f"{'='*50}")
        
        knn = KNN(k=k)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        
        accuracy, report = compute_metrics(y_test, predictions)
        
        print(f"\nAccuracy: {accuracy:.4f}")
        print("\nClassification Report:")
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
