import numpy as np

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
        # Convert images to binary
        X_binary = self.binarize(X_train)
        
        self.classes = np.unique(y_train)
        n_classes = len(self.classes)
        n_features = X_binary.shape[1]
        
        self.feature_probs = np.zeros((n_classes, n_features))
        self.class_priors = np.zeros(n_classes)
        
        # Calculate probabilities for each digit class
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
    
    print(f"\n{'='*50}")
    print("Running Naïve Bayes Classifier")
    print(f"{'='*50}")
    
    nb = NaiveBayes(threshold=0.5, smoothing=1e-10)
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)
    
    accuracy, report = compute_metrics(y_test, predictions)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
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
