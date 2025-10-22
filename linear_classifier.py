import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class LinearClassifierNumPy:
    def __init__(self, input_dim=784, output_dim=10, lr=0.01, epochs=100):
        self.W = np.random.randn(output_dim, input_dim) * 0.01
        self.b = np.zeros(output_dim)
        self.lr = lr
        self.epochs = epochs
        self.loss_history = []
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        return X @ self.W.T + self.b
    
    def compute_loss(self, y_pred, y_true):
        n = y_true.shape[0]
        y_one_hot = np.zeros_like(y_pred)
        y_one_hot[np.arange(n), y_true.astype(int)] = 1
        loss = np.sum((y_pred - y_one_hot) ** 2) / (2 * n)
        return loss
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        n = X_train.shape[0]
        
        for epoch in range(self.epochs):
            logits = self.forward(X_train)
            loss = self.compute_loss(logits, y_train)
            self.loss_history.append(loss)
            
            y_one_hot = np.zeros_like(logits)
            y_one_hot[np.arange(n), y_train.astype(int)] = 1
            
            d_logits = (logits - y_one_hot) / n
            d_W = d_logits.T @ X_train
            d_b = np.sum(d_logits, axis=0)
            
            self.W -= self.lr * d_W
            self.b -= self.lr * d_b
            
            if (epoch + 1) % 10 == 0:
                train_acc = np.mean(self.predict(X_train) == y_train)
                val_acc = np.mean(self.predict(X_val) == y_val) if X_val is not None else 0
                print(f'Epoch {epoch+1}/{self.epochs}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    
    def predict(self, X):
        logits = self.forward(X)
        return np.argmax(logits, axis=1)

class LinearClassifierPyTorch(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super(LinearClassifierPyTorch, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

def train_pytorch_linear(model, X_train, y_train, X_val, y_val, epochs=100, lr=0.001, batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train_t.size(0))
        epoch_loss = 0
        
        for i in range(0, X_train_t.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_X = X_train_t[indices]
            batch_y = y_train_t[indices]
            
            batch_y_one_hot = torch.zeros(batch_y.size(0), 10).to(device)
            batch_y_one_hot.scatter_(1, batch_y.unsqueeze(1), 1)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y_one_hot)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / (X_train_t.size(0) / batch_size)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                train_outputs = model(X_train_t)
                train_acc = (train_outputs.argmax(1) == y_train_t).float().mean().item()
                val_outputs = model(X_val_t)
                val_acc = (val_outputs.argmax(1) == y_val_t).float().mean().item()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    
    return model, loss_history

def run_linear_experiments(X_train, y_train, X_test, y_test, use_pytorch=False):
    from utils import compute_metrics, plot_confusion_matrix, visualize_weights
    
    print(f"\n{'='*50}")
    print(f"Running Linear Classifier ({'PyTorch' if use_pytorch else 'NumPy'})")
    print(f"{'='*50}")
    
    if use_pytorch:
        model = LinearClassifierPyTorch(input_dim=784, output_dim=10)
        model, loss_history = train_pytorch_linear(model, X_train, y_train, X_test, y_test, epochs=200, lr=0.01)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(device)
            predictions = model(X_test_t).argmax(1).cpu().numpy()
        weights = model.linear.weight.detach().cpu().numpy()
    else:
        model = LinearClassifierNumPy(lr=0.01, epochs=200)
        model.fit(X_train, y_train, X_test, y_test)
        predictions = model.predict(X_test)
        weights = model.W
        loss_history = model.loss_history
    
    accuracy, report = compute_metrics(y_test, predictions)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    plot_confusion_matrix(y_test, predictions, title='Linear Classifier Confusion Matrix', save_path='linear_confusion.png')
    visualize_weights(weights, title='Linear Classifier Weights', save_path='linear_weights.png')
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'report': report,
        'weights': weights,
        'loss_history': loss_history
    }
