import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[256, 128], output_dim=10, dropout=0.2):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def train_mlp(model, X_train, y_train, X_val, y_val, epochs=50, lr=0.001, batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    val_acc_history = []
    
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train_t.size(0))
        epoch_loss = 0
        
        for i in range(0, X_train_t.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_X = X_train_t[indices]
            batch_y = y_train_t[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / (X_train_t.size(0) / batch_size)
        loss_history.append(avg_loss)
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_acc = (val_outputs.argmax(1) == y_val_t).float().mean().item()
            val_acc_history.append(val_acc)
            train_outputs = model(X_train_t)
            train_acc = (train_outputs.argmax(1) == y_train_t).float().mean().item()
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    
    return model, loss_history, val_acc_history

def run_mlp_experiment(X_train, y_train, X_test, y_test, hidden_dims=[256, 128], epochs=50):
    from utils import compute_metrics, plot_confusion_matrix
    
    print(f"\n{'='*50}")
    print(f"Running MLP (hidden layers: {hidden_dims})")
    print(f"{'='*50}")
    
    model = MLP(input_dim=784, hidden_dims=hidden_dims, output_dim=10)
    model, loss_history, val_acc_history = train_mlp(model, X_train, y_train, X_test, y_test, epochs=epochs)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(device)
        predictions = model(X_test_t).argmax(1).cpu().numpy()
    
    accuracy, report = compute_metrics(y_test, predictions)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    plot_confusion_matrix(y_test, predictions, title='MLP Confusion Matrix', save_path='mlp_confusion.png')
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'report': report,
        'loss_history': loss_history,
        'val_acc_history': val_acc_history
    }
