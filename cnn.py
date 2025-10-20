import torch
import torch.nn as nn
import torch.optim as optim

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
    
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_cnn(model, X_train, y_train, X_val, y_val, epochs=30, lr=0.001, batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
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
        num_batches = 0
        
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
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
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

def run_cnn_experiment(X_train, y_train, X_test, y_test, epochs=30):
    from utils import compute_metrics, plot_confusion_matrix
    
    print(f"\n{'='*50}")
    print("Running CNN")
    print(f"{'='*50}")
    
    model = CNN(num_classes=10)
    model, loss_history, val_acc_history = train_cnn(model, X_train, y_train, X_test, y_test, epochs=epochs)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(device)
        predictions = model(X_test_t).argmax(1).cpu().numpy()
    
    accuracy, report = compute_metrics(y_test, predictions)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    plot_confusion_matrix(y_test, predictions, title='CNN Confusion Matrix', save_path='cnn_confusion.png')
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'report': report,
        'loss_history': loss_history,
        'val_acc_history': val_acc_history,
        'model': model
    }
