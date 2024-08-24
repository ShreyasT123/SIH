import torch
from torch import nn, optim
import torchmetrics as met
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('Aiml/synthetic_ui_dynamics.csv')

# Encode 'user_type' as binary (human=0, bot=1)

# Features and labels
X = df.drop(['interaction_id', 'user_type'], axis=1).values
y = df['user_type'].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


class UIdynamicsDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create Dataset objects
train_dataset = UIdynamicsDataset(X_train, y_train)
test_dataset = UIdynamicsDataset(X_test, y_test)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class NN(nn.Module):
    def __init__(self) -> None:
        super(NN, self).__init__()
        self.fc1 = nn.Linear(12, 64)  # Adjust input size based on features
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 2)  # Two classes: human (0) and bot (1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return self.softmax(x)

# Initialize the Model, Loss, and Optimizer
model = NN()
criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Metrics Collection
metrics = met.MetricCollection({
    'accuracy': met.Accuracy(task='binary'),
    'precision': met.Precision(task='binary'),
    'recall': met.Recall(task='binary'),
    'f1': met.F1Score(task='binary'),
    'confusion_matrix': met.ConfusionMatrix(task='binary')
})

# Training Function
def train_model(model, train_loader, optimizer, criterion, metrics, epochs=10):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Print loss for the epoch
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader)}")

        # Validation
        model.eval()
        with torch.no_grad():
            y_true, y_pred = [], []
            for features, labels in test_loader:
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.numpy())
                y_pred.extend(predicted.numpy())
            
            y_true = torch.tensor(y_true)
            y_pred = (torch.tensor(y_pred)>=0.8).int()
            metrics_result = metrics(y_pred, y_true)
            
            # Print metrics in a readable format
            print(f"Epoch {epoch+1}/{epochs} Metrics:")
            print(f"  Accuracy: {metrics_result['accuracy'].item():.4f}")
            print(f"  Precision: {metrics_result['precision'].item():.4f}")
            print(f"  Recall: {metrics_result['recall'].item():.4f}")
            print(f"  F1 Score: {metrics_result['f1'].item():.4f}")
            print(f"  Confusion Matrix:")
            print(f"    True Negatives: {metrics_result['confusion_matrix'][0, 0]}")
            print(f"    False Positives: {metrics_result['confusion_matrix'][0, 1]}")
            print(f"    False Negatives: {metrics_result['confusion_matrix'][1, 0]}")
            print(f"    True Positives: {metrics_result['confusion_matrix'][1, 1]}")


# Train the model
train_model(model, train_loader, optimizer, criterion, metrics, epochs=10)

# Export the model to an ONNX file
