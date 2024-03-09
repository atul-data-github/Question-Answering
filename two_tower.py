import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Define the Siamese network with two different towers
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        # Define the query tower
        self.query_tower = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Define the document tower
        self.doc_tower = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Define the final layer to calculate the similarity
        self.fc = nn.Linear(32, 1)

    def forward(self, query, document):
        query_output = self.query_tower(query)
        doc_output = self.doc_tower(document)
        concatenated = torch.abs(query_output - doc_output)
        return self.fc(concatenated)

# Generate some random data
data_size = 1000
input_size = 10

query_data = torch.randn(data_size, input_size)
doc_data = torch.randn(data_size, input_size)
labels = torch.randint(0, 2, (data_size,), dtype=torch.float32)  # 0 for dissimilar, 1 for similar

# Split data into training and validation sets
train_size = int(0.8 * data_size)
train_query, val_query = query_data[:train_size], query_data[train_size:]
train_doc, val_doc = doc_data[:train_size], doc_data[train_size:]
train_labels, val_labels = labels[:train_size], labels[train_size:]

# Create data loaders
train_dataset = torch.utils.data.TensorDataset(train_query, train_doc, train_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(val_query, val_doc, val_labels)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

# Initialize the model, loss function, and optimizer
model = SiameseNetwork()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for query, doc, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(query, doc)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Validation loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for query, doc, labels in val_loader:
        outputs = model(query, doc)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        correct += (predicted.squeeze() == labels).sum().item()

print(f"Accuracy on validation set: {correct / total * 100:.2f}%")
