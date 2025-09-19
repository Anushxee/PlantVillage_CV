import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import json
import os

# Parameters
num_classes = 12
batch_size = 16
num_epochs = 5   # increase as needed
learning_rate = 1e-4
dataset_path = "dataset/"  # folder containing 12-class dataset

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load old model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)

state_dict = torch.load("model/model.pth", map_location=device)
state_dict.pop('fc.weight', None)
state_dict.pop('fc.bias', None)
model.load_state_dict(state_dict, strict=False)
model = model.to(device)

# Data transforms
data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Dataset & loader
train_dataset = datasets.ImageFolder(root=dataset_path, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print(f"Number of training images: {len(train_dataset)}")
print(f"Classes found: {train_dataset.classes}")


# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Save updated model
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/model_12class.pth")
print("Updated 12-class model saved as model_12class.pth")
