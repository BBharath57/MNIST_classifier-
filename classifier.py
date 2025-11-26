import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------
# 1. Configuration & Device Setup
# -----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 25

# -----------------------------------------
# 2. Data Preparation
# -----------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("Downloading Data...")
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)


# -----------------------------------------
# 3. Model Definition
# -----------------------------------------
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# Initialize Model
model = NeuralNet().to(device)

# -----------------------------------------
# 4. Loss and Optimizer
# -----------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# -----------------------------------------
# 5. Training Loop
# -----------------------------------------
def train_one_epoch(epoch_index):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)

        scores = model(data)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch_index} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")


def check_accuracy(loader, model):
    model.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    acc = float(num_correct) / float(num_samples) * 100
    print(f"Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    print("\nStarting Training...")
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(epoch)
        check_accuracy(test_loader, model)

    print("\nTraining Complete!")

    # --- SAVE THE MODEL ---
    save_path = "mnist_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")