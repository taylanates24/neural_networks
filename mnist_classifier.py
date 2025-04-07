import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import json
import matplotlib.pyplot as plt
import os

# Create a directory for saving plots if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(42)

# Define device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

# Load MNIST dataset
def load_data(batch_size=64):
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Define the neural network model - Using only fully connected layers (MLP)
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        # Input layer: 28x28 = 784 input features
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        
        # Dropout for regularization
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)
    
    def forward(self, x):
        # Flatten the input: [batch_size, 1, 28, 28] -> [batch_size, 784]
        x = x.view(-1, 28 * 28)
        
        # First hidden layer
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout1(x)
        
        # Second hidden layer
        x = self.fc2(x)
        x = F.relu(x)
        # x = self.dropout1(x)
        
        # Third hidden layer
        x = self.fc3(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        
        # Output layer
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

# Training function
def train(model, train_loader, optimizer, epoch, train_losses, train_accuracies):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Calculate loss
        loss = F.nll_loss(output, target)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Accumulate loss
        running_loss += loss.item()
        
        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # Print progress
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    # Calculate average loss for this epoch
    avg_loss = running_loss / len(train_loader)
    avg_accuracy = 100. * correct / total
    
    train_losses.append(avg_loss)
    train_accuracies.append(avg_accuracy)
    
    print(f'Train Epoch: {epoch} Average Loss: {avg_loss:.6f}, Accuracy: {avg_accuracy:.2f}%')
    return avg_loss, avg_accuracy

# Testing function
def test(model, test_loader, test_losses, test_accuracies):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            
            # Get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    # Store metrics
    test_losses.append(test_loss)
    test_accuracies.append(accuracy)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

# Function to plot metrics at each epoch - overwriting the same file
def plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, epoch):
    plt.figure(figsize=(12, 8))
    
    # Create a figure with two subplots (2 rows, 1 column)
    plt.subplot(2, 1, 1)
    
    # Plot Loss on the first subplot
    epochs = range(1, epoch + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, test_losses, 'r-', label='Test Loss')
    plt.title(f'Training and Test Loss (Epoch {epoch})')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Accuracy on the second subplot
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, test_accuracies, 'g-', label='Test Accuracy')
    plt.title(f'Training and Test Accuracy (Epoch {epoch})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Save the figure (overwriting the previous one)
    plt.tight_layout()
    plt.savefig('plots/training_progress.png')
    plt.close()

# Main function
def main():
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.01
    momentum = 0.5
    epochs = 10
    
    # Lists to store metrics
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    # Load data
    train_loader, test_loader = load_data(batch_size)
    
    # Initialize model
    model = MNISTNet().to(device)
    
    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    
    # Training loop
    best_accuracy = 0
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train(model, train_loader, optimizer, epoch, train_losses, train_accuracies)
        test_accuracy = test(model, test_loader, test_losses, test_accuracies)
        
        # Save the best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), "mnist_model.pt")
            print(f"Model saved with accuracy: {best_accuracy:.2f}%")
        
        # Plot and save metrics after each epoch (overwriting the file)
        plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, epoch)
        
        # Save metrics for current epoch
        metrics = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
            'current_epoch': epoch
        }
        
        with open('training_metrics.json', 'w') as f:
            json.dump(metrics, f)
        
        print(f"Plot updated for epoch {epoch} - saved as 'plots/training_progress.png'")
    
    print("Training complete. Final plot has been saved as 'plots/training_progress.png'")

if __name__ == "__main__":
    main() 