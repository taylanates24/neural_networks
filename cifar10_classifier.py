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
from tqdm import tqdm

# Create a directory for saving plots if it doesn't exist
os.makedirs('plots_cifar', exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(42)

# Define device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CIFAR-10 classes
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

# Define transformations for CIFAR-10
# CIFAR-10 normalizes using these specific mean and std values for RGB channels
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # Data augmentation
    transforms.RandomHorizontalFlip(),     # Data augmentation
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# Load CIFAR-10 dataset
def load_data(batch_size=128):
    train_dataset = datasets.CIFAR10('./data_cifar', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('./data_cifar', train=False, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

# Define a neural network model for CIFAR-10
class CIFAR10Net(nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        # Input: 32x32x3 color images
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 32 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 10)
        
        # Dropout for regularization
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.4)
    
    def forward(self, x):
        # Flatten the input: [batch_size, 3, 32, 32] -> [batch_size, 3072]
        x = x.view(-1, 32 * 32 * 3)
        
        # First hidden layer
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Second hidden layer
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Third hidden layer
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Output layer
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

# Define a CNN model for CIFAR-10 (better performance)
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten the tensor
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

# Training function
def train(model, train_loader, optimizer, epoch, train_losses, train_accuracies):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Create a tqdm progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=True)
    
    for batch_idx, (data, target) in enumerate(pbar):
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
        current_correct = pred.eq(target.view_as(pred)).sum().item()
        correct += current_correct
        total += target.size(0)
        
        # Calculate current running metrics
        current_loss = running_loss / (batch_idx + 1)
        current_acc = 100. * correct / total
        
        # Update progress bar with current metrics
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.2f}%'
        })
    
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
    
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    # Create a tqdm progress bar for test
    pbar = tqdm(test_loader, desc='Test', leave=True)
    
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Sum up batch loss
            batch_loss = F.nll_loss(output, target, reduction='sum').item()
            test_loss += batch_loss
            
            # Get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct_tensor = pred.eq(target.view_as(pred))
            batch_correct = correct_tensor.sum().item()
            correct += batch_correct
            
            # Compute accuracy for each class
            correct_tensor = correct_tensor.squeeze()
            for i in range(len(target)):
                label = target[i]
                class_correct[label] += correct_tensor[i].item()
                class_total[label] += 1
            
            # Update progress bar with current metrics
            current_loss = test_loss / ((pbar.n + 1) * target.size(0))
            current_acc = 100. * correct / ((pbar.n + 1) * target.size(0))
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    # Print per-class accuracy
    print("\nPer-class accuracy:")
    for i in range(10):
        class_acc = 100 * class_correct[i] / class_total[i]
        print(f'Accuracy of {CIFAR10_CLASSES[i]}: {class_acc:.2f}%')
    
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
    plt.title(f'CIFAR-10: Training and Test Loss (Epoch {epoch})')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Accuracy on the second subplot
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, test_accuracies, 'g-', label='Test Accuracy')
    plt.title(f'CIFAR-10: Training and Test Accuracy (Epoch {epoch})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Save the figure (overwriting the previous one)
    plt.tight_layout()
    plt.savefig('plots_cifar/training_progress.png')
    plt.close()

# Function to visualize some predictions
def visualize_predictions(model, test_loader, num_samples=5):
    model.eval()
    
    # Get a batch of test images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs, 1)
    
    # Show images and predictions
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        # Convert image from tensor and denormalize
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        plt.title(f"True: {CIFAR10_CLASSES[labels[i]]}\nPred: {CIFAR10_CLASSES[predicted[i].item()]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('plots_cifar/sample_predictions.png')
    plt.close()

# Main function
def main():
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.001
    epochs = 25
    
    # Lists to store metrics
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = load_data(batch_size)
    
    # Initialize model - choose between MLP and CNN
    #model = CIFAR10Net().to(device)  # MLP model - simpler but less accurate
    model = CIFAR10CNN().to(device)  # CNN model - better performance
    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Define optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    
    # Training loop
    best_accuracy = 0
    for epoch in range(1, epochs + 1):
        print(f"\n{'='*60}\nEpoch {epoch}/{epochs}")
        train_loss, train_accuracy = train(model, train_loader, optimizer, epoch, train_losses, train_accuracies)
        test_accuracy = test(model, test_loader, test_losses, test_accuracies)
        
        # Adjust learning rate based on validation loss
        scheduler.step(test_losses[-1])
        
        # Save the best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), "cifar10_model.pt")
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
        
        with open('cifar10_metrics.json', 'w') as f:
            json.dump(metrics, f)
        
        # Visualize predictions 
        if epoch % 5 == 0 or epoch == epochs:
            visualize_predictions(model, test_loader)
        
        # Summary for this epoch
        print(f"Epoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"  Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        print(f"  Best Test Accuracy: {best_accuracy:.2f}%")
            
    print("\n" + "="*80)
    print("Training complete. Final plot has been saved as 'plots_cifar/training_progress.png'")
    print(f"Final metrics: Train accuracy: {train_accuracies[-1]:.2f}%, Test accuracy: {test_accuracies[-1]:.2f}%")
    print(f"Best test accuracy: {best_accuracy:.2f}%")


def objective(trial, cfg_path, lr_min, lr_max, ml_min, ml_max, csv_file, seed, print_method, print_rank):
    # Suggest a learning rate using the updated method
    lr = trial.suggest_float('lr', lr_min, lr_max, log=True)
    ml = trial.suggest_int('ml', ml_min, ml_max, step=2)
    # Load the configuration with the suggested learning rate
    cfg = YAMLConfig(cfg_path, hp_tuning=True, lr=lr, ml=ml)
    
    print(f"lr: {lr}")
    print(f"ml: {ml}")
    # Create a deep copy of the config to modify
    current_cfg = copy.deepcopy(cfg)
    
    # Access the internal dictionary or attribute of YAMLConfig
    config_dict = current_cfg.__dict__

    # Update the learning rate in the optimizer config
    dist_utils.setup_distributed(print_rank, print_method, seed=seed)
    # Initialize the solver with the updated config
    solver = DetSolver(current_cfg)
    
    # Train the model and get the ap_50 metric
    ap_50 = solver.fit()

    # Save the hyperparameters and results
    save_hyperparameters_and_results(trial, ap_50, csv_file)

    return ap_50 


def hyperparameter_tuning():

    pass

    
    
    
    
if __name__ == "__main__":
    hyperparameter_tuning() 