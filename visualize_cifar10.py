import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from cifar10_classifier import CIFAR10CNN, CIFAR10Net

# CIFAR-10 classes
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

# Define device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load metrics from training
def load_metrics(file_path='cifar10_metrics.json'):
    """Load training metrics from JSON file"""
    if not os.path.exists(file_path):
        print(f"Error: Metrics file '{file_path}' not found.")
        print("Please run 'cifar10_classifier.py' first to generate training metrics.")
        return None
    
    with open(file_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics

# Load the CIFAR-10 test dataset
def load_test_data(batch_size=100):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    test_dataset = datasets.CIFAR10('./data_cifar', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return test_loader

# Plot loss curves
def plot_loss_curves(metrics):
    """Plot training and test loss curves"""
    epochs = range(1, len(metrics['train_losses']) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics['train_losses'], 'b-', label='Training Loss')
    plt.plot(epochs, metrics['test_losses'], 'r-', label='Test Loss')
    plt.title('CIFAR-10: Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots_cifar/loss_curves.png')
    plt.show()

# Plot accuracy curves
def plot_accuracy_curves(metrics):
    """Plot training and test accuracy curves"""
    epochs = range(1, len(metrics['train_accuracies']) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics['train_accuracies'], 'b-', label='Training Accuracy')
    plt.plot(epochs, metrics['test_accuracies'], 'g-', label='Test Accuracy')
    plt.title('CIFAR-10: Training and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Calculate and plot the generalization gap
    gap = [train - test for train, test in zip(metrics['train_accuracies'], metrics['test_accuracies'])]
    plt.fill_between(epochs, metrics['train_accuracies'], metrics['test_accuracies'], 
                     color='red', alpha=0.2, label='Generalization Gap')
    
    plt.savefig('plots_cifar/accuracy_curves.png')
    plt.show()

# Plot combined metrics
def plot_combined_metrics(metrics):
    """Create a combined plot with loss and accuracy"""
    epochs = range(1, len(metrics['train_losses']) + 1)
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot losses on the left y-axis
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(epochs, metrics['train_losses'], 'b-', label='Training Loss')
    ax1.plot(epochs, metrics['test_losses'], 'c-', label='Test Loss')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # Create a second y-axis for accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy (%)', color='tab:red')
    ax2.plot(epochs, metrics['train_accuracies'], 'm-', label='Training Accuracy')
    ax2.plot(epochs, metrics['test_accuracies'], 'r-', label='Test Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.title('CIFAR-10 Training Metrics: Loss and Accuracy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots_cifar/combined_metrics.png')
    plt.show()

# Load trained model
def load_model(model_path='cifar10_model.pt', use_cnn=True):
    """Load the trained model"""
    if use_cnn:
        model = CIFAR10CNN()
    else:
        model = CIFAR10Net()
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"Model loaded from {model_path}")
        return model
    except:
        print(f"Error: Could not load model from {model_path}")
        return None

# Plot confusion matrix
def plot_confusion_matrix(model, test_loader):
    """Plot confusion matrix for the model's predictions on test data"""
    # Initialize confusion matrix
    conf_matrix = np.zeros((10, 10), dtype=int)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            # Update confusion matrix
            for t, p in zip(target.view(-1), predicted.view(-1)):
                conf_matrix[t.item(), p.item()] += 1
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, CIFAR10_CLASSES, rotation=45)
    plt.yticks(tick_marks, CIFAR10_CLASSES)
    
    # Add normalized values inside the boxes
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('plots_cifar/confusion_matrix.png')
    plt.show()

# Visualize sample predictions
def visualize_sample_predictions(model, test_loader, num_samples=10):
    """Visualize some sample predictions"""
    model.eval()
    
    # Get a batch of test images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs, 1)
    
    # Show images and predictions
    plt.figure(figsize=(15, 8))
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        # Convert image from tensor and denormalize
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Add color for correct/incorrect predictions
        correct = predicted[i] == labels[i]
        color = 'green' if correct else 'red'
        
        plt.imshow(img)
        plt.title(f"True: {CIFAR10_CLASSES[labels[i]]}\nPred: {CIFAR10_CLASSES[predicted[i]]}", 
                  color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('plots_cifar/sample_predictions.png')
    plt.show()

# Calculate per-class accuracy
def calculate_class_accuracy(model, test_loader):
    """Calculate and plot per-class accuracy"""
    # Initialize counters
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == target).squeeze()
            
            # Update counters
            for i in range(len(target)):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # Calculate accuracy for each class
    class_accuracy = []
    for i in range(10):
        accuracy = 100 * class_correct[i] / class_total[i]
        class_accuracy.append(accuracy)
        print(f'Accuracy of {CIFAR10_CLASSES[i]}: {accuracy:.2f}%')
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(range(10), class_accuracy, align='center')
    plt.xticks(range(10), CIFAR10_CLASSES, rotation=45)
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Accuracy on CIFAR-10')
    
    # Add values above bars
    for i, v in enumerate(class_accuracy):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig('plots_cifar/class_accuracy.png')
    plt.show()
    
    return class_accuracy

def main():
    # Make sure the plots directory exists
    os.makedirs('plots_cifar', exist_ok=True)
    
    # Load metrics
    metrics = load_metrics()
    if not metrics:
        return
    
    # Print summary
    print("\nCIFAR-10 Training Metrics Summary:")
    print(f"Number of epochs: {len(metrics['train_losses'])}")
    print(f"Final training loss: {metrics['train_losses'][-1]:.4f}")
    print(f"Final test loss: {metrics['test_losses'][-1]:.4f}")
    print(f"Final training accuracy: {metrics['train_accuracies'][-1]:.2f}%")
    print(f"Final test accuracy: {metrics['test_accuracies'][-1]:.2f}%")
    print(f"Best test accuracy: {max(metrics['test_accuracies']):.2f}%")
    
    # Plot metrics
    plot_loss_curves(metrics)
    plot_accuracy_curves(metrics)
    plot_combined_metrics(metrics)
    
    # Load model and test data
    model = load_model()
    if model:
        test_loader = load_test_data()
        
        # Generate additional visualizations
        plot_confusion_matrix(model, test_loader)
        visualize_sample_predictions(model, test_loader)
        calculate_class_accuracy(model, test_loader)
    
    print("\nVisualization complete. Plots saved in 'plots_cifar' directory")

if __name__ == "__main__":
    main() 