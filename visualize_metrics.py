import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_metrics(file_path='training_metrics.json'):
    """Load training metrics from JSON file"""
    if not os.path.exists(file_path):
        print(f"Error: Metrics file '{file_path}' not found.")
        print("Please run 'mnist_classifier.py' first to generate training metrics.")
        return None
    
    with open(file_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics

def plot_loss_curves(metrics):
    """Plot training and test loss curves"""
    epochs = range(1, len(metrics['train_losses']) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metrics['train_losses'], 'b-', label='Training Loss')
    plt.plot(epochs, metrics['test_losses'], 'r-', label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curves.png')
    plt.show()

def plot_accuracy_curve(metrics):
    """Plot training and test accuracy curves"""
    epochs = range(1, len(metrics['test_accuracies']) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metrics['train_accuracies'], 'b-', label='Training Accuracy')
    plt.plot(epochs, metrics['test_accuracies'], 'g-', label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    # Add horizontal lines for accuracy boundaries
    plt.axhline(y=95, color='r', linestyle='--', alpha=0.3)
    plt.axhline(y=97, color='b', linestyle='--', alpha=0.3)
    # Add text annotations for the horizontal lines
    plt.text(1, 95.5, '95%', color='r')
    plt.text(1, 97.5, '97%', color='b')
    plt.ylim(90, 100)  # Adjust y-axis to focus on the relevant range
    plt.savefig('accuracy_curve.png')
    plt.show()

def combined_plot(metrics):
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
    
    plt.title('Training Metrics: Loss and Accuracy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('combined_metrics.png')
    plt.show()

def main():
    # Load metrics
    metrics = load_metrics()
    if not metrics:
        return
    
    # Print summary
    print("\nTraining Metrics Summary:")
    print(f"Number of epochs: {len(metrics['train_losses'])}")
    print(f"Final training loss: {metrics['train_losses'][-1]:.4f}")
    print(f"Final test loss: {metrics['test_losses'][-1]:.4f}")
    print(f"Final training accuracy: {metrics['train_accuracies'][-1]:.2f}%")
    print(f"Final test accuracy: {metrics['test_accuracies'][-1]:.2f}%")
    print(f"Best test accuracy: {max(metrics['test_accuracies']):.2f}%")
    
    # Plot individual graphs
    plot_loss_curves(metrics)
    plot_accuracy_curve(metrics)
    
    # Plot combined graph
    combined_plot(metrics)
    
    print("\nPlots saved as 'loss_curves.png', 'accuracy_curve.png', and 'combined_metrics.png'")

if __name__ == "__main__":
    main() 