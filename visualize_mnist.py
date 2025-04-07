import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from mnist_classifier import MNISTNet

def load_model():
    """Load the trained model"""
    model = MNISTNet()
    model.load_state_dict(torch.load('mnist_model.pt', map_location=torch.device('cpu')))
    model.eval()
    return model

def get_test_data():
    """Load the MNIST test dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, transform=transform, download=True)
    return test_dataset

def visualize_predictions(model, dataset, num_samples=10):
    """Visualize model predictions on random test samples"""
    # Get random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        # Get the image and label
        img, label = dataset[idx]
        
        # Get model prediction
        with torch.no_grad():
            output = model(img.unsqueeze(0))
            _, predicted = torch.max(output.data, 1)
        
        # Display image
        img_display = img.squeeze().numpy()
        axes[i].imshow(img_display, cmap='gray')
        
        # Set title with true label and prediction
        title = f"True: {label}, Pred: {predicted.item()}"
        color = 'green' if label == predicted.item() else 'red'
        axes[i].set_title(title, color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_predictions.png')
    plt.show()

def plot_sample_digits(dataset, num_samples=10):
    """Plot sample digits from the dataset"""
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        img_display = img.squeeze().numpy()
        axes[i].imshow(img_display, cmap='gray')
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_samples.png')
    plt.show()

def main():
    # Load test dataset
    test_dataset = get_test_data()
    
    # Plot sample digits
    print("Plotting sample MNIST digits...")
    plot_sample_digits(test_dataset)
    
    try:
        # Load trained model
        model = load_model()
        
        # Visualize predictions
        print("Visualizing model predictions...")
        visualize_predictions(model, test_dataset)
    except FileNotFoundError:
        print("Model file 'mnist_model.pt' not found. Please train the model first using 'mnist_classifier.py'.")

if __name__ == "__main__":
    main() 