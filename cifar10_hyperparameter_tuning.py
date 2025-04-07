import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import csv
import copy
import optuna
import argparse
import logging
import pickle
import sqlite3
from sqlalchemy import create_engine
from datetime import datetime
from tqdm import tqdm

# Import model definitions from main script
from cifar10_classifier import CIFAR10CNN, CIFAR10_CLASSES, device

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Set random seed for reproducibility
torch.manual_seed(42)

# Define transformations for CIFAR-10
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
def load_data(batch_size):
    train_dataset = datasets.CIFAR10('./data_cifar', train=True, download=True, transform=transform_train)
    
    # Split training set into train and validation (80/20 split)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader

# Training function for a single epoch
def train_epoch(model, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Create a tqdm progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} Train', leave=True)
    
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
        correct += pred.eq(target.view_as(pred)).sum().item()
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
    
    return avg_loss, avg_accuracy

# Validation function
def validate(model, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    # Create a tqdm progress bar for validation
    pbar = tqdm(val_loader, desc='Validation', leave=True)
    
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Sum up batch loss
            batch_loss = F.nll_loss(output, target, reduction='sum').item()
            val_loss += batch_loss
            
            # Get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            batch_correct = pred.eq(target.view_as(pred)).sum().item()
            correct += batch_correct
            total += target.size(0)
            
            # Update progress bar with current metrics
            current_loss = val_loss / ((pbar.n + 1) * target.size(0))
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })
    
    val_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / total
    
    return val_loss, accuracy

def restore_database(dump_path, db_path):
    """Restore the database from a SQL dump file."""
    con = sqlite3.connect(db_path)
    with open(dump_path, 'r') as f:
        sql_script = f.read()
    con.executescript(sql_script)
    con.close()
    print(f"Database restored from {dump_path} to {db_path}")

def get_study_name(db_path):
    """Get the study name from the database."""
    con = sqlite3.connect(db_path)
    cursor = con.cursor()
    cursor.execute("SELECT study_name FROM studies")
    study_name = cursor.fetchone()[0]
    con.close()
    return study_name

def dump_database(db_path, dump_path, study, sampler_path):
    """Dump the database to a SQL file and save the sampler."""
    # Ensure the directory for the database path exists
    db_dir = os.path.dirname(db_path)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)

    con = sqlite3.connect(db_path)
    with open(dump_path, 'w') as f:
        for line in con.iterdump():
            f.write(f"{line}\n")
    con.close()

    with open(sampler_path, "wb") as fout:
        pickle.dump(study.sampler, fout)

    print(f"Database dumped to {dump_path}")

def dump_database_callback(study, trial, db_path, dump_path, sampler_path):
    """Callback function to dump the database after each trial."""
    # Dump the database
    con = sqlite3.connect(db_path)
    with open(dump_path, 'w') as f:
        for line in con.iterdump():
            f.write(f"{line}\n")
    con.close()

    # Dump the sampler
    with open(sampler_path, "wb") as fout:
        pickle.dump(study.sampler, fout)

    print(f"Database and sampler dumped to {dump_path} and {sampler_path}")

def save_hyperparameters_and_results(trial, accuracy, csv_file):
    """Save the hyperparameters and their corresponding accuracy to a CSV file."""
    # Define the fieldnames for the CSV file
    fieldnames = ['trial_number', 'batch_size', 'learning_rate', 'accuracy']
    
    # Check if the file exists to determine if we need to write the header
    file_exists = os.path.exists(csv_file)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    # Open the CSV file in append mode
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # Write the header if the file does not exist
        if not file_exists:
            writer.writeheader()
        
        # Write the current trial's data
        writer.writerow({
            'trial_number': trial.number,
            'batch_size': trial.params['batch_size'],
            'learning_rate': trial.params['learning_rate'],
            'accuracy': accuracy
        })

def objective(trial, bs_min, bs_max, lr_min, lr_max, csv_file, seed):
    """The objective function for Optuna to optimize."""
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Suggest hyperparameters to try
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    lr = trial.suggest_float('learning_rate', lr_min, lr_max, log=True)
    
    print(f"Trial {trial.number}: batch_size={batch_size}, learning_rate={lr}")
    
    # Load data with the suggested batch size
    train_loader, val_loader = load_data(batch_size)
    
    # Create model and optimizer
    model = CIFAR10CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    num_epochs = 2  # Train for 2 epochs per trial
    best_val_accuracy = 0
    
    for epoch in range(1, num_epochs + 1):
        # Train and validate
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, epoch)
        val_loss, val_accuracy = validate(model, val_loader)
        
        # Report metrics to Optuna
        trial.report(val_accuracy, epoch)
        
        # Log results
        logging.info(f"Trial {trial.number}, Epoch {epoch}, Batch Size: {batch_size}, "
                    f"LR: {lr:.6f}, Train Loss: {train_loss:.4f}, "
                    f"Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_accuracy:.2f}%")
        
        # Keep track of the best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
        
        # Handle pruning if trial should be pruned
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    # Save results to CSV
    save_hyperparameters_and_results(trial, best_val_accuracy, csv_file)
    
    return best_val_accuracy

def tune_hyperparameters(bs_min, bs_max, lr_min, lr_max, out_path, n_trials=50, start_trials=10, seed=42, db_path=None, sampler_path=None, resume=False):
    """Main function to run hyperparameter tuning with Optuna."""
    # Create a unique directory for each experiment using the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use the existing database for the study if resuming
    if resume and db_path:
        study_name = get_study_name(db_path)
        database_url = f'sqlite:///{db_path}'
        dump_path = f"{study_name}.sql"
        sampler_path = f"{study_name}_sampler.pkl"
        experiment_path = os.path.dirname(db_path)
        csv_file = os.path.join(experiment_path, "hyperparameter_results.csv")
    else:
        experiment_path = os.path.join(out_path, f"experiment_{timestamp}")
        os.makedirs(experiment_path, exist_ok=True)
        
        # Define a unique CSV filename for the session
        csv_file = os.path.join(experiment_path, "hyperparameter_results.csv")
        study_name = f'{experiment_path}/cifar10_study'
        database_url = f'sqlite:///{study_name}.db'
        
        dump_path = f"{study_name}.sql"
        sampler_path = f"{study_name}_sampler.pkl"
    
    # Create storage engine
    storage = optuna.storages.RDBStorage(
        url=database_url,
        engine_kwargs={
            'connect_args': {
                'check_same_thread': False
            }
        }
    )
    
    # Define the sampler
    if resume and sampler_path and os.path.exists(sampler_path):
        with open(sampler_path, "rb") as fin:
            sampler = pickle.load(fin)
    else:
        sampler = optuna.samplers.TPESampler(       
            n_startup_trials=start_trials,
            multivariate=True,
            seed=seed
        )
    
    # Create or load an Optuna study
    if resume and db_path and os.path.exists(db_path):
        study = optuna.load_study(study_name=study_name, storage=storage, sampler=sampler)
        logging.info(f"Resuming study '{study_name}' with {len(study.trials)} previous trials")
    else:
        study = optuna.create_study(
            direction="maximize",  # We want to maximize validation accuracy
            storage=storage, 
            sampler=sampler, 
            study_name=study_name, 
            load_if_exists=resume,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
        )
        logging.info(f"Created new study '{study_name}'")
    
    # Enqueue some predefined trials
    # These are good starting points that we want to explicitly evaluate
    study.enqueue_trial({'batch_size': 32, 'learning_rate': 0.001})
    study.enqueue_trial({'batch_size': 64, 'learning_rate': 0.001})
    study.enqueue_trial({'batch_size': 128, 'learning_rate': 0.001})
    study.enqueue_trial({'batch_size': 256, 'learning_rate': 0.001})
    study.enqueue_trial({'batch_size': 32, 'learning_rate': 0.0001})
    study.enqueue_trial({'batch_size': 64, 'learning_rate': 0.0001})
    study.enqueue_trial({'batch_size': 128, 'learning_rate': 0.0001})
    study.enqueue_trial({'batch_size': 256, 'learning_rate': 0.0001})
    study.enqueue_trial({'batch_size': 32, 'learning_rate': 0.00001})
    study.enqueue_trial({'batch_size': 64, 'learning_rate': 0.00001})
    study.enqueue_trial({'batch_size': 128, 'learning_rate': 0.00001})
    study.enqueue_trial({'batch_size': 256, 'learning_rate': 0.00001})      

    
    
    logging.info(f"Starting Optuna study with {n_trials} trials (2 epochs each)...")
    
    # Optimize the study with a callback to dump the database
    study.optimize(
        lambda trial: objective(trial, bs_min, bs_max, lr_min, lr_max, csv_file, seed),
        n_trials=n_trials,
        callbacks=[lambda study, trial: dump_database_callback(study, trial, f'{study_name}.db', dump_path, sampler_path)]
    )
    
    # Get the best trial
    best_trial = study.best_trial
    logging.info(f"Best Batch Size: {best_trial.params['batch_size']}, Best Learning Rate: {best_trial.params['learning_rate']}, "
                f"Best Validation Accuracy: {best_trial.value:.2f}%")
    
    # Save the best trial results to a JSON file
    best_result = {
        'best_batch_size': best_trial.params['batch_size'],
        'best_learning_rate': best_trial.params['learning_rate'],
        'best_accuracy': best_trial.value
    }
    json_file = os.path.join(experiment_path, f"best_hyperparameters.json")
    with open(json_file, 'w') as f:
        json.dump(best_result, f, indent=4)
    
    # Visualize results
    try:
        visualize_optuna_results(study, experiment_path)
    except Exception as e:
        logging.error(f"Error creating visualizations: {e}")
    
    return best_trial.params['batch_size'], best_trial.params['learning_rate'], best_trial.value

def visualize_optuna_results(study, output_dir):
    """Create and save visualizations for Optuna study results."""
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot optimization history
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title('Optimization History')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optimization_history.png'))
    
    # Plot parameter importances
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title('Parameter Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'param_importances.png'))
    
    # Plot parallel coordinate plot
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.title('Parallel Coordinate')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parallel_coordinate.png'))
    
    # Plot slice plot
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_slice(study)
    plt.title('Slice Plot')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'slice_plot.png'))
    
    # Plot contour plot
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_contour(study)
    plt.title('Contour Plot')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'contour_plot.png'))
    
    plt.close('all')

def train_with_best_params(batch_size, learning_rate, output_dir, num_epochs=10, seed=42):
    """Train a model with the best hyperparameters found by Optuna."""
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    logging.info(f"Training final model with best parameters: Batch Size = {batch_size}, Learning Rate = {learning_rate}")
    
    # Load data with the best batch size
    train_loader, val_loader = load_data(batch_size)
    
    # Create model and optimizer
    model = CIFAR10CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
    
    # Lists to store metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Training loop
    best_val_accuracy = 0
    for epoch in range(1, num_epochs + 1):
        # Train and validate
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, epoch)
        val_loss, val_accuracy = validate(model, val_loader)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        # Adjust learning rate
        scheduler.step(val_loss)
        
        # Log results
        logging.info(f"Final Model - Epoch {epoch}/{num_epochs}, "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(output_dir, "cifar10_model_best.pt"))
            logging.info(f"Saved best model with validation accuracy: {best_val_accuracy:.2f}%")
    
    # Plot the final metrics
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, 'r-', label='Validation Loss')
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_model_metrics.png'))
    plt.close()
    
    # Save metrics to file
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_batch_size': batch_size,
        'best_learning_rate': learning_rate,
        'best_val_accuracy': best_val_accuracy
    }
    
    with open(os.path.join(output_dir, 'final_metrics.json'), 'w') as f:
        json.dump(metrics, f)
    
    logging.info(f"Final training completed. Best validation accuracy: {best_val_accuracy:.2f}%")
    logging.info(f"Model saved as '{os.path.join(output_dir, 'cifar10_model_best.pt')}'")
    logging.info(f"Metrics saved to '{os.path.join(output_dir, 'final_metrics.json')}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter tuning with Optuna for CIFAR-10')
    parser.add_argument('--bs_min', type=int, default=32, help='Minimum batch size for tuning')
    parser.add_argument('--bs_max', type=int, default=256, help='Maximum batch size for tuning')
    parser.add_argument('--lr_min', type=float, default=1e-4, help='Minimum learning rate for tuning')
    parser.add_argument('--lr_max', type=float, default=1e-1, help='Maximum learning rate for tuning')
    parser.add_argument('--n_trials', type=int, default=200, help='Number of trials for Optuna optimization')
    parser.add_argument('--start_trials', type=int, default=50, help='Number of random trials before TPE kicks in')
    parser.add_argument('--out_path', type=str, default='optuna_results', help='Output path for database storage')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--db_path', type=str, default=None, help='Path to existing database file for resuming')
    parser.add_argument('--sampler_path', type=str, default=None, help='Path to existing sampler file for resuming')
    parser.add_argument('--resume', action='store_true', help='Flag to resume from an existing study')
    parser.add_argument('--train_final', action='store_true', help='Train final model with best parameters')
    parser.add_argument('--final_epochs', type=int, default=10, help='Number of epochs for final model training')
    args = parser.parse_args()
    
    print(f"Running hyperparameter optimization on device: {device}")
    
    # Run hyperparameter tuning
    best_bs, best_lr, best_accuracy = tune_hyperparameters(
        args.bs_min, args.bs_max, args.lr_min, args.lr_max, 
        args.out_path, args.n_trials, args.start_trials, args.seed,
        args.db_path, args.sampler_path, args.resume
    )
    
    print(f"\nBest hyperparameters:")
    print(f"  Batch Size: {best_bs}")
    print(f"  Learning Rate: {best_lr}")
    print(f"  Validation Accuracy: {best_accuracy:.2f}%")
    
    # Train final model if requested
    if args.train_final:
        experiment_dir = os.path.join(args.out_path, sorted(os.listdir(args.out_path))[-1])
        train_with_best_params(best_bs, best_lr, experiment_dir, args.final_epochs, args.seed)
    else:
        train_final = input("\nTrain final model with best parameters? (y/n): ").strip().lower()
        if train_final == 'y':
            num_epochs = int(input("Number of epochs to train (default=10): ") or 10)
            experiment_dir = os.path.join(args.out_path, sorted(os.listdir(args.out_path))[-1])
            train_with_best_params(best_bs, best_lr, experiment_dir, num_epochs, args.seed) 