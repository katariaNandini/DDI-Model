import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import numpy as np
from model import DDIMDL
from generate_dummy_data import generate_dummy_dataset, create_drug_pairs
import matplotlib.pyplot as plt

def train_model(model, train_pairs, train_labels, val_pairs, val_labels, 
                num_epochs=50, batch_size=32, learning_rate=0.001):
    """Train the DDIMDL model."""
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    # Training history
    train_losses = []
    val_losses = []
    val_aucs = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        # Process in batches
        for i in range(0, len(train_pairs), batch_size):
            batch_pairs = train_pairs[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            batch_preds = []
            for drug1, drug2 in batch_pairs:
                pred = model(drug1, drug2)
                batch_preds.append(pred)
            
            batch_preds = torch.cat(batch_preds)
            
            # Calculate loss
            loss = criterion(batch_preds, batch_labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        # Calculate average training loss
        avg_train_loss = total_loss / batch_count
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss, val_auc = evaluate_model(model, val_pairs, val_labels)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation AUC: {val_auc:.4f}")
        print("-" * 50)
    
    return train_losses, val_losses, val_aucs

def evaluate_model(model, val_pairs, val_labels):
    """Evaluate the model on validation/test data."""
    model.eval()
    criterion = nn.BCELoss()
    
    with torch.no_grad():
        val_preds = []
        total_loss = 0
        
        for drug1, drug2 in val_pairs:
            pred = model(drug1, drug2)
            val_preds.append(pred)
        
        val_preds = torch.cat(val_preds)
        val_loss = criterion(val_preds, val_labels).item()
        val_auc = roc_auc_score(val_labels.numpy(), val_preds.numpy())
        
    return val_loss, val_auc

def plot_metrics(train_losses, val_losses, val_aucs):
    """Plot training metrics."""
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot AUC
    plt.subplot(1, 2, 2)
    plt.plot(val_aucs)
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Validation AUC')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

if __name__ == "__main__":
    # Generate dummy dataset
    print("Generating dummy dataset...")
    dataset = generate_dummy_dataset(n_drugs=100)
    pairs, labels = create_drug_pairs(dataset)
    
    # Split data into train/val/test sets
    train_idx, test_idx = train_test_split(range(len(pairs)), test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)
    
    # Prepare data splits
    train_pairs = [pairs[i] for i in train_idx]
    train_labels = labels[train_idx]
    val_pairs = [pairs[i] for i in val_idx]
    val_labels = labels[val_idx]
    test_pairs = [pairs[i] for i in test_idx]
    test_labels = labels[test_idx]
    
    # Initialize model
    model = DDIMDL(
        chemical_dim=5,  # Number of chemical features
        protein_dim=10,  # Protein node feature dimension
        enzyme_dim=5,    # Number of enzyme types
        pathway_dim=8,   # Pathway feature dimension
        hidden_dim=64    # Hidden dimension for all encoders
    )
    
    # Train model
    print("Starting training...")
    train_losses, val_losses, val_aucs = train_model(
        model, train_pairs, train_labels, val_pairs, val_labels,
        num_epochs=50,
        batch_size=32,
        learning_rate=0.001
    )
    
    # Plot training metrics
    plot_metrics(train_losses, val_losses, val_aucs)
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_auc = evaluate_model(model, test_pairs, test_labels)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'ddimdl_model.pth') 