"""
Training infrastructure for autoencoders.

This module provides functions for training deep autoencoders with:
- Early stopping based on validation loss
- Training curve logging
- Model checkpointing
- Progress tracking
- Hardware and timing information
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, verbose: bool = True):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum change to qualify as improvement
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss: float, epoch: int) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss
            epoch: Current epoch number

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                improvement = self.best_loss - val_loss
                print(f"  Validation loss improved by {improvement:.6f}")
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0

        return self.early_stop


def create_dataloaders(
    X_train: np.ndarray,
    X_val: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders from numpy arrays.

    Args:
        X_train: Training features
        X_val: Validation features
        batch_size: Batch size for training
        shuffle: Whether to shuffle training data

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_val_tensor = torch.FloatTensor(X_val)

    # Create datasets (autoencoder: input = target)
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, X_val_tensor)

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 to avoid issues on Windows
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader


def train_autoencoder(
    model: nn.Module,
    X_train: np.ndarray,
    X_val: np.ndarray,
    device: torch.device,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 100,
    patience: int = 10,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train an autoencoder with early stopping.

    Args:
        model: Autoencoder model (DeepAutoencoder or DenoisingAutoencoder)
        X_train: Training features (normal transactions only!)
        X_val: Validation features (includes both normal and fraud)
        device: Device to train on
        learning_rate: Learning rate for Adam optimizer
        batch_size: Batch size
        max_epochs: Maximum number of epochs
        patience: Early stopping patience
        save_path: Path to save best model (optional)
        verbose: Print progress

    Returns:
        Dictionary containing:
            - 'model': Trained model
            - 'train_losses': List of training losses per epoch
            - 'val_losses': List of validation losses per epoch
            - 'best_epoch': Epoch with best validation loss
            - 'total_epochs': Total epochs trained
            - 'training_time': Total training time in seconds
            - 'param_count': Number of trainable parameters
            - 'device': Device used for training
    """
    # Move model to device
    model = model.to(device)

    # Setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=patience, verbose=verbose)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        X_train, X_val, batch_size=batch_size
    )

    # Training history
    train_losses = []
    val_losses = []

    # Count parameters
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print("\n" + "=" * 60)
        print("TRAINING AUTOENCODER")
        print("=" * 60)
        print(f"Model: {model.__class__.__name__}")
        print(f"Device: {device}")
        print(f"Trainable parameters: {param_count:,}")
        print(f"Training samples: {len(X_train):,}")
        print(f"Validation samples: {len(X_val):,}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Max epochs: {max_epochs}")
        print("=" * 60)

    # Training loop
    start_time = time.time()
    best_model_state = None

    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        if verbose:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Train]")
        else:
            pbar = train_loader

        for batch_x, batch_target in pbar:
            batch_x = batch_x.to(device)
            batch_target = batch_target.to(device)

            # Forward pass
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_target)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            if verbose and isinstance(pbar, tqdm):
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_x, batch_target in val_loader:
                batch_x = batch_x.to(device)
                batch_target = batch_target.to(device)

                reconstructed = model(batch_x)
                loss = criterion(reconstructed, batch_target)

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)

        if verbose:
            print(f"\nEpoch {epoch+1}/{max_epochs}:")
            print(f"  Train Loss: {avg_train_loss:.6f}")
            print(f"  Val Loss:   {avg_val_loss:.6f}")

        # Early stopping check
        if early_stopping(avg_val_loss, epoch):
            if verbose:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best epoch was {early_stopping.best_epoch+1}")
            break

        # Save best model
        if avg_val_loss == early_stopping.best_loss:
            best_model_state = model.state_dict().copy()

    total_time = time.time() - start_time

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save model if path provided
    if save_path:
        torch.save(model.state_dict(), save_path)
        if verbose:
            print(f"\nModel saved to: {save_path}")

    if verbose:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total epochs: {epoch+1}")
        print(f"Best epoch: {early_stopping.best_epoch+1}")
        print(f"Best val loss: {early_stopping.best_loss:.6f}")
        print(f"Training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print("=" * 60 + "\n")

    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': early_stopping.best_epoch,
        'total_epochs': epoch + 1,
        'training_time': total_time,
        'param_count': param_count,
        'device': str(device),
        'best_val_loss': early_stopping.best_loss
    }


def evaluate_reconstruction_error(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 64
) -> np.ndarray:
    """
    Compute reconstruction errors for anomaly scoring.

    Args:
        model: Trained autoencoder
        X: Features to evaluate
        device: Device to run on
        batch_size: Batch size for inference

    Returns:
        Array of reconstruction errors (MSE per sample)
    """
    model.eval()
    model = model.to(device)

    X_tensor = torch.FloatTensor(X)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    errors = []

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)

            # Get reconstruction error
            error = model.get_reconstruction_error(batch_x, reduction='none')
            errors.append(error.cpu().numpy())

    return np.concatenate(errors)


if __name__ == "__main__":
    # Test the training infrastructure
    from src.models import DeepAutoencoder

    print("Testing training infrastructure...")

    # Create dummy data
    np.random.seed(42)
    X_train = np.random.randn(1000, 29).astype(np.float32)
    X_val = np.random.randn(200, 29).astype(np.float32)

    # Create model
    model = DeepAutoencoder(input_dim=29, latent_dim=8)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train
    results = train_autoencoder(
        model=model,
        X_train=X_train,
        X_val=X_val,
        device=device,
        learning_rate=1e-3,
        batch_size=32,
        max_epochs=5,
        patience=3,
        verbose=True
    )

    print("\nTraining results:")
    for key, value in results.items():
        if key not in ['model', 'train_losses', 'val_losses']:
            print(f"  {key}: {value}")

    # Test evaluation
    print("\nTesting reconstruction error evaluation...")
    errors = evaluate_reconstruction_error(
        model=results['model'],
        X=X_val,
        device=device,
        batch_size=32
    )
    print(f"Reconstruction errors shape: {errors.shape}")
    print(f"Mean error: {errors.mean():.6f}")
    print(f"Std error: {errors.std():.6f}")
