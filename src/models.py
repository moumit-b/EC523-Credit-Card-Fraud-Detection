

import torch
import torch.nn as nn
from typing import List, Optional


class DeepAutoencoder(nn.Module):
    """
    Deep Autoencoder for unsupervised anomaly detection.

    Architecture:
        Encoder: input_dim -> hidden_dims -> latent_dim
        Decoder: latent_dim -> hidden_dims (reversed) -> input_dim

    The autoencoder is trained to reconstruct normal transactions.
    Fraudulent transactions should have higher reconstruction error.
    """

    def __init__(
        self,
        input_dim: int = 29,
        latent_dim: int = 8,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.2
    ):
        """
        Initialize the Deep Autoencoder.

        Args:
            input_dim: Number of input features (default 29: V1-V28 + Amount)
            latent_dim: Bottleneck dimension
            hidden_dims: List of hidden layer dimensions (default [64, 32])
            dropout_rate: Dropout probability for regularization
        """
        super(DeepAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [64, 32]
        self.dropout_rate = dropout_rate

        # Build encoder
        encoder_layers = []
        prev_dim = input_dim

        for hidden_dim in self.hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        # Add bottleneck layer
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))

        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder (symmetric to encoder)
        decoder_layers = []
        prev_dim = latent_dim

        # Reverse the hidden dimensions
        reversed_hidden_dims = list(reversed(self.hidden_dims))

        for hidden_dim in reversed_hidden_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        # Output layer (no activation - we want continuous output)
        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through autoencoder.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Reconstructed tensor of shape (batch_size, input_dim)
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Latent tensor of shape (batch_size, latent_dim)
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstruction.

        Args:
            z: Latent tensor of shape (batch_size, latent_dim)

        Returns:
            Reconstructed tensor of shape (batch_size, input_dim)
        """
        return self.decoder(z)

    def get_reconstruction_error(
        self,
        x: torch.Tensor,
        reduction: str = 'none'
    ) -> torch.Tensor:
        """
        Compute reconstruction error (MSE) for anomaly scoring.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            reduction: 'none' returns per-sample errors,
                      'mean' returns average error

        Returns:
            Reconstruction error(s)
        """
        reconstructed = self.forward(x)

        if reduction == 'none':
            # Per-sample MSE (average over features)
            error = torch.mean((x - reconstructed) ** 2, dim=1)
        elif reduction == 'mean':
            # Average MSE over all samples and features
            error = torch.mean((x - reconstructed) ** 2)
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

        return error

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        """String representation of the model."""
        param_count = self.count_parameters()
        return (
            f"DeepAutoencoder(\n"
            f"  input_dim={self.input_dim},\n"
            f"  latent_dim={self.latent_dim},\n"
            f"  hidden_dims={self.hidden_dims},\n"
            f"  dropout_rate={self.dropout_rate},\n"
            f"  total_params={param_count:,}\n"
            f")"
        )


class DenoisingAutoencoder(DeepAutoencoder):
    """
    Denoising Autoencoder variant.

    Adds Gaussian noise to inputs during training to learn robust
    representations. The model is trained to reconstruct the clean input
    from the noisy version.

    Inherits architecture from DeepAutoencoder.
    """

    def __init__(
        self,
        input_dim: int = 29,
        latent_dim: int = 8,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.2,
        noise_level: float = 0.1
    ):
        """
        Initialize the Denoising Autoencoder.

        Args:
            input_dim: Number of input features
            latent_dim: Bottleneck dimension
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            noise_level: Standard deviation of Gaussian noise to add
        """
        super(DenoisingAutoencoder, self).__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        )

        self.noise_level = noise_level

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to input.

        Args:
            x: Input tensor

        Returns:
            Noisy input tensor
        """
        if self.training and self.noise_level > 0:
            noise = torch.randn_like(x) * self.noise_level
            return x + noise
        else:
            return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with noise injection during training.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Reconstructed tensor of shape (batch_size, input_dim)
        """
        # Add noise only during training
        noisy_x = self.add_noise(x)

        # Encode noisy input
        latent = self.encoder(noisy_x)

        # Decode to reconstruct CLEAN input
        reconstructed = self.decoder(latent)

        return reconstructed

    def __repr__(self) -> str:
        """String representation of the model."""
        param_count = self.count_parameters()
        return (
            f"DenoisingAutoencoder(\n"
            f"  input_dim={self.input_dim},\n"
            f"  latent_dim={self.latent_dim},\n"
            f"  hidden_dims={self.hidden_dims},\n"
            f"  dropout_rate={self.dropout_rate},\n"
            f"  noise_level={self.noise_level},\n"
            f"  total_params={param_count:,}\n"
            f")"
        )


if __name__ == "__main__":
    # Test the models
    print("Testing Deep Autoencoder...")
    ae = DeepAutoencoder(input_dim=29, latent_dim=8, hidden_dims=[64, 32])
    print(ae)

    # Test forward pass
    x = torch.randn(10, 29)
    reconstructed = ae(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {reconstructed.shape}")

    # Test reconstruction error
    errors = ae.get_reconstruction_error(x, reduction='none')
    print(f"Per-sample errors shape: {errors.shape}")
    print(f"Mean reconstruction error: {errors.mean().item():.6f}")

    print("\n" + "="*60)
    print("Testing Denoising Autoencoder...")
    dae = DenoisingAutoencoder(
        input_dim=29,
        latent_dim=8,
        hidden_dims=[64, 32],
        noise_level=0.1
    )
    print(dae)

    # Test with noise
    dae.train()  # Enable training mode
    reconstructed_noisy = dae(x)
    print(f"\nWith noise - Input shape: {x.shape}")
    print(f"With noise - Output shape: {reconstructed_noisy.shape}")

    # Test without noise (eval mode)
    dae.eval()
    reconstructed_clean = dae(x)
    print(f"Without noise - Output shape: {reconstructed_clean.shape}")
