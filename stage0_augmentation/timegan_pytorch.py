"""
Pure PyTorch TimeGAN Implementation

Time-series Generative Adversarial Network for synthetic data generation.
Based on: Yoon et al. "Time-series Generative Adversarial Networks" NeurIPS 2019

This is a standalone PyTorch implementation without external dependencies.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class EmbedderNetwork(nn.Module):
    """Embeds real time-series into latent space."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            h: [batch, seq_len, hidden_dim]
        """
        h, _ = self.rnn(x)
        h = self.fc(h)
        h = self.activation(h)
        return h


class RecoveryNetwork(nn.Module):
    """Recovers time-series from latent embeddings."""
    
    def __init__(self, hidden_dim: int, output_dim: int, num_layers: int = 3):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [batch, seq_len, hidden_dim]
        Returns:
            x_tilde: [batch, seq_len, output_dim]
        """
        o, _ = self.rnn(h)
        x_tilde = self.fc(o)
        return x_tilde


class GeneratorNetwork(nn.Module):
    """Generates synthetic latent embeddings from noise."""
    
    def __init__(self, noise_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=noise_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Sigmoid()
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [batch, seq_len, noise_dim]
        Returns:
            e_hat: [batch, seq_len, hidden_dim]
        """
        o, _ = self.rnn(z)
        e_hat = self.fc(o)
        e_hat = self.activation(e_hat)
        return e_hat


class SupervisorNetwork(nn.Module):
    """Supervises generator to learn temporal dynamics."""
    
    def __init__(self, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Sigmoid()
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [batch, seq_len, hidden_dim]
        Returns:
            s: [batch, seq_len, hidden_dim]
        """
        o, _ = self.rnn(h)
        s = self.fc(o)
        s = self.activation(s)
        return s


class DiscriminatorNetwork(nn.Module):
    """Discriminates real from synthetic embeddings."""
    
    def __init__(self, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [batch, seq_len, hidden_dim]
        Returns:
            y: [batch, seq_len, 1] - real/synthetic scores
        """
        o, _ = self.rnn(h)
        y = self.fc(o)
        return y


class TimeGAN(nn.Module):
    """
    Time-series Generative Adversarial Network.
    
    Combines embedder, recovery, generator, supervisor, and discriminator
    networks to learn and generate realistic time-series data.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 24,
        noise_dim: int = 24,
        num_layers: int = 3,
        device: str = 'cpu'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.device = device
        
        # Networks
        self.embedder = EmbedderNetwork(input_dim, hidden_dim, num_layers)
        self.recovery = RecoveryNetwork(hidden_dim, input_dim, num_layers)
        self.generator = GeneratorNetwork(noise_dim, hidden_dim, num_layers)
        self.supervisor = SupervisorNetwork(hidden_dim, num_layers-1)
        self.discriminator = DiscriminatorNetwork(hidden_dim, num_layers)
        
        self.to(device)
    
    def _get_noise(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """Generate random noise."""
        return torch.randn(batch_size, seq_len, self.noise_dim, device=self.device)
    
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embed real data."""
        return self.embedder(x)
    
    def recover(self, h: torch.Tensor) -> torch.Tensor:
        """Recover data from embedding."""
        return self.recovery(h)
    
    def generate(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """Generate synthetic embedding from noise."""
        z = self._get_noise(batch_size, seq_len)
        e_hat = self.generator(z)
        return self.supervisor(e_hat)
    
    def discriminate(self, h: torch.Tensor) -> torch.Tensor:
        """Discriminate real/synthetic."""
        return self.discriminator(h)
    
    def sample(self, n_samples: int, seq_len: int) -> np.ndarray:
        """
        Generate synthetic time-series samples.
        
        Args:
            n_samples: Number of samples to generate
            seq_len: Sequence length
            
        Returns:
            Synthetic data [n_samples, seq_len, input_dim]
        """
        self.eval()
        with torch.no_grad():
            z = self._get_noise(n_samples, seq_len)
            e_hat = self.generator(z)
            h_hat = self.supervisor(e_hat)
            x_hat = self.recovery(h_hat)
        return x_hat.cpu().numpy()


class TimeGANTrainer:
    """Trainer for TimeGAN model."""
    
    def __init__(
        self,
        model: TimeGAN,
        lr: float = 1e-3,
        gamma: float = 1.0
    ):
        self.model = model
        self.gamma = gamma
        
        # Optimizers for each component
        self.opt_embedder = torch.optim.Adam(
            model.embedder.parameters(), lr=lr
        )
        self.opt_recovery = torch.optim.Adam(
            model.recovery.parameters(), lr=lr
        )
        self.opt_generator = torch.optim.Adam(
            model.generator.parameters(), lr=lr
        )
        self.opt_supervisor = torch.optim.Adam(
            model.supervisor.parameters(), lr=lr
        )
        self.opt_discriminator = torch.optim.Adam(
            model.discriminator.parameters(), lr=lr
        )
        
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def _train_embedder_step(self, x: torch.Tensor) -> float:
        """Train embedder and recovery networks."""
        self.opt_embedder.zero_grad()
        self.opt_recovery.zero_grad()
        
        # Reconstruction loss
        h = self.model.embed(x)
        x_tilde = self.model.recover(h)
        
        e_loss = 10 * torch.sqrt(self.mse_loss(x, x_tilde))
        
        # Supervised loss
        h_hat_supervise = self.model.supervisor(h)
        g_loss_s = self.mse_loss(h[:, 1:, :], h_hat_supervise[:, :-1, :])
        
        loss = e_loss + 0.1 * g_loss_s
        loss.backward()
        
        self.opt_embedder.step()
        self.opt_recovery.step()
        
        return loss.item()
    
    def _train_supervisor_step(self, x: torch.Tensor) -> float:
        """Train supervisor network."""
        self.opt_supervisor.zero_grad()
        
        h = self.model.embed(x)
        h_hat_supervise = self.model.supervisor(h)
        
        g_loss_s = self.mse_loss(h[:, 1:, :], h_hat_supervise[:, :-1, :])
        g_loss_s.backward()
        
        self.opt_supervisor.step()
        
        return g_loss_s.item()
    
    def _train_generator_step(self, x: torch.Tensor) -> float:
        """Train generator network."""
        self.opt_generator.zero_grad()
        self.opt_supervisor.zero_grad()
        
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Real embedding
        h = self.model.embed(x)
        
        # Generate synthetic
        h_hat = self.model.generate(batch_size, seq_len)
        
        # Discriminator outputs
        y_fake = self.model.discriminate(h_hat)
        
        # Generator loss: fool discriminator
        g_loss_u = self.bce_loss(y_fake, torch.ones_like(y_fake))
        
        # Supervisor loss
        h_hat_supervise = self.model.supervisor(h)
        g_loss_s = self.mse_loss(h[:, 1:, :], h_hat_supervise[:, :-1, :])
        
        # Moment loss
        g_loss_v = torch.mean(torch.abs(
            torch.sqrt(h_hat.var(dim=0) + 1e-6) - torch.sqrt(h.var(dim=0) + 1e-6)
        ))
        g_loss_v += torch.mean(torch.abs(h_hat.mean(dim=0) - h.mean(dim=0)))
        
        loss = g_loss_u + self.gamma * g_loss_s + 100 * g_loss_v
        loss.backward()
        
        self.opt_generator.step()
        self.opt_supervisor.step()
        
        return loss.item()
    
    def _train_discriminator_step(self, x: torch.Tensor) -> float:
        """Train discriminator network."""
        self.opt_discriminator.zero_grad()
        
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Real embedding
        h = self.model.embed(x)
        h_hat = self.model.generate(batch_size, seq_len)
        
        # Discriminator outputs
        y_real = self.model.discriminate(h)
        y_fake = self.model.discriminate(h_hat.detach())
        
        # Discriminator loss
        d_loss_real = self.bce_loss(y_real, torch.ones_like(y_real))
        d_loss_fake = self.bce_loss(y_fake, torch.zeros_like(y_fake))
        
        d_loss = d_loss_real + d_loss_fake
        
        # Only update if discriminator is not too strong
        if d_loss.item() > 0.15:
            d_loss.backward()
            self.opt_discriminator.step()
        
        return d_loss.item()
    
    def train(
        self,
        data: np.ndarray,
        epochs: int = 1000,
        batch_size: int = 16,
        print_every: int = 100
    ) -> dict:
        """
        Train TimeGAN model.
        
        Args:
            data: Training data [samples, seq_len, features]
            epochs: Number of training epochs
            batch_size: Batch size
            print_every: Print frequency
            
        Returns:
            Training history dictionary
        """
        self.model.train()
        
        # Convert to tensor
        data_tensor = torch.FloatTensor(data).to(self.model.device)
        n_samples = len(data_tensor)
        
        history = {'e_loss': [], 's_loss': [], 'g_loss': [], 'd_loss': []}
        
        logger.info(f"Phase 1: Embedding Training ({epochs // 5} steps)")
        # Phase 1: Train embedder
        for step in range(epochs // 5):
            idx = np.random.choice(n_samples, min(batch_size, n_samples), replace=False)
            x_batch = data_tensor[idx]
            e_loss = self._train_embedder_step(x_batch)
            history['e_loss'].append(e_loss)
        
        logger.info(f"Phase 2: Supervisor Training ({epochs // 5} steps)")
        # Phase 2: Train supervisor
        for step in range(epochs // 5):
            idx = np.random.choice(n_samples, min(batch_size, n_samples), replace=False)
            x_batch = data_tensor[idx]
            s_loss = self._train_supervisor_step(x_batch)
            history['s_loss'].append(s_loss)
        
        logger.info(f"Phase 3: Joint Training ({epochs - 2 * epochs // 5} steps)")
        # Phase 3: Joint training
        for step in range(epochs - 2 * epochs // 5):
            idx = np.random.choice(n_samples, min(batch_size, n_samples), replace=False)
            x_batch = data_tensor[idx]
            
            # Train generator twice for each discriminator step
            g_loss = self._train_generator_step(x_batch)
            g_loss = self._train_generator_step(x_batch)
            
            # Train embedder
            e_loss = self._train_embedder_step(x_batch)
            
            # Train discriminator
            d_loss = self._train_discriminator_step(x_batch)
            
            history['g_loss'].append(g_loss)
            history['d_loss'].append(d_loss)
            
            if (step + 1) % print_every == 0:
                logger.info(
                    f"Step {step + 1}: G_loss={g_loss:.4f}, D_loss={d_loss:.4f}"
                )
        
        logger.info("TimeGAN training complete!")
        return history
