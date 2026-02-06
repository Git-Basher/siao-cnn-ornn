"""
TimeGAN-based Data Augmentor for NPP Time-Series

Uses TimeGAN (Time-series Generative Adversarial Network) to generate
realistic synthetic time-series data that preserves temporal dynamics.

References:
- Yoon et al. "Time-series Generative Adversarial Networks" NeurIPS 2019
- ydata-synthetic library: https://github.com/ydataai/ydata-synthetic
"""

import numpy as np
import os
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeGANAugmentor:
    """
    TimeGAN-based synthetic data generator for NPP time-series.
    
    Generates class-conditional synthetic samples to balance the dataset
    and expand from ~81 samples to 600+ samples.
    
    Attributes:
        seq_len: Sequence length for time-series windows
        n_features: Number of sensor features
        models: Dictionary mapping class labels to trained TimeGAN models
        output_dir: Directory to save augmented data
    """
    
    def __init__(
        self,
        seq_len: int = 100,
        n_features: int = 11,
        hidden_dim: int = 24,
        num_layers: int = 3,
        output_dir: str = 'data/augmented'
    ):
        """
        Initialize the TimeGAN augmentor.
        
        Args:
            seq_len: Length of time-series sequences
            n_features: Number of features per timestep
            hidden_dim: Hidden dimension for TimeGAN networks
            num_layers: Number of layers in TimeGAN networks
            output_dir: Directory to save augmented data
        """
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[int, object] = {}
        self._synth_available = self._check_ydata_synthetic()
    
    def _check_ydata_synthetic(self) -> bool:
        """Check if PyTorch TimeGAN is available (always True now)."""
        return True
    
    def _get_class_samples(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        class_label: int
    ) -> np.ndarray:
        """Extract samples for a specific class."""
        mask = y == class_label
        return X[mask]
    
    def fit_class(
        self,
        X_class: np.ndarray,
        class_label: int,
        epochs: int = 1000,
        batch_size: int = 16
    ) -> None:
        """
        Train a TimeGAN model for a specific class.
        
        Args:
            X_class: Time-series data for this class [samples, seq_len, features]
            class_label: Integer class label
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        from .timegan_pytorch import TimeGAN, TimeGANTrainer
        import torch
        
        logger.info(f"Training TimeGAN for class {class_label} with {len(X_class)} samples")
        
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize TimeGAN
        model = TimeGAN(
            input_dim=X_class.shape[2],
            hidden_dim=self.hidden_dim,
            noise_dim=self.hidden_dim,
            num_layers=self.num_layers,
            device=device
        )
        
        # Train
        trainer = TimeGANTrainer(model, lr=5e-4)
        trainer.train(
            X_class,
            epochs=epochs,
            batch_size=min(batch_size, len(X_class)),
            print_every=max(1, epochs // 10)
        )
        
        self.models[class_label] = model
        
        # Save trained model
        model_path = self.output_dir / f'timegan_class_{class_label}.pt'
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved TimeGAN model to {model_path}")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        batch_size: int = 16
    ) -> None:
        """
        Train TimeGAN models for all classes.
        
        Args:
            X: Full dataset [samples, seq_len, features]
            y: Class labels [samples]
            epochs: Training epochs per class
            batch_size: Training batch size
        """
        unique_classes = np.unique(y)
        logger.info(f"Training TimeGAN for {len(unique_classes)} classes")
        
        for class_label in unique_classes:
            X_class = self._get_class_samples(X, y, class_label)
            if len(X_class) < 2:
                logger.warning(f"Class {class_label} has too few samples ({len(X_class)}), skipping")
                continue
            self.fit_class(X_class, int(class_label), epochs, batch_size)
    
    def generate(
        self,
        n_samples: int,
        class_label: int
    ) -> np.ndarray:
        """
        Generate synthetic samples for a specific class.
        
        Args:
            n_samples: Number of synthetic samples to generate
            class_label: Class to generate samples for
            
        Returns:
            Synthetic data array [n_samples, seq_len, features]
        """
        if class_label not in self.models:
            logger.error(f"No trained model for class {class_label}")
            return np.array([])
        
        synth = self.models[class_label]
        synthetic_data = synth.sample(n_samples, self.seq_len)
        logger.info(f"Generated {n_samples} synthetic samples for class {class_label}")
        
        return synthetic_data
    
    def augment_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_per_class: int = 100,
        include_original: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment dataset to have target samples per class.
        
        Args:
            X: Original data [samples, seq_len, features]
            y: Original labels [samples]
            target_per_class: Target number of samples per class
            include_original: Whether to include original data in output
            
        Returns:
            Tuple of (augmented_X, augmented_y)
        """
        unique_classes = np.unique(y)
        all_X = [] if not include_original else [X]
        all_y = [] if not include_original else [y]
        
        for class_label in unique_classes:
            current_count = np.sum(y == class_label)
            samples_needed = max(0, target_per_class - current_count)
            
            if samples_needed > 0 and class_label in self.models:
                synthetic_X = self.generate(samples_needed, class_label)
                synthetic_y = np.full(samples_needed, class_label)
                all_X.append(synthetic_X)
                all_y.append(synthetic_y)
                logger.info(
                    f"Class {class_label}: {current_count} original + "
                    f"{samples_needed} synthetic = {target_per_class} total"
                )
        
        X_augmented = np.concatenate(all_X, axis=0)
        y_augmented = np.concatenate(all_y, axis=0)
        
        logger.info(f"Final augmented dataset: {len(X_augmented)} samples")
        return X_augmented, y_augmented
    
    def save_augmented(
        self,
        X: np.ndarray,
        y: np.ndarray,
        prefix: str = ''
    ) -> Tuple[Path, Path]:
        """Save augmented data to disk."""
        X_path = self.output_dir / f'{prefix}X_augmented.npy'
        y_path = self.output_dir / f'{prefix}y_augmented.npy'
        
        np.save(X_path, X)
        np.save(y_path, y)
        
        logger.info(f"Saved augmented data to {self.output_dir}")
        return X_path, y_path
    
    def load_augmented(self, prefix: str = '') -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load augmented data from disk if available."""
        X_path = self.output_dir / f'{prefix}X_augmented.npy'
        y_path = self.output_dir / f'{prefix}y_augmented.npy'
        
        if X_path.exists() and y_path.exists():
            X = np.load(X_path)
            y = np.load(y_path)
            logger.info(f"Loaded augmented data: {len(X)} samples")
            return X, y
        return None
    
    def load_model(self, class_label: int) -> bool:
        """Load a trained TimeGAN model from disk."""
        from .timegan_pytorch import TimeGAN
        import torch
        
        model_path = self.output_dir / f'timegan_class_{class_label}.pt'
        
        if model_path.exists():
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = TimeGAN(
                input_dim=self.n_features,
                hidden_dim=self.hidden_dim,
                noise_dim=self.hidden_dim,
                num_layers=self.num_layers,
                device=device
            )
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            self.models[class_label] = model
            logger.info(f"Loaded TimeGAN model for class {class_label}")
            return True
        return False
    
    def load_all_models(self) -> int:
        """Load all available TimeGAN models from disk."""
        loaded = 0
        for model_file in self.output_dir.glob('timegan_class_*.pt'):
            class_label = int(model_file.stem.split('_')[-1])
            if self.load_model(class_label):
                loaded += 1
        return loaded


def run_augmentation(
    X: np.ndarray,
    y: np.ndarray,
    target_per_class: int = 100,
    epochs: int = 1000,
    use_cache: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to run full augmentation pipeline.
    
    Args:
        X: Original data
        y: Original labels
        target_per_class: Target samples per class
        epochs: Training epochs
        use_cache: Whether to use cached augmented data
        
    Returns:
        Tuple of (augmented_X, augmented_y)
    """
    augmentor = TimeGANAugmentor(
        seq_len=X.shape[1] if len(X.shape) > 1 else 100,
        n_features=X.shape[2] if len(X.shape) > 2 else X.shape[1]
    )
    
    # Try to load cached data
    if use_cache:
        cached = augmentor.load_augmented()
        if cached is not None:
            return cached
    
    # Try to load existing models
    models_loaded = augmentor.load_all_models()
    
    # Train models if needed
    if models_loaded < len(np.unique(y)):
        augmentor.fit(X, y, epochs=epochs)
    
    # Generate augmented dataset
    X_aug, y_aug = augmentor.augment_dataset(X, y, target_per_class)
    
    # Save for future use
    augmentor.save_augmented(X_aug, y_aug)
    
    return X_aug, y_aug
