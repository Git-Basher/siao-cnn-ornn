"""
Stage 0: TimeGAN Data Augmentation Module

Generates synthetic time-series data using TimeGAN to augment
the IP-200 reactor dataset from ~81 samples to 600+ samples.
"""

from .time_gan_augmentor import TimeGANAugmentor
from .quality_metrics import (
    calculate_discriminative_score,
    calculate_predictive_score,
    plot_pca_comparison
)

__all__ = [
    'TimeGANAugmentor',
    'calculate_discriminative_score',
    'calculate_predictive_score',
    'plot_pca_comparison'
]
