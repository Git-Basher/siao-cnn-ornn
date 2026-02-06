"""
Quality Metrics for Synthetic Data Validation

Provides metrics to evaluate the quality of generated synthetic data:
1. PCA Visualization - Visual comparison of real vs synthetic distributions
2. Discriminative Score - How well can a classifier distinguish real from synthetic
3. Predictive Score - How well does synthetic data preserve temporal patterns
"""

import numpy as np
import logging
from typing import Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def plot_pca_comparison(
    X_real: np.ndarray,
    X_synthetic: np.ndarray,
    save_path: Optional[str] = None,
    title: str = 'Real vs Synthetic Data (PCA)'
) -> None:
    """
    Create PCA visualization comparing real and synthetic data distributions.
    
    Args:
        X_real: Real data [samples, seq_len, features]
        X_synthetic: Synthetic data [samples, seq_len, features]
        save_path: Path to save the plot
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        logger.error("matplotlib and sklearn required for PCA visualization")
        return
    
    # Flatten time-series for PCA
    X_real_flat = X_real.reshape(X_real.shape[0], -1)
    X_synthetic_flat = X_synthetic.reshape(X_synthetic.shape[0], -1)
    
    # Combine for PCA fitting
    X_combined = np.vstack([X_real_flat, X_synthetic_flat])
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_combined)
    
    # Split back
    n_real = len(X_real)
    X_real_pca = X_pca[:n_real]
    X_synthetic_pca = X_pca[n_real:]
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(X_real_pca[:, 0], X_real_pca[:, 1], 
                c='blue', alpha=0.6, label=f'Real ({n_real})', s=50)
    plt.scatter(X_synthetic_pca[:, 0], X_synthetic_pca[:, 1], 
                c='red', alpha=0.6, label=f'Synthetic ({len(X_synthetic)})', s=50)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved PCA plot to {save_path}")
    
    plt.close()


def calculate_discriminative_score(
    X_real: np.ndarray,
    X_synthetic: np.ndarray,
    test_ratio: float = 0.2
) -> float:
    """
    Calculate discriminative score: how well can a classifier distinguish real from synthetic.
    
    Lower score = better synthetic data (harder to distinguish).
    Perfect score = 0.5 (random guessing).
    
    Args:
        X_real: Real data [samples, seq_len, features]
        X_synthetic: Synthetic data [samples, seq_len, features]
        test_ratio: Fraction of data to use for testing
        
    Returns:
        Discriminative score (0-1, lower is better)
    """
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
    except ImportError:
        logger.error("sklearn required for discriminative score")
        return -1.0
    
    # Flatten for classification
    X_real_flat = X_real.reshape(X_real.shape[0], -1)
    X_synthetic_flat = X_synthetic.reshape(X_synthetic.shape[0], -1)
    
    # Create labels (0 = real, 1 = synthetic)
    y_real = np.zeros(len(X_real))
    y_synthetic = np.ones(len(X_synthetic))
    
    # Combine
    X = np.vstack([X_real_flat, X_synthetic_flat])
    y = np.concatenate([y_real, y_synthetic])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42, stratify=y
    )
    
    # Train discriminator
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    disc_score = accuracy_score(y_test, y_pred)
    
    logger.info(f"Discriminative Score: {disc_score:.4f} (closer to 0.5 is better)")
    return disc_score


def calculate_predictive_score(
    X_real: np.ndarray,
    X_synthetic: np.ndarray,
    prediction_horizon: int = 10
) -> Tuple[float, float]:
    """
    Calculate predictive score: how well do models trained on synthetic data
    predict real data sequences.
    
    Uses a simple LSTM to predict next timesteps.
    
    Args:
        X_real: Real data [samples, seq_len, features]
        X_synthetic: Synthetic data [samples, seq_len, features]
        prediction_horizon: Number of timesteps to predict
        
    Returns:
        Tuple of (train_on_real_mae, train_on_synthetic_mae)
    """
    try:
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error
    except ImportError:
        logger.error("sklearn required for predictive score")
        return -1.0, -1.0
    
    # Use simple approach: predict last timesteps from earlier ones
    def prepare_sequences(X, horizon):
        X_input = X[:, :-horizon, :].reshape(X.shape[0], -1)
        y_target = X[:, -horizon:, :].reshape(X.shape[0], -1)
        return X_input, y_target
    
    # Prepare data
    X_real_in, y_real_out = prepare_sequences(X_real, prediction_horizon)
    X_synth_in, y_synth_out = prepare_sequences(X_synthetic, prediction_horizon)
    
    # Train on real, test on real
    model_real = Ridge()
    model_real.fit(X_real_in[:int(0.8*len(X_real_in))], 
                   y_real_out[:int(0.8*len(y_real_out))])
    pred_real = model_real.predict(X_real_in[int(0.8*len(X_real_in)):])
    mae_real = mean_absolute_error(y_real_out[int(0.8*len(y_real_out)):], pred_real)
    
    # Train on synthetic, test on real
    model_synth = Ridge()
    model_synth.fit(X_synth_in, y_synth_out)
    pred_synth = model_synth.predict(X_real_in[int(0.8*len(X_real_in)):])
    mae_synth = mean_absolute_error(y_real_out[int(0.8*len(y_real_out)):], pred_synth)
    
    logger.info(f"Predictive Score - Real: {mae_real:.4f}, Synthetic: {mae_synth:.4f}")
    return mae_real, mae_synth


def validate_synthetic_data(
    X_real: np.ndarray,
    X_synthetic: np.ndarray,
    output_dir: str = 'results/augmentation'
) -> dict:
    """
    Run full validation suite on synthetic data.
    
    Args:
        X_real: Real data
        X_synthetic: Synthetic data
        output_dir: Directory to save validation results
        
    Returns:
        Dictionary with all validation metrics
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # PCA visualization
    plot_pca_comparison(
        X_real, X_synthetic,
        save_path=f'{output_dir}/pca_real_vs_synthetic.png'
    )
    
    # Discriminative score
    disc_score = calculate_discriminative_score(X_real, X_synthetic)
    results['discriminative_score'] = disc_score
    results['discriminative_quality'] = 'Good' if disc_score < 0.7 else 'Poor'
    
    # Predictive score
    mae_real, mae_synth = calculate_predictive_score(X_real, X_synthetic)
    results['predictive_mae_real'] = mae_real
    results['predictive_mae_synthetic'] = mae_synth
    results['predictive_quality'] = 'Good' if mae_synth < 1.5 * mae_real else 'Poor'
    
    # Summary
    logger.info("=" * 50)
    logger.info("Synthetic Data Quality Report")
    logger.info("=" * 50)
    logger.info(f"Discriminative Score: {disc_score:.4f} ({results['discriminative_quality']})")
    logger.info(f"Predictive MAE (Real): {mae_real:.4f}")
    logger.info(f"Predictive MAE (Synth): {mae_synth:.4f} ({results['predictive_quality']})")
    
    return results
