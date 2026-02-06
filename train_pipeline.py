"""
Complete SIAO-CNN-ORNN Training Pipeline

Integrates all components:
1. Data Pipeline - Load and preprocess reactor data
2. Window Processor - Create sliding windows
3. CNN Feature Extractor - Extract spatial features
4. ORNN - SIAO-optimized RNN for classification

Author: SIAO-CNN-ORNN Integration
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_complete_pipeline(
    data_dir: str = 'data/',
    window_size: int = 50,
    stride: int = 25,
    cnn_embedding_dim: int = 256,
    wks_dim: int = 43,  # WKS features dimension (num signals)
    rnn_hidden_size: int = 128,
    num_classes: int = 6,
    test_size: float = 0.2,
    wks_pop_size: int = 15,
    wks_max_iter: int = 30,
    siao_pop_size: int = 20,
    siao_max_iter: int = 50,
    bp_epochs: int = 100,
    batch_size: int = 32
) -> Dict:
    """
    Run the complete SIAO-CNN-ORNN training pipeline.
    
    Args:
        data_dir: Path to data directory
        window_size: Sliding window size
        stride: Window stride
        cnn_embedding_dim: CNN output dimension
        rnn_hidden_size: RNN hidden size
        num_classes: Number of output classes
        test_size: Validation split ratio
        siao_pop_size: SIAO population size
        siao_max_iter: SIAO iterations
        bp_epochs: Backpropagation epochs
        batch_size: Training batch size
    
    Returns:
        Dictionary with results and history
    """
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # =========================================================================
    # Step 1: Load and Preprocess Data
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 1: Loading and Preprocessing Data")
    logger.info("=" * 60)
    
    from stage1_data.data_pipeline import IP200DataPipeline
    
    pipeline = IP200DataPipeline(
        data_dir=data_dir,
        normalization='zscore',
        outlier_window=5,
        handle_missing='interpolate'
    )
    
    X_raw, y_raw = pipeline.run()
    logger.info(f"Raw data: X={X_raw.shape}, y={y_raw.shape}")
    
    # =========================================================================
    # Step 2: Create Sliding Windows
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 2: Creating Sliding Windows")
    logger.info("=" * 60)
    
    from stage1_data.window_processor import SlidingWindowProcessor
    
    window_proc = SlidingWindowProcessor(
        window_size=window_size,
        stride=stride,
        padding='zero'
    )
    
    X_windows, y_windows = window_proc.transform(X_raw, y_raw)
    logger.info(f"Windowed data: X={X_windows.shape}, y={y_windows.shape}")
    
    # =========================================================================
    # Step 3: Train/Validation Split
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 3: Train/Validation Split")
    logger.info("=" * 60)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_windows, y_windows,
        test_size=test_size,
        random_state=42,
        stratify=y_windows
    )
    
    logger.info(f"Train: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Val:   X={X_val.shape}, y={y_val.shape}")
    
    # =========================================================================
    # Step 4: CNN Feature Extraction
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 4: CNN Feature Extraction")
    logger.info("=" * 60)
    
    from stage3_models.cnn_model import CNNFeatureExtractor
    
    cnn = CNNFeatureExtractor(
        input_channels=1,
        input_time_steps=window_size,
        input_features=X_windows.shape[2],
        conv_channels=(32, 64, 128),
        embedding_dim=cnn_embedding_dim,
        dropout=0.3
    ).to(device)
    
    # Extract CNN features
    cnn.eval()
    with torch.no_grad():
        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
        
        # Process in batches to avoid OOM
        train_embeddings = []
        for i in range(0, len(X_train_t), batch_size):
            batch = X_train_t[i:i+batch_size]
            emb = cnn(batch)
            train_embeddings.append(emb.cpu().numpy())
        X_train_cnn = np.vstack(train_embeddings)
        
        val_embeddings = []
        for i in range(0, len(X_val_t), batch_size):
            batch = X_val_t[i:i+batch_size]
            emb = cnn(batch)
            val_embeddings.append(emb.cpu().numpy())
        X_val_cnn = np.vstack(val_embeddings)
    
    logger.info(f"CNN embeddings: Train={X_train_cnn.shape}, Val={X_val_cnn.shape}")
    
    # =========================================================================
    # Step 4.5: WKS Feature Extraction (Aquila-optimized)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 4.5: WKS Feature Extraction (Aquila-optimized)")
    logger.info("=" * 60)
    
    from stage4_optimizers.aquila_optimizer import WKSOptimizer, compute_wks
    
    # Optimize omega using Aquila on training data
    wks_optimizer = WKSOptimizer(
        pop_size=wks_pop_size,
        max_iter=wks_max_iter,
        omega_bounds=(0.1, 10.0)
    )
    
    optimal_omega, wks_fitness, _ = wks_optimizer.optimize(X_train, y_train)
    logger.info(f"Optimal omega: {optimal_omega:.4f}, Fitness: {wks_fitness:.4f}")
    
    # Extract WKS features for train and val
    X_train_wks = wks_optimizer.extract_wks_features(X_train)
    X_val_wks = wks_optimizer.extract_wks_features(X_val)
    
    # Flatten WKS features if needed: [samples, num_features]
    if len(X_train_wks.shape) > 2:
        X_train_wks = X_train_wks.reshape(X_train_wks.shape[0], -1)
        X_val_wks = X_val_wks.reshape(X_val_wks.shape[0], -1)
    
    logger.info(f"WKS features: Train={X_train_wks.shape}, Val={X_val_wks.shape}")
    
    # =========================================================================
    # Step 4.75: Concatenate CNN + WKS Features
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 4.75: Concatenating CNN + WKS Features")
    logger.info("=" * 60)
    
    # Concatenate: [samples, cnn_dim + wks_dim]
    X_train_combined = np.concatenate([X_train_cnn, X_train_wks], axis=1)
    X_val_combined = np.concatenate([X_val_cnn, X_val_wks], axis=1)
    
    logger.info(f"Combined features: Train={X_train_combined.shape}, Val={X_val_combined.shape}")
    
    # Reshape for RNN: [samples, seq_len=1, features]
    X_train_rnn = X_train_combined[:, np.newaxis, :]
    X_val_rnn = X_val_combined[:, np.newaxis, :]
    
    logger.info(f"RNN input: Train={X_train_rnn.shape}, Val={X_val_rnn.shape}")
    
    # =========================================================================
    # Step 5: ORNN Training (SIAO + Backprop)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 5: ORNN Training (SIAO + Backprop)")
    logger.info("=" * 60)
    
    from stage3_models.ornn_model import ORNN, SIAOORNNTrainer, plot_ornn_training
    
    # Combined input size = CNN embedding + WKS features
    combined_input_size = X_train_combined.shape[1]
    
    ornn = ORNN(
        input_size=combined_input_size,
        hidden_size=rnn_hidden_size,
        num_layers=1,
        cell_type='gru'
    )
    
    trainer = SIAOORNNTrainer(
        ornn=ornn,
        output_size=num_classes,
        device=device,
        siao_pop_size=siao_pop_size,
        siao_max_iter=siao_max_iter,
        bp_epochs=bp_epochs,
        bp_lr=0.001
    )
    
    history = trainer.train(
        X_train_rnn, y_train,
        X_val_rnn, y_val,
        batch_size=batch_size
    )
    
    # =========================================================================
    # Step 6: Final Evaluation
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 6: Final Evaluation")
    logger.info("=" * 60)
    
    # Training accuracy
    train_preds = trainer.predict(X_train_rnn)
    train_acc = (train_preds == y_train).mean()
    
    # Validation accuracy
    val_preds = trainer.predict(X_val_rnn)
    val_acc = (val_preds == y_val).mean()
    
    logger.info(f"Final Train Accuracy: {train_acc:.4f}")
    logger.info(f"Final Val Accuracy:   {val_acc:.4f}")
    
    # Class-wise accuracy
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(y_val, val_preds, digits=4))
    
    # Plot training history
    plot_ornn_training(history)
    
    # =========================================================================
    # Return Results
    # =========================================================================
    results = {
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'history': history,
        'cnn_model': cnn,
        'ornn_trainer': trainer,
        'y_val': y_val,
        'val_preds': val_preds
    }
    
    return results


# =============================================================================
# Quick Start Function
# =============================================================================

def quick_start():
    """
    Quick start with default parameters.
    
    Usage in Colab:
        from train_pipeline import quick_start
        results = quick_start()
    """
    return run_complete_pipeline(
        data_dir='data/',
        window_size=50,
        stride=25,
        cnn_embedding_dim=256,
        wks_dim=43,
        rnn_hidden_size=128,
        num_classes=6,
        test_size=0.2,
        wks_pop_size=15,
        wks_max_iter=30,
        siao_pop_size=15,
        siao_max_iter=30,
        bp_epochs=50,
        batch_size=32
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("SIAO-CNN-ORNN Complete Training Pipeline")
    print("=" * 60)
    
    results = quick_start()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final Validation Accuracy: {results['val_accuracy']:.4f}")
