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
from sklearn.model_selection import StratifiedKFold
from typing import Tuple, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_complete_pipeline(
    data_dir: str = 'data/',
    window_size: int = 100, # Increased from 50 to match specs
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
    bp_epochs: int = 100,  # Scaled to match iteration count
    batch_size: int = 163, # Updated to 163 as per research specs
    use_class_weights: bool = True
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
    
    # Rich Imports
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    
    console = Console()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"[bold blue]Device:[/bold blue] {device}")
    
    # =========================================================================
    # Step 1: Load and Preprocess Data
    # =========================================================================
    console.print(Panel("[bold green]Step 1: Loading and Preprocessing Data[/bold green]", box=box.DOUBLE))
    
    from stage1_data.data_pipeline import IP200DataPipeline
    
    with console.status("[bold green]Ingesting data...[/bold green]", spinner="dots"):
        pipeline = IP200DataPipeline(
            data_dir=data_dir,
            normalization='zscore',
            outlier_window=5,
            handle_missing='interpolate'
        )
        
        # Use caching to speed up subsequent runs
        X_raw, y_raw = pipeline.run(use_cache=True)
    
    console.print(f" [bold]Raw data loaded:[/bold] X={X_raw.shape}, y={y_raw.shape}")
    
    # =========================================================================
    # Step 2: Create Sliding Windows
    # =========================================================================
    console.print(Panel("[bold green]Step 2: Creating Sliding Windows[/bold green]", box=box.DOUBLE))
    
    from stage1_data.window_processor import SlidingWindowProcessor
    
    with console.status("[bold green]Processing windows...[/bold green]", spinner="dots"):
        window_proc = SlidingWindowProcessor(
            window_size=window_size,
            stride=stride,
            padding='zero'
        )
        
        X_windows, y_windows = window_proc.transform(X_raw, y_raw)
    
    console.print(f" [bold]Windowed data:[/bold] X={X_windows.shape}, y={y_windows.shape}")
    
    # =========================================================================
    # Step 3: 5-Fold Cross-Validation Setup
    # =========================================================================
    console.print(Panel("[bold green]Step 3: 5-Fold Cross-Validation Setup[/bold green]", box=box.DOUBLE))
    
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, classification_report
    from stage6_visualization import plot_training_results, plot_confusion_matrix_heatmap

    
    # Initialize Stratified K-Fold
    # Ensure n_splits=5 as per research specs
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_accuracies = []
    
    # =========================================================================
    # Step 4: Cross-Validation Loop
    # =========================================================================
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_windows, y_windows)):
        console.print(f"\n[bold magenta]=== Fold {fold+1}/5 ===[/bold magenta]")
        
        X_train, X_val = X_windows[train_idx], X_windows[val_idx]
        y_train, y_val = y_windows[train_idx], y_windows[val_idx]
        
        console.print(f" [bold]Train:[/bold] {len(y_train)} samples, [bold]Val:[/bold] {len(y_val)} samples")

        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        
        # ---------------------------------------------------------------------
        # CNN Feature Extraction (Fresh for each fold)
        # ---------------------------------------------------------------------
        input_channels = X_train.shape[2]
        
        from stage3_models.cnn_model import create_cnn_extractor
        cnn = create_cnn_extractor(
            input_shape=(window_size, input_channels),
            embedding_dim=cnn_embedding_dim,
            dropout=0.2
        ).to(device)
        
        def extract_cnn_features_batch(model, X_tensor, batch_size=64):
            model.eval()
            embeddings = []
            with torch.no_grad():
                for i in range(0, len(X_tensor), batch_size):
                    batch = X_tensor[i:i+batch_size].unsqueeze(1)
                    emb = model(batch)
                    embeddings.append(emb.cpu().numpy())
            return np.vstack(embeddings)

        with console.status(f"[bold]Fold {fold+1}: Extracting CNN features...[/bold]", spinner="dots"):
            X_train_cnn = extract_cnn_features_batch(cnn, X_train_t)
            X_val_cnn = extract_cnn_features_batch(cnn, X_val_t)
            
        # ---------------------------------------------------------------------
        # Statistical & WKS Features
        # ---------------------------------------------------------------------
        from stage4_optimizers.aquila_optimizer import WKSOptimizer
        from stage2_features.feature_extractor import extract_statistical_features
        
        with console.status(f"[bold]Fold {fold+1}: Extracting Statistical features...[/bold]", spinner="dots"):
            X_train_stat = extract_statistical_features(X_train)
            X_val_stat = extract_statistical_features(X_val)

        wks_opt = WKSOptimizer(pop_size=wks_pop_size, max_iter=wks_max_iter)
        
        with console.status(f"[bold]Fold {fold+1}: Optimizing WKS parameters...[/bold]", spinner="simpleDotsScrolling"):
            # Suppress logging inside the status context if possible by reducing log level temporarily or just rely on console
            optimal_omega, _, _ = wks_opt.optimize(X_train, y_train)
            
        X_train_wks = wks_opt.extract_wks_features(X_train, omega=optimal_omega)
        X_val_wks = wks_opt.extract_wks_features(X_val, omega=optimal_omega)
        
        # Combine Features
        X_train_combined = np.hstack([X_train_cnn, X_train_stat, X_train_wks])
        X_val_combined = np.hstack([X_val_cnn, X_val_stat, X_val_wks])
        
        # ---------------------------------------------------------------------
        # ORNN Training
        # ---------------------------------------------------------------------
        from stage3_models.ornn_model import ORNN, SIAOORNNTrainer
        
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
            bp_lr=0.001,
            weight_bounds=(-1.0, 1.0),
            fc_dropout=0.2,
            weight_decay=1e-5,
            patience=20
        )
        
        # Class Weights
        if use_class_weights:
            class_counts = np.bincount(y_train, minlength=num_classes)
            # Add 1 to avoid division by zero if a class is missing in a fold (unlikely with StratifiedKFold)
            weights = len(y_train) / (num_classes * (class_counts + 1))
            weights_t = torch.tensor(weights, dtype=torch.float32).to(device)
            trainer.criterion = nn.CrossEntropyLoss(weight=weights_t)
        else:
            trainer.criterion = nn.CrossEntropyLoss()
            
        # Prepare Data for RNN (Numpy format for trainer.train)
        # Ensure 3D shape: [samples, 1, features]
        if X_train_combined.ndim == 2:
            X_train_rnn_np = np.expand_dims(X_train_combined, axis=1)
            X_val_rnn_np = np.expand_dims(X_val_combined, axis=1)
        else:
            X_train_rnn_np = X_train_combined
            X_val_rnn_np = X_val_combined
            
        # Used for evaluation later
        X_train_rnn = torch.tensor(X_train_rnn_np, dtype=torch.float32).to(device)
        X_val_rnn = torch.tensor(X_val_rnn_np, dtype=torch.float32).to(device)
        
        # Train - Pass numpy arrays as expected by SIAOORNNTrainer
        # It handles tensor conversion and dataloader creation internally
        console.print("[bold yellow]DEBUG: Calling trainer.train...[/bold yellow]")
        console.print(f"X_train_rnn_np type: {type(X_train_rnn_np)} shape: {X_train_rnn_np.shape}")
        
        result_dict = trainer.train(
            X_train_rnn_np, y_train,
            X_val_rnn_np, y_val,
            batch_size=batch_size
        )
        history = result_dict['backprop']
        
        # Plot training results
        plot_training_results(history, fold_idx=fold, save_dir='results/plots')
        
        # Evaluate Fold
        ornn.eval()
        trainer.fc.eval()
        with torch.no_grad():
            ornn_out, _ = ornn(X_val_rnn)
            last_hidden = ornn_out[:, -1, :]
            outputs = trainer.fc(last_hidden)
            _, preds = torch.max(outputs, 1)
            
            # Plot confusion matrix
            plot_confusion_matrix_heatmap(
                y_val, preds.cpu().numpy(), 
                classes=[str(c) for c in range(num_classes)],
                fold_idx=fold,
                save_dir='results/plots'
            )
            
            fold_acc = accuracy_score(y_val, preds.cpu().numpy())
            fold_accuracies.append(fold_acc)
            
        console.print(f"[bold green]Fold {fold+1} Accuracy: {fold_acc*100:.2f}%[/bold green]")
        
    # =========================================================================
    # Final Results Aggregation
    # =========================================================================
    avg_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    
    console.print(Panel(f"[bold]5-Fold Cross-Validation Results[/bold]\n"
                        f"\nFold Scores: {[f'{x*100:.2f}%' for x in fold_accuracies]}\n"
                        f"\n[bold green]Average Accuracy: {avg_acc*100:.2f}% (+/- {std_acc*100:.2f}%)[/bold green]\n"
                        f"Target: 98.74%", title="Final Report", box=box.DOUBLE))

    return {
        'avg_accuracy': avg_acc,
        'fold_accuracies': fold_accuracies,
        'std_accuracy': std_acc
    }


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
        window_size=100, # Updated to 100
        stride=25,
        cnn_embedding_dim=256,
        wks_dim=43,
        rnn_hidden_size=128,
        num_classes=6,
        test_size=0.2, # Still used for fallback if needed, but CV overrides logic
        wks_pop_size=15,
        wks_max_iter=30,
        siao_pop_size=15,
        siao_max_iter=30,
        bp_epochs=100,
        batch_size=163
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
    print(f"Final Average Accuracy: {results['avg_accuracy']:.4f}")
