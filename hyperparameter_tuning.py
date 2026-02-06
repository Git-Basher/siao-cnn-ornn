"""
Hyperparameter Tuning with Optuna for SIAO-CNN-ORNN

This script uses Bayesian optimization to find optimal hyperparameters.
It runs a single fold of cross-validation per trial for efficiency.

Usage:
    uv run hyperparameter_tuning.py
"""

import optuna
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import logging
import warnings

# Suppress verbose logging during optimization
logging.getLogger('stage4_optimizers.aquila_optimizer').setLevel(logging.WARNING)
logging.getLogger('stage4_optimizers.siao_optimizer').setLevel(logging.WARNING)
warnings.filterwarnings('ignore')

def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function. Returns validation accuracy for a single fold.
    """
    # Sample hyperparameters
    rnn_hidden_size = trial.suggest_int('rnn_hidden_size', 64, 256, step=32)
    cnn_embedding_dim = trial.suggest_int('cnn_embedding_dim', 128, 512, step=64)
    bp_lr = trial.suggest_float('bp_lr', 1e-4, 1e-2, log=True)
    fc_dropout = trial.suggest_float('fc_dropout', 0.1, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    siao_pop_size = trial.suggest_int('siao_pop_size', 10, 30, step=5)
    siao_max_iter = trial.suggest_int('siao_max_iter', 20, 60, step=10)
    num_layers = trial.suggest_int('num_layers', 1, 2)
    
    # Fixed parameters
    window_size = 100
    stride = 25
    wks_pop_size = 15
    wks_max_iter = 30
    bp_epochs = 50  # Reduced for faster tuning
    batch_size = 163
    num_classes = 6
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data Loading (cached)
    from stage1_data.data_pipeline import IP200DataPipeline
    from stage1_data.window_processor import SlidingWindowProcessor
    
    pipeline = IP200DataPipeline(data_dir='data/')
    X_raw, y_raw = pipeline.run(use_cache=True)
    
    window_proc = SlidingWindowProcessor(window_size=window_size, stride=stride)
    X_windows, y_windows = window_proc.transform(X_raw, y_raw)
    
    # Single Fold for Speed
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(iter(skf.split(X_windows, y_windows)))
    
    X_train, X_val = X_windows[train_idx], X_windows[val_idx]
    y_train, y_val = y_windows[train_idx], y_windows[val_idx]
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    
    # Feature Extraction
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
    
    X_train_cnn = extract_cnn_features_batch(cnn, X_train_t)
    X_val_cnn = extract_cnn_features_batch(cnn, X_val_t)
    
    from stage4_optimizers.aquila_optimizer import WKSOptimizer
    from stage2_features.feature_extractor import extract_statistical_features
    
    X_train_stat = extract_statistical_features(X_train)
    X_val_stat = extract_statistical_features(X_val)
    
    wks_opt = WKSOptimizer(pop_size=wks_pop_size, max_iter=wks_max_iter)
    optimal_omega, _, _ = wks_opt.optimize(X_train, y_train)
    X_train_wks = wks_opt.extract_wks_features(X_train, omega=optimal_omega)
    X_val_wks = wks_opt.extract_wks_features(X_val, omega=optimal_omega)
    
    X_train_combined = np.hstack([X_train_cnn, X_train_stat, X_train_wks])
    X_val_combined = np.hstack([X_val_cnn, X_val_stat, X_val_wks])
    
    # ORNN Training
    from stage3_models.ornn_model import ORNN, SIAOORNNTrainer
    
    combined_input_size = X_train_combined.shape[1]
    
    ornn = ORNN(
        input_size=combined_input_size,
        hidden_size=rnn_hidden_size,
        num_layers=num_layers,
        cell_type='gru'
    )
    
    trainer = SIAOORNNTrainer(
        ornn=ornn,
        output_size=num_classes,
        device=device,
        siao_pop_size=siao_pop_size,
        siao_max_iter=siao_max_iter,
        bp_epochs=bp_epochs,
        bp_lr=bp_lr,
        weight_bounds=(-1.0, 1.0),
        fc_dropout=fc_dropout,
        weight_decay=weight_decay,
        patience=15
    )
    
    # Class Weights
    class_counts = np.bincount(y_train, minlength=num_classes)
    weights = len(y_train) / (num_classes * (class_counts + 1))
    weights_t = torch.tensor(weights, dtype=torch.float32).to(device)
    trainer.criterion = nn.CrossEntropyLoss(weight=weights_t)
    
    # Prepare Data
    if X_train_combined.ndim == 2:
        X_train_rnn_np = np.expand_dims(X_train_combined, axis=1)
        X_val_rnn_np = np.expand_dims(X_val_combined, axis=1)
    else:
        X_train_rnn_np = X_train_combined
        X_val_rnn_np = X_val_combined
    
    X_val_rnn = torch.tensor(X_val_rnn_np, dtype=torch.float32).to(device)
    
    # Train
    trainer.train(X_train_rnn_np, y_train, X_val_rnn_np, y_val, batch_size=batch_size)
    
    # Evaluate
    ornn.eval()
    trainer.fc.eval()
    with torch.no_grad():
        ornn_out, _ = ornn(X_val_rnn)
        last_hidden = ornn_out[:, -1, :]
        outputs = trainer.fc(last_hidden)
        _, preds = torch.max(outputs, 1)
        val_acc = accuracy_score(y_val, preds.cpu().numpy())
    
    return val_acc


def main():
    print("=" * 60)
    print("SIAO-CNN-ORNN Hyperparameter Tuning with Optuna")
    print("=" * 60)
    
    # Create study (maximize accuracy)
    study = optuna.create_study(
        direction='maximize',
        study_name='siao_cnn_ornn_tuning',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Run optimization
    n_trials = 20  # Adjust based on available time
    print(f"Running {n_trials} trials...")
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"\nBest Trial: {study.best_trial.number}")
    print(f"Best Validation Accuracy: {study.best_value:.4f}")
    print("\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    with open('results/optuna_best_params.txt', 'w') as f:
        f.write(f"Best Accuracy: {study.best_value:.4f}\n\n")
        f.write("Best Hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")
    
    print("\nResults saved to results/optuna_best_params.txt")
    

if __name__ == '__main__':
    import os
    os.makedirs('results', exist_ok=True)
    main()
