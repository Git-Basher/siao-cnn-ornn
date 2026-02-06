#!/usr/bin/env python3
"""
TimeGAN Data Augmentation Runner

Standalone script to train TimeGAN models and generate augmented data.
Run this before training the main model with --augment flag.

Usage:
    uv run run_augmentation.py --epochs 1000 --target-per-class 100
"""

import argparse
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic data using TimeGAN'
    )
    parser.add_argument(
        '--epochs', type=int, default=1000,
        help='Number of training epochs for TimeGAN (default: 1000)'
    )
    parser.add_argument(
        '--target-per-class', type=int, default=100,
        help='Target samples per class (default: 100, total ~600)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=16,
        help='Training batch size (default: 16)'
    )
    parser.add_argument(
        '--force-retrain', action='store_true',
        help='Force retraining even if models exist'
    )
    parser.add_argument(
        '--validate-only', action='store_true',
        help='Only validate existing augmented data'
    )
    parser.add_argument(
        '--data-dir', type=str, default='data/',
        help='Data directory (default: data/)'
    )
    args = parser.parse_args()
    
    # Rich console for nice output
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich import box
        console = Console()
        use_rich = True
    except ImportError:
        console = None
        use_rich = False
    
    def print_header(text):
        if use_rich:
            console.print(Panel(text, style="bold green"))
        else:
            print(f"\n{'='*60}\n{text}\n{'='*60}")
    
    print_header("TimeGAN Data Augmentation")
    
    # =========================================================================
    # Step 1: Load Original Data
    # =========================================================================
    print_header("Step 1: Loading Original Data")
    
    from stage1_data.data_pipeline import IP200DataPipeline
    from stage1_data.window_processor import WindowProcessor
    
    # Load data
    pipeline = IP200DataPipeline(data_dir=args.data_dir)
    X_raw, y_raw = pipeline.run(use_cache=True)
    
    logger.info(f"Loaded {len(X_raw)} raw samples")
    logger.info(f"Shape: {X_raw.shape}")
    
    # Create windows
    processor = WindowProcessor(window_size=100, stride=25)
    X_windows, y_windows = processor.create_windows(X_raw, y_raw)
    
    logger.info(f"Created {len(X_windows)} windows")
    
    # Print class distribution
    unique, counts = np.unique(y_windows, return_counts=True)
    
    if use_rich:
        table = Table(title="Original Class Distribution", box=box.ROUNDED)
        table.add_column("Class", style="cyan")
        table.add_column("Count", style="green")
        table.add_column("Target", style="yellow")
        table.add_column("To Generate", style="magenta")
        
        for cls, count in zip(unique, counts):
            to_gen = max(0, args.target_per_class - count)
            table.add_row(str(int(cls)), str(count), str(args.target_per_class), str(to_gen))
        
        console.print(table)
    else:
        logger.info("Class distribution:")
        for cls, count in zip(unique, counts):
            logger.info(f"  Class {cls}: {count} samples")
    
    # =========================================================================
    # Step 2: Train TimeGAN Models (or load existing)
    # =========================================================================
    print_header("Step 2: TimeGAN Training")
    
    from stage0_augmentation import TimeGANAugmentor
    
    augmentor = TimeGANAugmentor(
        seq_len=X_windows.shape[1],
        n_features=X_windows.shape[2]
    )
    
    # Check for existing augmented data
    if not args.force_retrain:
        cached = augmentor.load_augmented()
        if cached is not None:
            X_aug, y_aug = cached
            logger.info(f"Loaded cached augmented data: {len(X_aug)} samples")
            
            if not args.validate_only:
                logger.info("Use --force-retrain to regenerate")
    
    if args.validate_only:
        cached = augmentor.load_augmented()
        if cached is None:
            logger.error("No augmented data found! Run without --validate-only first.")
            return
        X_aug, y_aug = cached
    else:
        # Check for existing models
        models_loaded = augmentor.load_all_models()
        logger.info(f"Loaded {models_loaded} existing TimeGAN models")
        
        if models_loaded < len(unique) or args.force_retrain:
            logger.info("Training TimeGAN models...")
            logger.info(f"This may take ~30-60 minutes on CPU")
            augmentor.fit(
                X_windows, y_windows,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
        
        # Generate augmented dataset
        X_aug, y_aug = augmentor.augment_dataset(
            X_windows, y_windows,
            target_per_class=args.target_per_class
        )
        
        # Save
        augmentor.save_augmented(X_aug, y_aug)
    
    # =========================================================================
    # Step 3: Validate Synthetic Data Quality
    # =========================================================================
    print_header("Step 3: Quality Validation")
    
    from stage0_augmentation.quality_metrics import validate_synthetic_data
    
    # Separate real from synthetic for validation
    n_original = len(X_windows)
    X_synthetic = X_aug[n_original:]
    
    if len(X_synthetic) > 0:
        results = validate_synthetic_data(
            X_windows[:min(100, len(X_windows))],  # Subsample for speed
            X_synthetic[:min(100, len(X_synthetic))],
            output_dir='results/augmentation'
        )
        
        if use_rich:
            table = Table(title="Quality Metrics", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_column("Quality", style="yellow")
            
            table.add_row(
                "Discriminative Score",
                f"{results['discriminative_score']:.4f}",
                results['discriminative_quality']
            )
            table.add_row(
                "Predictive MAE (Real)",
                f"{results['predictive_mae_real']:.4f}",
                "-"
            )
            table.add_row(
                "Predictive MAE (Synthetic)",
                f"{results['predictive_mae_synthetic']:.4f}",
                results['predictive_quality']
            )
            
            console.print(table)
    else:
        logger.warning("No synthetic samples generated")
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print_header("Augmentation Complete!")
    
    # Final distribution
    unique_aug, counts_aug = np.unique(y_aug, return_counts=True)
    
    if use_rich:
        table = Table(title="Final Dataset Distribution", box=box.ROUNDED)
        table.add_column("Class", style="cyan")
        table.add_column("Original", style="blue")
        table.add_column("Augmented", style="green")
        
        for cls in unique:
            orig = counts[unique == cls][0] if cls in unique else 0
            aug = counts_aug[unique_aug == cls][0] if cls in unique_aug else 0
            table.add_row(str(int(cls)), str(orig), str(aug))
        
        table.add_row("TOTAL", str(len(X_windows)), str(len(X_aug)), style="bold")
        console.print(table)
        
        console.print(Panel(
            f"[bold green]Augmented data saved to: data/augmented/[/bold green]\n"
            f"Run training with: [bold cyan]uv run train_pipeline.py --augment[/bold cyan]",
            title="Next Steps"
        ))
    else:
        logger.info(f"Original: {len(X_windows)} samples")
        logger.info(f"Augmented: {len(X_aug)} samples")
        logger.info("Next: uv run train_pipeline.py --augment")


if __name__ == '__main__':
    main()
