"""
Optimized RNN (ORNN) with SIAO Weight Optimization

Implements a Recurrent Neural Network whose initial weights are
optimized using Self-Improved Aquila Optimizer (SIAO) followed
by fine-tuning with backpropagation.

Architecture:
- GRU-based recurrent layer (configurable to vanilla RNN)
- SIAO optimization for weight initialization
- Backpropagation fine-tuning
- Output embeddings for classification head

Author: Recurrent Neural Network Specialist
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable
import logging
from tqdm import tqdm, trange
from rich.console import Console

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# Use Console instead of standard logger for visual consistency
console = Console()
logger = logging.getLogger(__name__)


# =============================================================================
# Optimized RNN Cell
# =============================================================================

class ORNNCell(nn.Module):
    """
    Optimized RNN Cell with exposed weights for SIAO optimization.
    
    Implements: h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True
    ):
        super(ORNNCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Learnable parameters (exposed for SIAO)
        self.W_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        
        if bias:
            self.b_ih = nn.Parameter(torch.Tensor(hidden_size))
            self.b_hh = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('b_ih', None)
            self.register_parameter('b_hh', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters with Xavier uniform."""
        nn.init.xavier_uniform_(self.W_ih)
        nn.init.xavier_uniform_(self.W_hh)
        if self.b_ih is not None:
            nn.init.zeros_(self.b_ih)
            nn.init.zeros_(self.b_hh)
    
    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [batch, input_size]
            h: Hidden state [batch, hidden_size]
        
        Returns:
            New hidden state [batch, hidden_size]
        """
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        
        # RNN computation
        h_new = torch.tanh(
            F.linear(x, self.W_ih, self.b_ih) +
            F.linear(h, self.W_hh, self.b_hh)
        )
        
        return h_new
    
    def get_weight_vector(self) -> np.ndarray:
        """Flatten all weights into a single vector for SIAO."""
        weights = []
        weights.append(self.W_ih.data.cpu().numpy().flatten())
        weights.append(self.W_hh.data.cpu().numpy().flatten())
        if self.b_ih is not None:
            weights.append(self.b_ih.data.cpu().numpy())
            weights.append(self.b_hh.data.cpu().numpy())
        return np.concatenate(weights)
    
    def set_weight_vector(self, weights: np.ndarray, device: torch.device):
        """Set weights from flattened vector."""
        idx = 0
        
        # W_ih
        size = self.hidden_size * self.input_size
        self.W_ih.data = torch.tensor(
            weights[idx:idx+size].reshape(self.hidden_size, self.input_size),
            dtype=torch.float32, device=device
        )
        idx += size
        
        # W_hh
        size = self.hidden_size * self.hidden_size
        self.W_hh.data = torch.tensor(
            weights[idx:idx+size].reshape(self.hidden_size, self.hidden_size),
            dtype=torch.float32, device=device
        )
        idx += size
        
        # Biases
        if self.b_ih is not None:
            size = self.hidden_size
            self.b_ih.data = torch.tensor(
                weights[idx:idx+size], dtype=torch.float32, device=device
            )
            idx += size
            self.b_hh.data = torch.tensor(
                weights[idx:idx+size], dtype=torch.float32, device=device
            )


# =============================================================================
# Optimized GRU Cell
# =============================================================================

class OGRUCell(nn.Module):
    """
    Optimized GRU Cell with exposed weights for SIAO optimization.
    
    GRU equations:
    r_t = sigmoid(W_ir * x_t + W_hr * h_{t-1} + b_r)
    z_t = sigmoid(W_iz * x_t + W_hz * h_{t-1} + b_z)
    n_t = tanh(W_in * x_t + r_t * (W_hn * h_{t-1}) + b_n)
    h_t = (1 - z_t) * n_t + z_t * h_{t-1}
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True
    ):
        super(OGRUCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input weights (3 gates: reset, update, new)
        self.W_ir = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_iz = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_in = nn.Parameter(torch.Tensor(hidden_size, input_size))
        
        # Hidden weights
        self.W_hr = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_hz = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_hn = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        
        if bias:
            self.b_ir = nn.Parameter(torch.Tensor(hidden_size))
            self.b_iz = nn.Parameter(torch.Tensor(hidden_size))
            self.b_in = nn.Parameter(torch.Tensor(hidden_size))
            self.b_hr = nn.Parameter(torch.Tensor(hidden_size))
            self.b_hz = nn.Parameter(torch.Tensor(hidden_size))
            self.b_hn = nn.Parameter(torch.Tensor(hidden_size))
        else:
            for name in ['b_ir', 'b_iz', 'b_in', 'b_hr', 'b_hz', 'b_hn']:
                self.register_parameter(name, None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize with Xavier uniform."""
        for name, param in self.named_parameters():
            if 'W_' in name:
                nn.init.xavier_uniform_(param)
            elif 'b_' in name:
                nn.init.zeros_(param)
    
    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass."""
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        
        # Reset gate
        r = torch.sigmoid(
            F.linear(x, self.W_ir, self.b_ir) +
            F.linear(h, self.W_hr, self.b_hr)
        )
        
        # Update gate
        z = torch.sigmoid(
            F.linear(x, self.W_iz, self.b_iz) +
            F.linear(h, self.W_hz, self.b_hz)
        )
        
        # New gate
        n = torch.tanh(
            F.linear(x, self.W_in, self.b_in) +
            r * F.linear(h, self.W_hn, self.b_hn)
        )
        
        # Hidden state
        h_new = (1 - z) * n + z * h
        
        return h_new
    
    def get_weight_vector(self) -> np.ndarray:
        """Flatten all weights for SIAO."""
        weights = []
        for name, param in self.named_parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)
    
    def set_weight_vector(self, weights: np.ndarray, device: torch.device):
        """Set weights from vector."""
        idx = 0
        for name, param in self.named_parameters():
            size = param.numel()
            param.data = torch.tensor(
                weights[idx:idx+size].reshape(param.shape),
                dtype=torch.float32, device=device
            )
            idx += size


# =============================================================================
# Optimized RNN (ORNN) Module
# =============================================================================

class ORNN(nn.Module):
    """
    Optimized RNN with SIAO weight initialization.
    
    Supports both vanilla RNN and GRU variants.
    Output can be used as input to fully connected classifier.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        cell_type: str = 'gru',  # 'rnn' or 'gru'
        dropout: float = 0.0,
        bidirectional: bool = False
    ):
        super(ORNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Build RNN layers
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            
            if cell_type == 'gru':
                cell = OGRUCell(layer_input_size, hidden_size)
            else:
                cell = ORNNCell(layer_input_size, hidden_size)
            
            self.cells.append(cell)
            
            if bidirectional:
                if cell_type == 'gru':
                    cell_bw = OGRUCell(layer_input_size, hidden_size)
                else:
                    cell_bw = ORNNCell(layer_input_size, hidden_size)
                self.cells.append(cell_bw)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        logger.info(f"ORNN: input={input_size}, hidden={hidden_size}, "
                   f"layers={num_layers}, type={cell_type}, bidir={bidirectional}")
    
    def forward(
        self,
        x: torch.Tensor,
        h_0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input sequence [batch, seq_len, input_size]
            h_0: Initial hidden state [num_layers * num_directions, batch, hidden_size]
        
        Returns:
            output: Sequence of hidden states [batch, seq_len, hidden_size * num_directions]
            h_n: Final hidden state [num_layers * num_directions, batch, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Initialize hidden states
        if h_0 is None:
            h_0 = torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size,
                device=device
            )
        
        # Process each layer
        layer_input = x
        h_n_list = []
        
        for layer in range(self.num_layers):
            # Forward direction
            fw_idx = layer * self.num_directions
            h_fw = h_0[fw_idx]
            outputs_fw = []
            
            for t in range(seq_len):
                h_fw = self.cells[fw_idx](layer_input[:, t, :], h_fw)
                outputs_fw.append(h_fw)
            
            outputs_fw = torch.stack(outputs_fw, dim=1)
            h_n_list.append(h_fw)
            
            if self.bidirectional:
                # Backward direction
                bw_idx = fw_idx + 1
                h_bw = h_0[bw_idx]
                outputs_bw = []
                
                for t in range(seq_len - 1, -1, -1):
                    h_bw = self.cells[bw_idx](layer_input[:, t, :], h_bw)
                    outputs_bw.insert(0, h_bw)
                
                outputs_bw = torch.stack(outputs_bw, dim=1)
                h_n_list.append(h_bw)
                
                layer_input = torch.cat([outputs_fw, outputs_bw], dim=2)
            else:
                layer_input = outputs_fw
            
            # Apply dropout between layers
            if layer < self.num_layers - 1:
                if not isinstance(layer_input, torch.Tensor):
                    logger.error(f"Dropout Input Type Error: Expected Tensor, got {type(layer_input)}")
                layer_input = self.dropout_layer(layer_input)
        
        h_n = torch.stack(h_n_list, dim=0)
        
        return layer_input, h_n
    
    def get_output_size(self) -> int:
        """Return output feature size."""
        return self.hidden_size * self.num_directions
    
    def get_weight_vector(self) -> np.ndarray:
        """Get all weights as a flat vector for SIAO."""
        weights = []
        for cell in self.cells:
            weights.append(cell.get_weight_vector())
        return np.concatenate(weights)
    
    def set_weight_vector(self, weights: np.ndarray, device: torch.device):
        """Set weights from flat vector."""
        idx = 0
        for cell in self.cells:
            cell_weights = cell.get_weight_vector()
            size = len(cell_weights)
            cell.set_weight_vector(weights[idx:idx+size], device)
            idx += size
    
    def get_weight_dim(self) -> int:
        """Get total number of weights."""
        return len(self.get_weight_vector())


# =============================================================================
# SIAO-Optimized RNN Trainer
# =============================================================================

class SIAOORNNTrainer:
    """
    Trainer for ORNN with SIAO weight optimization.
    
    Training process:
    1. Use SIAO to find optimal weight initialization
    2. Fine-tune with backpropagation
    """
    
    def __init__(
        self,
        ornn: ORNN,
        output_size: int,
        device: torch.device,
        siao_pop_size: int = 20,
        siao_max_iter: int = 50,
        bp_epochs: int = 100,
        bp_lr: float = 0.001,
        weight_bounds: Tuple[float, float] = (-1.0, 1.0),
        fc_dropout: float = 0.5,
        weight_decay: float = 1e-4,
        patience: Optional[int] = 20
    ):
        """
        Initialize trainer.
        
        Args:
            ornn: ORNN model
            output_size: Number of output classes
            device: GPU/CPU device
            siao_pop_size: SIAO population size
            siao_max_iter: SIAO iterations
            bp_epochs: Backpropagation epochs
            bp_lr: Learning rate for BP
            weight_bounds: Weight bounds for SIAO
        """
        self.ornn = ornn.to(device)
        self.device = device
        self.siao_pop_size = siao_pop_size
        self.siao_max_iter = siao_max_iter
        self.bp_epochs = bp_epochs
        self.bp_lr = bp_lr
        self.weight_bounds = weight_bounds
        
        # Output layer
        self.bp_lr = bp_lr
        self.weight_bounds = weight_bounds
        self.weight_decay = weight_decay
        self.patience = patience
        
        # Output layer with Dropout
        self.fc = nn.Sequential(
            nn.Dropout(fc_dropout) if fc_dropout > 0 else nn.Identity(),
            nn.Linear(ornn.get_output_size(), output_size)
        ).to(device)
        
        # Loss function (default, can be overridden)
        self.criterion = nn.CrossEntropyLoss()
        
        # Track training history
        self.siao_history = []
        self.bp_history = []
    
    def _create_objective(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> Callable:
        """Create objective function for SIAO."""
        
        def objective(weights: np.ndarray) -> float:
            # Set weights
            self.ornn.set_weight_vector(weights, self.device)
            
            # Forward pass
            with torch.no_grad():
                output, _ = self.ornn(X)
                # Use last hidden state
                last_hidden = output[:, -1, :]
                
                try:
                    logits = self.fc(last_hidden)
                except Exception as e:
                    print(f"CRASH IN OBJECTIVE: {e}")
                    print(f"last_hidden type: {type(last_hidden)}")
                    print(f"last_hidden shape: {last_hidden.shape if hasattr(last_hidden, 'shape') else 'no shape'}")
                    raise e
                
                # Compute loss
                loss = self.criterion(logits, y)
            
            return loss.item()
        
        return objective
    
    def siao_optimize(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor
    ) -> np.ndarray:
        """
        Optimize ORNN weights using SIAO.
        
        Args:
            X_train: Training data [batch, seq_len, features]
            y_train: Training labels [batch]
        
        Returns:
            Optimized weight vector
        """
        from stage4_optimizers.siao_optimizer import SelfImprovedAquilaOptimizer
        
        logger.info("=" * 60)
        logger.info("Starting SIAO Weight Optimization for ORNN")
        logger.info("=" * 60)
        
        # Get weight dimension
        weight_dim = self.ornn.get_weight_dim()
        logger.info(f"Weight dimension: {weight_dim}")
        
        # Create bounds
        lb = self.weight_bounds[0] * np.ones(weight_dim)
        ub = self.weight_bounds[1] * np.ones(weight_dim)
        
        # Create objective
        objective = self._create_objective(X_train, y_train)
        
        # Run SIAO
        siao = SelfImprovedAquilaOptimizer(
            objective_func=objective,
            dim=weight_dim,
            lb=lb,
            ub=ub,
            pop_size=self.siao_pop_size,
            max_iter=self.siao_max_iter,
            chaos_method='combined',
            minimize=True
        )
        
        best_weights, best_loss, info = siao.optimize()
        
        # Set optimized weights
        self.ornn.set_weight_vector(best_weights, self.device)
        self.siao_history = info['history'].tolist()
        
        logger.info(f"SIAO optimization complete. Best loss: {best_loss:.4f}")
        
        return best_weights
    
    def backprop_finetune(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict:
        """
        Fine-tune with backpropagation.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        
        Returns:
            Training history dict
        """
        logger.info("=" * 60)
        logger.info("Starting Backpropagation Fine-tuning")
        logger.info("=" * 60)
        
        # Combine parameters
        params = list(self.ornn.parameters()) + list(self.fc.parameters())
        optimizer = optim.Adam(params, lr=self.bp_lr, weight_decay=self.weight_decay)
        # Change to StepLR as per requested specs
        # Scaled step_size to 40 to match research paper's iteration count (125 iters)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=40, gamma=0.2
        )
        
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        # Early Stopping Variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        console.print(f"[bold]Starting Grid Search / Backprop finetuning for {self.bp_epochs} epochs...[/bold]")
        
        # Use trange for epoch progress
        epoch_pbar = trange(self.bp_epochs, desc="Training Epochs", leave=True)
        
        for epoch in epoch_pbar:
            # Training
            self.ornn.train()
            self.fc.train()
            
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            # Use tqdm for batch progress
            batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False, total=len(train_loader))
            
            for X_batch, y_batch in batch_pbar:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward
                output, _ = self.ornn(X_batch)
                last_hidden = output[:, -1, :]
                logits = self.fc(last_hidden)
                
                loss = self.criterion(logits, y_batch)
                
                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = logits.max(1)
                correct += predicted.eq(y_batch).sum().item()
                total += y_batch.size(0)
                
                # Update batch pbar
                batch_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            train_loss = epoch_loss / len(train_loader)
            train_acc = correct / total
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation
            val_loss = 0.0
            val_acc = 0.0
            if val_loader:
                val_loss, val_acc = self._evaluate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
            
            # Scheduler step (StepLR steps at the end of epoch, not dependent on validation metric)
            scheduler.step()
            
            # Early Stopping Check
            if self.patience is not None and val_loader: # Only apply if patience is set and validation data is provided
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    best_model_state = {
                        'ornn': self.ornn.state_dict(),
                        'fc': self.fc.state_dict()
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        console.print(f"[bold yellow]Early stopping triggered at epoch {epoch+1}[/bold yellow]")
                        # Restore best model
                        if best_model_state:
                            self.ornn.load_state_dict(best_model_state['ornn'])
                            self.fc.load_state_dict(best_model_state['fc'])
                        break # Exit the training loop
            
            # Update epoch description
            epoch_pbar.set_description(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            
            if (epoch + 1) % 50 == 0:
                 # Minimal logging to console to keep history
                 logger.info(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        # If training finished without early stopping, and early stopping was enabled,
        # restore the best model if it was better than the final model.
        # This ensures we always return the model with the best validation performance.
        if self.patience is not None and val_loader and best_model_state and patience_counter < self.patience:
             # This condition means the loop completed, but we still want the best model found
             # if the last epoch wasn't the best.
             # The `break` above handles the case where patience ran out.
             # If the loop finished naturally, `best_model_state` would hold the best.
             # If the last epoch was the best, `best_model_state` would be the current state.
             # So, simply restoring `best_model_state` is generally the right approach.
             self.ornn.load_state_dict(best_model_state['ornn'])
             self.fc.load_state_dict(best_model_state['fc'])
             console.print(f"[bold green]Restored best model based on validation loss.[/bold green]")

        self.bp_history = history
        return history
    
    def _evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on data loader."""
        self.ornn.eval()
        self.fc.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                output, _ = self.ornn(X_batch)
                last_hidden = output[:, -1, :]
                logits = self.fc(last_hidden)
                
                loss = self.criterion(logits, y_batch)
                total_loss += loss.item()
                
                _, predicted = logits.max(1)
                correct += predicted.eq(y_batch).sum().item()
                total += y_batch.size(0)
        
        return total_loss / len(data_loader), correct / total
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        batch_size: int = 32
    ) -> Dict:
        """
        Full training pipeline: SIAO + Backprop.
        
        Args:
            X_train: Training data [samples, seq_len, features]
            y_train: Training labels [samples]
            X_val: Validation data
            y_val: Validation labels
            batch_size: Batch size for BP
        
        Returns:
            Training history
        """
        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.long, device=self.device)
        
        # Step 1: SIAO optimization
        self.siao_optimize(X_train_t, y_train_t)
        
        # Step 2: Backprop fine-tuning
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        history = self.backprop_finetune(train_loader, val_loader)
        
        return {'siao': self.siao_history, 'backprop': history}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        self.ornn.eval()
        self.fc.eval()
        
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            output, _ = self.ornn(X_t)
            last_hidden = output[:, -1, :]
            logits = self.fc(last_hidden)
            predictions = logits.argmax(dim=1)
        
        return predictions.cpu().numpy()
    
    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Get ORNN embeddings (for downstream tasks)."""
        self.ornn.eval()
        
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            output, _ = self.ornn(X_t)
            embeddings = output[:, -1, :]  # Last hidden state
        
        return embeddings.cpu().numpy()


# =============================================================================
# Visualization
# =============================================================================

def plot_ornn_training(history: Dict, save_path: Optional[str] = None):
    """Plot ORNN training history."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # SIAO convergence
        if 'siao' in history and history['siao']:
            axes[0].plot(history['siao'], 'b-', linewidth=2)
            axes[0].set_xlabel('Iteration', fontsize=12)
            axes[0].set_ylabel('Loss', fontsize=12)
            axes[0].set_title('SIAO Weight Optimization', fontsize=14)
            axes[0].grid(True, alpha=0.3)
        
        # BP loss
        if 'backprop' in history:
            bp = history['backprop']
            axes[1].plot(bp['train_loss'], 'b-', label='Train', linewidth=2)
            if bp['val_loss']:
                axes[1].plot(bp['val_loss'], 'r--', label='Val', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Loss', fontsize=12)
            axes[1].set_title('Backprop Fine-tuning Loss', fontsize=14)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # BP accuracy
            axes[2].plot(bp['train_acc'], 'b-', label='Train', linewidth=2)
            if bp['val_acc']:
                axes[2].plot(bp['val_acc'], 'r--', label='Val', linewidth=2)
            axes[2].set_xlabel('Epoch', fontsize=12)
            axes[2].set_ylabel('Accuracy', fontsize=12)
            axes[2].set_title('Backprop Fine-tuning Accuracy', fontsize=14)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Training plot saved to {save_path}")
        
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib not available")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("Optimized RNN (ORNN) with SIAO - Demo")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create dummy data
    num_samples = 200
    seq_len = 50
    input_size = 43
    num_classes = 6
    
    X = np.random.randn(num_samples, seq_len, input_size).astype(np.float32)
    y = np.random.randint(0, num_classes, size=num_samples)
    
    print(f"Data: X={X.shape}, y={y.shape}")
    
    # Create ORNN
    ornn = ORNN(
        input_size=input_size,
        hidden_size=64,
        num_layers=1,
        cell_type='gru'
    )
    
    print(f"ORNN weight dimension: {ornn.get_weight_dim()}")
    
    # Create trainer
    trainer = SIAOORNNTrainer(
        ornn=ornn,
        output_size=num_classes,
        device=device,
        siao_pop_size=10,
        siao_max_iter=20,
        bp_epochs=50,
        bp_lr=0.001
    )
    
    # Train
    history = trainer.train(X, y, batch_size=32)
    
    # Evaluate
    predictions = trainer.predict(X)
    accuracy = (predictions == y).mean()
    print(f"\nFinal Accuracy: {accuracy:.4f}")
    
    # Plot
    plot_ornn_training(history)
