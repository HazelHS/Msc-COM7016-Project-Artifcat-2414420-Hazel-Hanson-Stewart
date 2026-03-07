# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""xLSTM-TS Time-Series Forecasting model.

Implementation derived from Lopez et al. (2024) “xLSTM-TS: Extended Long
Short-Term Memory for Time-Series Forecasting”.

The model is constructed from three parts; sLSTM_Block (Scalar-memory LSTM layer with 
Conv1d + Multi-Head, Attention + feed-forward network), mLSTM_Block (Matrix-memory 
LSTM layer with ELU exponential gating and dual projection) and xLSTM_TS_Model (Full 
4-block stack (mLSTM → sLSTM → mLSTM → mLSTM, followed by 
attention-weighted temporal aggregation).
"""

import sys
from pathlib import Path

# Insert the project root so dependency_checker can be resolved regardless of the current working directory when the script is launched.
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_path))

from dependency_checker import *
import torch.nn.functional as F

class SequenceDataset(Dataset): # (Anthropic, 2026)
    """PyTorch Dataset for sliding-window time-series sequences.

    Args:
        features: Feature array of shape [N, sequence_length, n_features].
        targets: Target array of shape [N, forecast_horizon].
    """

    def __init__(self, features, targets): # (Anthropic, 2026)
        """Store feature and target arrays as instance attributes."""
        self.features = features
        self.targets = targets

    def __len__(self): # (Anthropic, 2026)
        """Return the total number of sequence samples."""
        return len(self.features)

    def __getitem__(self, idx): # (Anthropic, 2026)
        """Return the (feature_tensor, target_tensor) pair at index idx.

        Args:
            idx: Integer sample index.

        Returns:
            A tuple (X, y) of float32 tensors.
        """
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor(self.targets[idx])

def directional_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor: # (Anthropic, 2026)
    """Compute a combined MSE + directional-accuracy loss.

    Args:
        y_true: Ground-truth tensor of shape [batch, forecast_horizon].
        y_pred: Predicted tensor of shape [batch, forecast_horizon].

    Returns:
        A scalar loss tensor.
    """
    # MSE component
    mse = F.mse_loss(y_pred, y_true)
    
    # Directional component
    y_true_direction = (y_true[:, 1:] > y_true[:, :-1]).float()
    y_pred_direction = (y_pred[:, 1:] > y_pred[:, :-1]).float()
    
    # Binary cross-entropy for directional accuracy
    directional = F.binary_cross_entropy(y_pred_direction, y_true_direction)
    
    # Scale down the directional component to be comparable to MSE
    directional_scaled = directional / 10.0
    
    # Combine losses with weighting (α=0.7)
    alpha = 0.7
    # return alpha * mse + (1 - alpha) * directional_scaled
    return F.mse_loss(y_pred, y_true) # lopez paper's loss function

class sLSTM_Block(nn.Module): # (Anthropic, 2026)
    """Scalar-memory LSTM block.

    Applies LayerNorm, Conv1d, LSTM, MultiheadAttention, and a feed-forward
    network in sequence, then adds the block input as a residual.

    Args:
        input_dim: Number of input feature channels.
        embedding_dim: Output and hidden dimension for Conv, LSTM, and Attention.
        kernel_size: Convolution kernel width for the Conv1d layer.
        num_heads: Number of heads for MultiheadAttention.
        ff_factor: Expansion ratio for the feed-forward hidden layer;
          ff_dim = embedding_dim * ff_factor.
    """

    def __init__(self, input_dim, embedding_dim=64, kernel_size=2, num_heads=2, ff_factor=1.1): # (Anthropic, 2026)
        """Build all sub-layers of the sLSTM block."""
        super(sLSTM_Block, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.conv = nn.Conv1d(in_channels=input_dim, 
                              out_channels=embedding_dim, 
                              kernel_size=kernel_size, 
                              padding='same')
        self.lstm = nn.LSTM(input_size=embedding_dim, 
                            hidden_size=embedding_dim, 
                            batch_first=True)
        
        # PyTorch's MultiheadAttention uses different parameter naming
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim,
                                               num_heads=num_heads,
                                               batch_first=True)
        
        # Feedforward network with projection factor
        ff_dim = int(embedding_dim * ff_factor)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim)
        )

    def forward(self, x): # (Anthropic, 2026)
        """Compute the sLSTM block forward pass.

        Args:
            x: Input tensor of shape [batch, sequence_length, input_dim].

        Returns:
            Tensor of shape [batch, sequence_length, embedding_dim].
        """
        # Apply layer normalisation before the sub-layers (pre-LN style)
        norm_x = self.layer_norm(x)

        # Conv1d expects [batch, channels, sequence]; transpose in and out
        conv_x = self.conv(norm_x.transpose(1, 2)).transpose(1, 2)

        # Sequence modelling with LSTM; discard the hidden/cell states
        lstm_out, _ = self.lstm(conv_x)

        # Self-attention: query = key = value = lstm output (encoder-style)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Position-wise feed-forward then add input residual
        ff_out = self.feed_forward(attn_out)
        return x + ff_out


class mLSTM_Block(nn.Module): # (Anthropic, 2026)
    """Matrix-memory LSTM block with ELU+1 exponential gating.

    Applies LayerNorm, Conv1d with ELU+1 activation, LSTM, MultiheadAttention,
    and a two-layer projection bottleneck, then adds the block input as a residual.

    Args:
        input_dim: Number of input feature channels.
        embedding_dim: Hidden dimension for all sub-layers.
        kernel_size: Conv1d kernel width.
        projection_size: Multiplier applied to embedding_dim in the first
          projection layer before projecting back down.
        num_heads: Number of MultiheadAttention heads.
    """

    def __init__(self, input_dim, embedding_dim=64, kernel_size=4, projection_size=2, num_heads=2): # (Anthropic, 2026)
        """Build Conv, ELU, LSTM, Attention, and projection sub-layers."""
        super(mLSTM_Block, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        # Fix padding to preserve sequence length
        padding = kernel_size // 2
        #else:
        #    padding = kernel_size // 2 - 1
        self.conv = nn.Conv1d(in_channels=input_dim,
                             out_channels=embedding_dim,
                             kernel_size=kernel_size,
                             padding='same')
        self.elu = nn.ELU()
        self.lstm = nn.LSTM(input_size=embedding_dim,
                           hidden_size=embedding_dim,
                           batch_first=True)
        
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim,
                                              num_heads=num_heads,
                                              batch_first=True)
        
        # Two-layer projection bottleneck: expand then compress back to embedding_dim
        self.projection1 = nn.Linear(embedding_dim, embedding_dim * projection_size)
        self.projection2 = nn.Linear(embedding_dim * projection_size, embedding_dim)

    def forward(self, x): # (Anthropic, 2026)
        """Compute the mLSTM block forward pass.

        Args:
            x: Input tensor of shape [batch, sequence_length, input_dim].

        Returns:
            Tensor of shape [batch, sequence_length, embedding_dim].
        """
        # Apply layer normalisation before the sub-layers (pre-LN style)
        norm_x = self.layer_norm(x)

        # Conv1d + ELU + 1.0: the +1 keeps values positive, approximating the exponential gate in the matrix-memory LSTM formulation.
        conv_x = self.elu(self.conv(norm_x.transpose(1, 2))).transpose(1, 2) + 1.0

        # Sequence modelling; hidden/cell states discarded (stateless)
        lstm_out, _ = self.lstm(conv_x)

        # Self-attention over the full sequence
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Bottleneck projection: expand then compress back to embedding_dim
        proj = self.projection1(attn_out)
        proj = self.projection2(proj)

        # Add input residual so gradients flow cleanly through the block
        return x + proj

class xLSTM_TS_Model(nn.Module): # (Anthropic, 2026)
    """Complete xLSTM-TS model following Lopez et al. (2024).

    Stacks four xLSTM blocks (mLSTM → sLSTM → mLSTM → mLSTM), then
    aggregates the output sequence via attention-weighted temporal pooling
    before a linear projection produces the multi-step forecast.

    Args:
        input_shape: Tuple (sequence_length, n_features) describing the
          look-back window dimensions.
        embedding_dim: Common hidden dimension used across all blocks.
        output_size: Number of future timesteps to predict.
    """

    def __init__(self, input_shape, embedding_dim=64, output_size=7): # (Anthropic, 2026)
        """Build all layers: initial projection, four xLSTM blocks, layer norm,
        attention aggregator, and output projection."""
        super(xLSTM_TS_Model, self).__init__()
        
        sequence_length, n_features = input_shape
        
        # Initial linear projection to embedding dimension
        self.initial_projection = nn.Linear(n_features, embedding_dim)
        
        # xLSTM block stack (4 blocks: mLSTM → sLSTM → mLSTM → mLSTM)
        self.mlstm_block1 = mLSTM_Block(embedding_dim, 
                                       embedding_dim=embedding_dim,
                                       kernel_size=4,
                                       projection_size=2,
                                       num_heads=2)
        
        self.slstm_block = sLSTM_Block(embedding_dim,
                                      embedding_dim=embedding_dim,
                                      kernel_size=2,
                                      num_heads=2,
                                      ff_factor=1.1)
        
        self.mlstm_block2 = mLSTM_Block(embedding_dim,
                                       embedding_dim=embedding_dim,
                                       kernel_size=4,
                                       projection_size=2,
                                       num_heads=2)
        
        self.mlstm_block3 = mLSTM_Block(embedding_dim,
                                       embedding_dim=embedding_dim,
                                       kernel_size=4,
                                       projection_size=2,
                                       num_heads=2)
        
        # Final layer normalization
        self.final_layer_norm = nn.LayerNorm(embedding_dim)
        
        # ADD: Attention-based temporal aggregation (matching TensorFlow)
        self.attention_weights = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Tanh()
        )
        # Final linear projection to output
        self.output_projection = nn.Linear(embedding_dim, output_size)
        
    def forward(self, x): # (Anthropic, 2026)
        """Run a forward pass through the full xLSTM-TS model.

        Args:
            x: Input tensor of shape [batch, sequence_length, n_features].

        Returns:
            Prediction tensor of shape [batch, output_size].
        """
        # Project raw features to the shared embedding dimension
        x = self.initial_projection(x)      # [batch, seq_len, embedding_dim]

        # Pass sequentially through the 4-block xLSTM stack
        x = self.mlstm_block1(x)            # mLSTM block 1
        x = self.slstm_block(x)             # sLSTM block
        x = self.mlstm_block2(x)            # mLSTM block 2
        x = self.mlstm_block3(x)            # mLSTM block 3

        # Layer-normalise the block stack output (pre-aggregation)
        x = self.final_layer_norm(x)

        # Attention-weighted temporal pooling: learn which timesteps matter
        attention_scores = self.attention_weights(x)          # [batch, seq_len, 1]
        attention_weights = F.softmax(attention_scores, dim=1) # normalise over time
        x = torch.sum(x * attention_weights, dim=1)           # [batch, embedding_dim]

        # Map aggregated embedding to the multi-step forecast horizon
        x = self.output_projection(x)       # [batch, output_size]

        return x

class TrainingProgressTracker: # (Anthropic, 2026)
    """Epoch- and batch-level tqdm progress tracker for the training loop.

    A new tqdm bar is created at the start of each epoch and updated after
    every batch, then closed at the end of the epoch.

    Args:
        total_epochs: Total number of training epochs.
        steps_per_epoch: Number of mini-batches per epoch.
    """

    def __init__(self, total_epochs: int, steps_per_epoch: int) -> None: # (Anthropic, 2026)
        """Initialise bookkeeping variables; no tqdm bar is created yet."""
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.current_epoch = 0
        self.pbar = None    # tqdm bar; created at the start of each epoch

    def update_batch(self, batch_loss: float, batch_mae: float) -> None: # (Anthropic, 2026)
        """Advance the tqdm bar by one step and refresh the postfix metrics.

        Args:
            batch_loss: Mini-batch training loss.
            batch_mae: Mini-batch mean absolute error.
        """
        if self.pbar is not None:
            self.pbar.set_postfix({
                'loss': f"{batch_loss:.4f}",
                'mae': f"{batch_mae:.4f}",
                'epoch': f"{self.current_epoch}/{self.total_epochs}"
            })
            self.pbar.update(1)

    def update_epoch(
        self,
        epoch_loss: float,
        val_loss: float | None = None,
        epoch_mae: float | None = None,
    ) -> None: # (Anthropic, 2026)
        """Close the current epoch's tqdm bar and open a fresh one for the next epoch.

        Args:
            epoch_loss: Mean training loss for the completed epoch.
            val_loss: Optional validation loss for the completed epoch.
            epoch_mae: Optional mean absolute error for the completed epoch.
        """
        # Close the bar that belonged to the just-finished epoch
        if self.pbar is not None:
            self.pbar.close()

        self.current_epoch += 1

        # Create a fresh bar for the upcoming epoch (if any remain)
        if self.current_epoch < self.total_epochs:
            from tqdm import tqdm
            self.pbar = tqdm(
                total=self.steps_per_epoch,
                desc='Training Progress',
                unit='batch',
                postfix={
                    'loss': f"{epoch_loss:.4f}",
                    'mae': f"{epoch_mae:.4f}" if epoch_mae is not None else 'N/A',
                    'epoch': f"{self.current_epoch}/{self.total_epochs}"
                }
            )

    def close(self) -> None: # (Anthropic, 2026)
        """Close the active tqdm bar if one is open."""
        if self.pbar is not None:
            self.pbar.close()
