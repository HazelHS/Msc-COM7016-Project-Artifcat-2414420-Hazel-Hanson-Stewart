# Add the dedpendancies to the system path
import sys
from pathlib import Path

# Add the project root directory to sys.path
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_path))

from dependency_checker import *
import torch.nn.functional as F

class SequenceDataset(Dataset):
    """PyTorch Dataset for time series sequences"""
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor(self.targets[idx])

def directional_loss(y_true, y_pred):
    """
    Custom loss function that combines MSE with directional accuracy
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Combined loss value
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


class sLSTM_Block(nn.Module):
    """
    Scalar Memory LSTM block with convolutional processing
    """
    def __init__(self, input_dim, embedding_dim=64, kernel_size=2, num_heads=2, ff_factor=1.1):
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
        
    def forward(self, x):
        # Layer normalization
        norm_x = self.layer_norm(x)
        
        # Transpose for Conv1d which expects [batch, channels, sequence]
        # While our input is [batch, sequence, features]
        conv_x = self.conv(norm_x.transpose(1, 2)).transpose(1, 2)
        
        # LSTM layer
        lstm_out, _ = self.lstm(conv_x)
        
        # Multi-head attention - PyTorch requires different params than TF
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Feedforward network
        ff_out = self.feed_forward(attn_out)
        
        # Residual connection
        return x + ff_out


class mLSTM_Block(nn.Module):
    """
    More specialized approximation with exponential-like behavior
    """
    def __init__(self, input_dim, embedding_dim=64, kernel_size=4, projection_size=2, num_heads=2):
        super(mLSTM_Block, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        # Fix padding to preserve sequence length
        #if kernel_size % 2 == 1:
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
        
        # Projection to match dimensions
        self.projection1 = nn.Linear(embedding_dim, embedding_dim * projection_size)
        self.projection2 = nn.Linear(embedding_dim * projection_size, embedding_dim)
        
    def forward(self, x):
        # Layer normalization
        norm_x = self.layer_norm(x)
        
        # Conv layer with ELU+1 to approximate exponential gating
        conv_x = self.elu(self.conv(norm_x.transpose(1, 2))).transpose(1, 2) + 1.0
        
        # LSTM layer
        lstm_out, _ = self.lstm(conv_x)
        
        # Multi-head attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Projection
        proj = self.projection1(attn_out)
        proj = self.projection2(proj)
        
        # Residual connection
        return x + proj


class xLSTM_TS_Model(nn.Module):
    """
    Complete xLSTM-TS model matching Lopez et al. (2024) specifications
    """
    def __init__(self, input_shape, embedding_dim=64, output_size=7):
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
        
    def forward(self, x):
        # x shape: [batch, sequence_length, n_features]
        
        # Initial projection
        x = self.initial_projection(x)
        
        # Apply xLSTM blocks
        x = self.mlstm_block1(x)
        x = self.slstm_block(x)
        x = self.mlstm_block2(x)
        x = self.mlstm_block3(x)
        
        # Final normalization
        x = self.final_layer_norm(x)
        
        # NEW: Attention-based weighted sum
        attention_scores = self.attention_weights(x)  # [batch, seq_len, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch, seq_len, 1]
        x = torch.sum(x * attention_weights, dim=1)  # [batch, embedding_dim]
        
        # Final projection to output size
        x = self.output_projection(x)
        
        return x


class TrainingProgressTracker:
    """Replacement for Keras TrainingProgressCallback"""
    def __init__(self, total_epochs, steps_per_epoch):
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.current_epoch = 0
        self.pbar = None
        
    def update_batch(self, batch_loss, batch_mae):
        if self.pbar is not None:
            self.pbar.set_postfix({
                'loss': f"{batch_loss:.4f}",
                'mae': f"{batch_mae:.4f}",
                'epoch': f"{self.current_epoch}/{self.total_epochs}"
            })
            self.pbar.update(1)
        
    def update_epoch(self, epoch_loss, val_loss=None, epoch_mae=None):
        if self.pbar is not None:
            self.pbar.close()
        
        self.current_epoch += 1
        
        # Create new progress bar for next epoch
        if self.current_epoch < self.total_epochs:
            from tqdm import tqdm
            self.pbar = tqdm(
                total=self.steps_per_epoch,
                desc='Training Progress',
                unit='batch',
                postfix={
                    'loss': f"{epoch_loss:.4f}",
                    'mae': f"{epoch_mae:.4f}" if epoch_mae is not None else 'N/A',  # <--- FIX THIS LINE
                    'epoch': f"{self.current_epoch}/{self.total_epochs}"
                }
            )
        
    def close(self):
        if self.pbar is not None:
            self.pbar.close()
