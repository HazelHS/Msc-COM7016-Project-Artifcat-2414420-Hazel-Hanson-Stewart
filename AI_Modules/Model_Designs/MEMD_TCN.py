"""
MEMD-TCN: Stock Index Forecasting Model
========================================
Implementation based on:
  - Yao et al. (2023) "Stock index forecasting based on multivariate empirical 
    mode decomposition and temporal convolutional networks"
  - Rehman & Mandic (2010) "Multivariate empirical mode decomposition"
  - Bai et al. (2018) "An empirical evaluation of generic convolutional and 
    recurrent networks for sequence modeling"

Pipeline:
  1. MEMD  -> decompose 5-channel OHLCV into aligned IMFs + residual
  2. TCN   -> separate TCN per IMF-group predicts closing price component
  3. Sum   -> reconstruct final closing price forecast
"""

import sys
from pathlib import Path

# Add the project root directory to sys.path
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_path))

from dependency_checker import *
from torch.nn.utils import weight_norm
from scipy.interpolate import CubicSpline


# =============================================================================
# SECTION 1: MEMD — Rehman & Mandic (2010)
# =============================================================================

def _get_primes(n: int) -> list[int]:
    """Return first n prime numbers.

    Args:
        n: Count of primes to generate.

    Returns:
        List of the first n prime integers.
    """
    primes, candidates = [], list(range(2, n * 20))
    for x in candidates:
        if all(x % p != 0 for p in primes):
            primes.append(x)
            if len(primes) == n:
                return primes
    return primes


def _halton(i: int, base: int) -> float:
    """Generate i-th Halton sequence value in given base.

    Rehman & Mandic (2010), eq. 3.5

    Args:
        i: Sequence index (0-based).
        base: Numerical base for the quasi-random sequence.

    Returns:
        Quasi-random float in [0, 1).
    """
    f, r = 1.0, 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def _hammersley_directions(K: int, n_dims: int) -> np.ndarray:
    """
    Generate K direction vectors on (n_dims-1)-sphere using Hammersley
    low-discrepancy sequence.
    Rehman & Mandic (2010), Section 3b, eq. 3.2-3.6

    Args:
        K:      Number of direction vectors (paper uses K=512)
        n_dims: Dimensionality of signal (5 for OHLCV)

    Returns:
        directions: ndarray [K, n_dims] unit vectors on (n-1)-sphere
    """
    prime_list = _get_primes(n_dims - 1)

    # Build raw Hammersley points in [0,1]^n_dims
    raw = np.zeros((K, n_dims))
    for i in range(K):
        raw[i, 0] = i / K
        for j, p in enumerate(prime_list):
            raw[i, j + 1] = _halton(i, p)

    # Map to hyperspherical coordinates (eq. 3.2)
    # theta_j in [0, pi] for j=1..n-2, theta_{n-1} in [0, 2pi]
    angles = raw * np.pi
    angles[:, -1] *= 2.0  # last angle -> [0, 2pi]

    # Convert hyperspherical -> Cartesian (eq. 3.3)
    directions = np.zeros((K, n_dims))
    for k in range(K):
        th = angles[k]
        directions[k, 0] = np.cos(th[0])
        for j in range(1, n_dims - 1):
            directions[k, j] = np.prod(np.sin(th[:j])) * np.cos(th[j])
        directions[k, -1] = np.prod(np.sin(th[:-1])) * np.sin(th[-1])

    return directions  # [K, n_dims]


def _find_maxima(signal: np.ndarray) -> np.ndarray:
    """Return indices of local maxima in 1-D signal.

    Args:
        signal: 1-D array of numeric values.

    Returns:
        Integer array of indices where ``signal[i] > signal[i-1]`` and
        ``signal[i] > signal[i+1]``.
    """
    maxima = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            maxima.append(i)
    return np.array(maxima, dtype=int)


def _is_monotonic(X: np.ndarray) -> bool:
    """Check if all channels of multivariate signal are monotonic.

    Used as the MEMD termination condition (Rehman & Mandic 2010).

    Args:
        X: ndarray [n_channels, T] — multivariate signal.

    Returns:
        True if every channel is entirely non-decreasing or non-increasing.
    """
    for c in range(X.shape[0]):
        diff = np.diff(X[c])
        if not (np.all(diff >= 0) or np.all(diff <= 0)):
            return False
    return True


def _compute_multivariate_mean(
    X: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    """
    Compute local mean of multivariate signal via directional projections.
    Rehman & Mandic (2010), Algorithm 2, Steps 2-5

    Args:
        X:          ndarray [n_channels, T]
        directions: ndarray [K, n_channels]

    Returns:
        mean: ndarray [n_channels, T]
    """
    n_channels, T = X.shape
    t = np.arange(T)
    envelope_sum = np.zeros((n_channels, T))
    valid_count = 0

    for d in directions:
        # Step 2: project multivariate signal onto direction vector
        proj = X.T @ d  # [T]

        # Step 3: find maxima of projection
        maxima_idx = _find_maxima(proj)
        if len(maxima_idx) < 2:
            continue

        # Step 4: cubic spline interpolation of multivariate signal at maxima
        # Pad endpoints to cover full signal range
        idx = np.unique(np.concatenate([[0], maxima_idx, [T - 1]]))

        for c in range(n_channels):
            try:
                cs = CubicSpline(idx, X[c, idx], extrapolate=True)
                envelope_sum[c] += cs(t)
            except Exception:
                continue

        valid_count += 1

    # Step 5: mean envelope (eq. 3.7)
    if valid_count == 0:
        return np.zeros_like(X)
    return envelope_sum / valid_count


def memd(
    X: np.ndarray,
    K: int = 512,
    max_imfs: int = 12,
    max_sift: int = 50,
    sd_threshold: float = 0.2,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Multivariate Empirical Mode Decomposition.
    Rehman & Mandic (2010), Algorithm 2

    Args:
        X:            ndarray [n_channels, T] — multivariate signal
        K:            Number of direction vectors (paper: 512)
        max_imfs:     Maximum number of IMFs to extract
        max_sift:     Maximum sifting iterations per IMF
        sd_threshold: Sifting stopping criterion (Huang et al. 2003)

    Returns:
        imfs:     List of ndarray [n_channels, T], length = n_imfs
        residual: ndarray [n_channels, T]
    """
    directions = _hammersley_directions(K, X.shape[0])
    imfs = []
    residual = X.copy().astype(float)

    for imf_idx in range(max_imfs):
        h = residual.copy()

        # Sifting loop
        for _ in range(max_sift):
            m = _compute_multivariate_mean(h, directions)
            h_new = h - m

            # Stopping criterion — Huang et al. (2003) SD test
            sd = np.sum((h - h_new) ** 2) / (np.sum(h ** 2) + 1e-10)
            h = h_new

            if sd < sd_threshold:
                break

        imfs.append(h)
        residual = residual - h

        # Terminate if residual is monotonic in all channels
        if _is_monotonic(residual):
            break

    return imfs, residual


# =============================================================================
# SECTION 2: TCN — Bai et al. (2018) + Yao et al. (2023)
# =============================================================================

class _CausalConv1d(nn.Module):
    """
    Causal dilated 1-D convolution.
    Ensures no future information leaks into prediction.
    Bai et al. (2018), Section 3
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ):
        """Initialise a causal Conv1d with left-padding and weight normalisation.

        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output feature channels.
            kernel_size: Convolution kernel width.
            dilation: Dilation factor for the convolution.
        """
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels, out_channels, kernel_size,
                dilation=dilation, padding=self.padding,
            )
        )
        # Gaussian weight initialisation N(0, 0.01) — Bai et al. (2018)
        nn.init.normal_(self.conv.weight_v, 0, 0.01)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal convolution to *x*.

        Args:
            x: Input tensor of shape [batch, in_channels, time].

        Returns:
            Tensor of shape [batch, out_channels, time] with no future leakage.
        """
        out = self.conv(x)
        # Slice off future padding to maintain causality
        return out[:, :, :-self.padding] if self.padding > 0 else out


class ResidualBlock(nn.Module):
    """
    TCN Residual Block.
    2x (CausalConv -> ReLU -> Dropout) + residual connection.
    Bai et al. (2018), Figure 1

    Args:
        in_channels:  Input feature channels
        out_channels: Output feature channels
        kernel_size:  Conv kernel size (paper: k=2)
        dilation:     Dilation factor
        dropout:      Dropout probability
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = _CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = _CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        # Spatial dropout — zeros entire channels (Bai et al. 2018)
        self.dropout = nn.Dropout(dropout)

        # 1x1 projection shortcut if channel dimensions differ
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two causal conv layers with dropout and a residual connection.

        Args:
            x: Input tensor of shape [batch, in_channels, seq_len].

        Returns:
            Tensor of shape [batch, out_channels, seq_len] after residual add.
        """
        out = self.dropout(self.relu(self.conv1(x)))
        out = self.dropout(self.relu(self.conv2(out)))
        res = self.shortcut(x) if self.shortcut is not None else x
        return self.relu(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network for one IMF-group.
    Bai et al. (2018) architecture + Yao et al. (2023) hyperparameters.

    Input:  [batch, in_channels, seq_len]  — multivariate IMF group (5 OHLCV channels)
    Output: [batch, 1]                     — next closing price IMF value

    Args:
        in_channels: Number of input channels (5 for OHLCV)
        hidden:      Number of conv filters (32 for IMF1-4, 64 for IMF5+)
        kernel_size: Conv kernel size (paper: k=2)
        dilations:   Dilation sequence (paper: [1, 2, 4])
        dropout:     Dropout probability
    """
    def __init__(
        self,
        in_channels: int = 5,
        hidden: int = 32,
        kernel_size: int = 2,
        dilations: list[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4]  # Yao et al. (2023), Table 1

        layers = []
        for i, d in enumerate(dilations):
            c_in = in_channels if i == 0 else hidden
            layers.append(ResidualBlock(c_in, hidden, kernel_size, d, dropout))

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the TCN stack and return the next-step scalar prediction.

        Args:
            x: Input tensor of shape [batch, in_channels, seq_len].

        Returns:
            1-D tensor of shape [batch] with one predicted value per sample.
        """
        # x: [batch, channels, seq_len]
        out = self.network(x)            # [batch, hidden, seq_len]
        out = out[:, :, -1]              # last timestep only
        return self.fc(out).squeeze(-1)  # [batch]


# =============================================================================
# SECTION 3: MEMD-TCN Full Model — Yao et al. (2023)
# =============================================================================

class MEMD_TCN_Model(nn.Module):
    """
    Full MEMD-TCN Model.
    Yao et al. (2023) — Three-stage architecture:
      Stage 1 — MEMD:          Decompose OHLCV into frequency-aligned IMFs + residual
      Stage 2 — TCN (per IMF): Separate TCN predicts each closing price IMF component
      Stage 3 — Reconstruct:   Sum all component predictions for final forecast

    Hidden units follow Yao et al. (2023), Table 2:
      IMF 1-4  -> 32 units
      IMF 5+   -> 64 units

    Args:
        in_channels:  Number of OHLCV input channels (default: 5)
        kernel_size:  TCN conv kernel size (paper: 2)
        dilations:    TCN dilation sequence (paper: [1, 2, 4])
        dropout:      Dropout rate (default: 0.2)
        K:            MEMD direction vectors — Rehman & Mandic (2010): 512
        max_imfs:     Maximum IMFs for MEMD to extract (default: 12)

    Example:
        >>> model = MEMD_TCN_Model()
        >>> imfs, residual = model.decompose(ohlcv_tensor)   # [5, T]
        >>> pred = model.forward(imf_group, imf_idx=0)       # [batch, 5, lag]
    """

    def __init__(
        self,
        in_channels: int = 5,
        kernel_size: int = 2,
        dilations: list[int] = None,
        dropout: float = 0.2,
        K: int = 512,
        max_imfs: int = 12,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dilations = dilations or [1, 2, 4]
        self.dropout = dropout
        self.K = K
        self.max_imfs = max_imfs

        # Build a TCN for each possible IMF slot + 1 residual
        # Hidden units: IMF0-3 -> 32, IMF4+ -> 64  (Yao et al. Table 2)
        self.tcn_blocks = nn.ModuleList([
            TCN(
                in_channels=in_channels,
                hidden=32 if i < 4 else 64,
                kernel_size=kernel_size,
                dilations=self.dilations,
                dropout=dropout,
            )
            for i in range(max_imfs + 1)  # +1 for residual slot
        ])

    def decompose(
        self,
        X: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """
        Run MEMD on raw OHLCV data.
        Rehman & Mandic (2010), Algorithm 2

        Args:
            X: ndarray [n_channels, T] — raw OHLCV (channels-first)

        Returns:
            imfs:     list of ndarray [n_channels, T]
            residual: ndarray [n_channels, T]
        """
        return memd(X, K=self.K, max_imfs=self.max_imfs)

    def forward(
        self,
        x: torch.Tensor,
        imf_idx: int,
    ) -> torch.Tensor:
        """
        Forward pass for a single IMF group through its dedicated TCN.

        Args:
            x:       Tensor [batch, in_channels, seq_len] — one normalised IMF group
            imf_idx: Which IMF slot to route through (0-indexed)
                     Pass max_imfs for the residual slot

        Returns:
            Tensor [batch] — predicted closing price component for next timestep
        """
        idx = min(imf_idx, len(self.tcn_blocks) - 1)
        return self.tcn_blocks[idx](x)

    def reconstruct(self, component_predictions: list[torch.Tensor]) -> torch.Tensor:
        """
        Stage 3: Reconstruct final closing price by summing all IMF predictions.
        Yao et al. (2023), Figure 2 "Reconstruction by sum"

        Args:
            component_predictions: list of Tensor [batch] — one per IMF + residual

        Returns:
            Tensor [batch] — reconstructed closing price forecast
        """
        return torch.stack(component_predictions, dim=0).sum(dim=0)