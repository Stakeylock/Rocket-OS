"""
Transformer-Based Anomaly Detection (T-BAD) for propulsion telemetry.

Implements a simplified self-attention mechanism in pure numpy to learn
temporal correlations from nominal engine telemetry (turbopump RPM,
chamber pressure, vibration, etc.) and detect precursor signatures of
engine failure.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Time-series buffer
# ---------------------------------------------------------------------------

class TimeSeriesBuffer:
    """Fixed-length rolling window for multivariate sensor readings.

    Parameters
    ----------
    window_size : int
        Number of time-steps to store.
    num_channels : int
        Number of sensor channels.
    """

    def __init__(self, window_size: int = 128, num_channels: int = 6):
        self.window_size = window_size
        self.num_channels = num_channels
        self._buffer: np.ndarray = np.zeros((window_size, num_channels))
        self._count: int = 0  # Total samples pushed (may exceed window_size)

    def push(self, sample: np.ndarray) -> None:
        """Append a single (num_channels,) sample, evicting the oldest."""
        sample = np.asarray(sample, dtype=np.float64).ravel()
        assert sample.shape[0] == self.num_channels, (
            f"Expected {self.num_channels} channels, got {sample.shape[0]}"
        )
        self._buffer = np.roll(self._buffer, -1, axis=0)
        self._buffer[-1] = sample
        self._count += 1

    def push_batch(self, samples: np.ndarray) -> None:
        """Append multiple (T, num_channels) samples."""
        samples = np.asarray(samples, dtype=np.float64)
        for s in samples:
            self.push(s)

    @property
    def is_full(self) -> bool:
        return self._count >= self.window_size

    @property
    def data(self) -> np.ndarray:
        """Return the buffer contents as (window_size, num_channels)."""
        return self._buffer.copy()

    @property
    def valid_data(self) -> np.ndarray:
        """Return only the filled portion of the buffer."""
        n = min(self._count, self.window_size)
        return self._buffer[-n:].copy()

    def reset(self) -> None:
        self._buffer[:] = 0.0
        self._count = 0


# ---------------------------------------------------------------------------
# Lightweight linear algebra helpers (pure numpy)
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def _layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer normalisation over the last axis."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def _gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit activation."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


# ---------------------------------------------------------------------------
# Self-attention block
# ---------------------------------------------------------------------------

class SelfAttentionBlock:
    """Single-head self-attention + feed-forward in pure numpy.

    Parameters
    ----------
    d_model : int   -- dimensionality of each token embedding.
    d_ff    : int   -- feed-forward hidden dimension.
    """

    def __init__(self, d_model: int, d_ff: int = 64, rng: np.random.RandomState = None):
        if rng is None:
            rng = np.random.RandomState(42)

        scale = np.sqrt(2.0 / d_model)
        # Query, Key, Value projections
        self.W_q = rng.randn(d_model, d_model).astype(np.float64) * scale
        self.W_k = rng.randn(d_model, d_model).astype(np.float64) * scale
        self.W_v = rng.randn(d_model, d_model).astype(np.float64) * scale
        self.W_o = rng.randn(d_model, d_model).astype(np.float64) * scale

        # Feed-forward
        self.W1 = rng.randn(d_model, d_ff).astype(np.float64) * np.sqrt(2.0 / d_ff)
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.randn(d_ff, d_model).astype(np.float64) * np.sqrt(2.0 / d_model)
        self.b2 = np.zeros(d_model)

        self.d_model = d_model

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.  x: (T, d_model) -> (T, d_model)."""
        # Multi-head self-attention (single head here)
        Q = x @ self.W_q   # (T, d)
        K = x @ self.W_k
        V = x @ self.W_v

        scale = np.sqrt(self.d_model)
        attn_scores = (Q @ K.T) / scale          # (T, T)
        attn_weights = _softmax(attn_scores, axis=-1)
        attn_out = attn_weights @ V               # (T, d)
        attn_out = attn_out @ self.W_o

        # Residual + LayerNorm
        x = _layer_norm(x + attn_out)

        # Feed-forward
        ff = _gelu(x @ self.W1 + self.b1)
        ff = ff @ self.W2 + self.b2

        # Residual + LayerNorm
        x = _layer_norm(x + ff)
        return x


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

def _sinusoidal_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Standard sinusoidal positional encoding (T, d_model)."""
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, None]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term[:d_model // 2]) if d_model % 2 == 0 \
        else np.cos(position * div_term[:pe[:, 1::2].shape[1]])
    return pe


# ---------------------------------------------------------------------------
# Transformer-Based Anomaly Detector (T-BAD)
# ---------------------------------------------------------------------------

class TransformerAnomalyDetector:
    """Lightweight transformer for propulsion anomaly detection.

    Architecture
    ------------
    1. Input embedding: linear projection from num_channels to d_model.
    2. Positional encoding (sinusoidal).
    3. N stacked self-attention blocks.
    4. Output projection: predicts next-step sensor values.
    5. Anomaly score = reconstruction / prediction error.

    Training
    --------
    ``train_nominal(data)`` learns to predict the next time-step from
    nominal telemetry so that deviations from the learned temporal
    structure yield high anomaly scores.

    Channels (default)
    ------------------
    0: turbopump_rpm  (normalised)
    1: chamber_pressure
    2: vibration_x
    3: vibration_y
    4: vibration_z
    5: temperature
    """

    # Failure type catalogue
    FAILURE_TYPES = [
        "nominal",
        "turbopump_bearing_wear",
        "combustion_instability",
        "injector_blockage",
        "seal_leak",
        "thermal_runaway",
    ]

    def __init__(
        self,
        num_channels: int = 6,
        window_size: int = 64,
        d_model: int = 32,
        n_layers: int = 2,
        d_ff: int = 64,
        learning_rate: float = 1e-3,
        seed: int = 42,
    ):
        self.num_channels = num_channels
        self.window_size = window_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.lr = learning_rate
        self.rng = np.random.RandomState(seed)

        # Input projection: (num_channels) -> (d_model)
        scale_in = np.sqrt(2.0 / num_channels)
        self.W_in = self.rng.randn(num_channels, d_model).astype(np.float64) * scale_in
        self.b_in = np.zeros(d_model)

        # Transformer blocks
        self.blocks: List[SelfAttentionBlock] = [
            SelfAttentionBlock(d_model, d_ff, self.rng) for _ in range(n_layers)
        ]

        # Output projection: (d_model) -> (num_channels)  (predict next step)
        scale_out = np.sqrt(2.0 / d_model)
        self.W_out = self.rng.randn(d_model, num_channels).astype(np.float64) * scale_out
        self.b_out = np.zeros(num_channels)

        # Positional encoding cache
        self._pe = _sinusoidal_positional_encoding(window_size, d_model)

        # Normalisation statistics (learned in train_nominal)
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

        # Anomaly threshold (set after training)
        self._threshold: float = 0.5
        self._nominal_error_stats: Optional[Tuple[float, float]] = None

        # Failure-signature centroids (learned during training or set manually)
        self._failure_signatures: Dict[str, np.ndarray] = {}

        # Training flag
        self._is_trained: bool = False

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Run the transformer stack.

        Parameters
        ----------
        x : (T, num_channels) raw (normalised) sensor data.

        Returns
        -------
        predictions : (T, num_channels) predicted next-step values.
        """
        T = x.shape[0]
        # Input embedding
        h = x @ self.W_in + self.b_in           # (T, d_model)
        h = h + self._pe[:T]                     # Add positional encoding

        # Transformer blocks
        for block in self.blocks:
            h = block.forward(h)

        # Output head (predict next step for each position)
        pred = h @ self.W_out + self.b_out       # (T, num_channels)
        return pred

    def _normalise(self, data: np.ndarray) -> np.ndarray:
        """Z-score normalisation using stored statistics."""
        if self._mean is None or self._std is None:
            return data
        return (data - self._mean) / (self._std + 1e-8)

    def _denormalise(self, data: np.ndarray) -> np.ndarray:
        if self._mean is None or self._std is None:
            return data
        return data * (self._std + 1e-8) + self._mean

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_nominal(
        self,
        data: np.ndarray,
        epochs: int = 50,
        batch_windows: int = 16,
    ) -> Dict[str, List[float]]:
        """Train the anomaly detector on nominal telemetry.

        Parameters
        ----------
        data : (N, num_channels) array of nominal sensor readings
               collected over N time-steps.
        epochs : int
            Number of training epochs.
        batch_windows : int
            Number of random windows sampled per epoch.

        Returns
        -------
        dict with 'loss' key containing per-epoch loss values.
        """
        data = np.asarray(data, dtype=np.float64)
        assert data.ndim == 2 and data.shape[1] == self.num_channels

        # Compute normalisation statistics
        self._mean = np.mean(data, axis=0)
        self._std = np.std(data, axis=0)
        data_norm = self._normalise(data)

        N = data_norm.shape[0]
        losses: List[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0

            for _ in range(batch_windows):
                # Sample a random window
                start = self.rng.randint(0, max(1, N - self.window_size - 1))
                end = min(start + self.window_size, N - 1)
                x_window = data_norm[start:end]     # (W, C)
                y_target = data_norm[start + 1:end + 1]  # shifted by 1

                T = x_window.shape[0]
                if T < 4:
                    continue

                # Forward
                pred = self._forward(x_window)  # (T, C)

                # Trim to match target length
                min_len = min(pred.shape[0], y_target.shape[0])
                pred = pred[:min_len]
                y_target_trim = y_target[:min_len]

                # MSE loss
                error = pred - y_target_trim
                loss = np.mean(error ** 2)
                epoch_loss += loss

                # --- Simple gradient descent on output projection ---
                # (Full backprop through attention is expensive; we do
                #  partial updates on the input/output projections for
                #  a lightweight training loop.)
                # dL/dW_out = h^T @ d_error / (T*C)
                # We need the last hidden states; recompute cheaply
                h = x_window @ self.W_in + self.b_in + self._pe[:T]
                for block in self.blocks:
                    h = block.forward(h)
                h_trim = h[:min_len]

                grad_pred = 2.0 * error / (min_len * self.num_channels)
                grad_W_out = h_trim.T @ grad_pred
                grad_b_out = np.sum(grad_pred, axis=0)

                self.W_out -= self.lr * grad_W_out
                self.b_out -= self.lr * grad_b_out

                # Gradient on input projection
                # dL/dW_in (approximate, ignoring attention gradients)
                # Back-propagate through output: delta_h = grad_pred @ W_out^T
                delta_h = grad_pred @ self.W_out.T  # (min_len, d_model)
                # dW_in = x^T @ delta_h
                x_trim = x_window[:min_len]
                grad_W_in = x_trim.T @ delta_h
                grad_b_in = np.sum(delta_h, axis=0)

                self.W_in -= self.lr * 0.1 * grad_W_in  # Smaller LR for input
                self.b_in -= self.lr * 0.1 * grad_b_in

            avg_loss = epoch_loss / max(batch_windows, 1)
            losses.append(avg_loss)

        # Compute anomaly threshold from training data
        self._calibrate_threshold(data_norm)
        self._is_trained = True

        # Build basic failure signatures (synthetic perturbation centroids)
        self._build_failure_signatures()

        return {"loss": losses}

    def _calibrate_threshold(self, data_norm: np.ndarray) -> None:
        """Set the anomaly threshold from prediction errors on nominal data."""
        N = data_norm.shape[0]
        errors: List[float] = []

        # Slide a window across the training data
        stride = max(1, self.window_size // 4)
        for start in range(0, N - self.window_size - 1, stride):
            end = start + self.window_size
            x_window = data_norm[start:end]
            y_target = data_norm[start + 1:end + 1]

            pred = self._forward(x_window)
            min_len = min(pred.shape[0], y_target.shape[0])
            err = np.mean((pred[:min_len] - y_target[:min_len]) ** 2)
            errors.append(err)

        if errors:
            self._nominal_error_stats = (float(np.mean(errors)), float(np.std(errors)))
            # Set threshold at mean + 3*sigma
            self._threshold = self._nominal_error_stats[0] + 3.0 * self._nominal_error_stats[1]
        else:
            self._threshold = 0.5

    def _build_failure_signatures(self) -> None:
        """Create synthetic anomaly-direction centroids for failure-type
        classification.

        Each failure type has a characteristic error pattern across sensor
        channels.  These are hand-crafted signatures that the detector
        matches against.
        """
        C = self.num_channels
        self._failure_signatures = {
            "turbopump_bearing_wear": self._make_signature(
                [0], [1.0], C   # High error in turbopump RPM channel
            ),
            "combustion_instability": self._make_signature(
                [1, 2, 3, 4], [0.8, 0.5, 0.5, 0.5], C  # Pressure + vibrations
            ),
            "injector_blockage": self._make_signature(
                [1], [1.0], C   # Chamber pressure anomaly
            ),
            "seal_leak": self._make_signature(
                [1, 5], [0.6, 0.8], C   # Pressure drop + temperature change
            ),
            "thermal_runaway": self._make_signature(
                [5], [1.0], C   # Temperature channel dominant
            ),
        }

    @staticmethod
    def _make_signature(
        channels: List[int],
        weights: List[float],
        total_channels: int,
    ) -> np.ndarray:
        """Create a unit-norm signature vector."""
        sig = np.zeros(total_channels)
        for ch, w in zip(channels, weights):
            if ch < total_channels:
                sig[ch] = w
        norm = np.linalg.norm(sig)
        if norm > 1e-9:
            sig /= norm
        return sig

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(
        self,
        current_readings: np.ndarray,
        buffer: Optional[TimeSeriesBuffer] = None,
    ) -> Tuple[float, str]:
        """Detect anomalies in the latest sensor readings.

        Parameters
        ----------
        current_readings : (num_channels,) or (T, num_channels)
            Latest sensor readings.  If 1-D, a single time-step.
            If 2-D, a window of recent history.
        buffer : TimeSeriesBuffer, optional
            If provided, push current_readings into the buffer and use
            the full buffer window for detection.

        Returns
        -------
        anomaly_score : float in [0, 1].  0 = nominal, 1 = certain anomaly.
        failure_prediction : str  one of FAILURE_TYPES.
        """
        readings = np.asarray(current_readings, dtype=np.float64)

        # If a buffer is provided, push and use its full window
        if buffer is not None:
            if readings.ndim == 1:
                buffer.push(readings)
            else:
                buffer.push_batch(readings)
            window = buffer.valid_data
        elif readings.ndim == 1:
            # Single sample -- not enough context; return low-confidence score
            return 0.0, "nominal"
        else:
            window = readings

        # Need at least a few steps for meaningful prediction
        if window.shape[0] < 4:
            return 0.0, "nominal"

        # Normalise
        window_norm = self._normalise(window)

        # Trim to window_size
        if window_norm.shape[0] > self.window_size:
            window_norm = window_norm[-self.window_size:]

        T = window_norm.shape[0]

        # Forward pass: predict next step from each position
        pred = self._forward(window_norm)

        # Compute per-channel prediction error on the last few steps
        lookback = min(8, T - 1)
        y_actual = window_norm[-(lookback):]        # last `lookback` actual values
        y_pred = pred[-(lookback + 1):-1]            # predictions for those steps

        min_len = min(y_actual.shape[0], y_pred.shape[0])
        if min_len < 1:
            return 0.0, "nominal"

        y_actual = y_actual[:min_len]
        y_pred = y_pred[:min_len]

        per_channel_error = np.mean((y_actual - y_pred) ** 2, axis=0)  # (C,)
        total_error = float(np.mean(per_channel_error))

        # Convert to anomaly score [0, 1]
        if self._nominal_error_stats is not None:
            mu, sigma = self._nominal_error_stats
            if sigma < 1e-12:
                sigma = 1e-3
            z_score = (total_error - mu) / sigma
            # Sigmoid mapping: z=0 -> 0.5, z=3 -> ~0.95
            anomaly_score = 1.0 / (1.0 + np.exp(-z_score + 1.5))
        else:
            # No calibration -- use raw sigmoid
            anomaly_score = 1.0 / (1.0 + np.exp(-10.0 * (total_error - 0.1)))

        anomaly_score = float(np.clip(anomaly_score, 0.0, 1.0))

        # Classify failure type by matching error pattern to signatures
        failure_prediction = self._classify_failure(per_channel_error)

        return anomaly_score, failure_prediction

    def _classify_failure(self, per_channel_error: np.ndarray) -> str:
        """Match a per-channel error vector to the closest failure signature."""
        if not self._failure_signatures:
            return "nominal"

        # Normalise the error pattern
        err_norm = np.linalg.norm(per_channel_error)
        if err_norm < 1e-9:
            return "nominal"
        pattern = per_channel_error / err_norm

        best_type = "nominal"
        best_sim = -1.0

        for ftype, sig in self._failure_signatures.items():
            sim = float(np.dot(pattern, sig))
            if sim > best_sim:
                best_sim = sim
                best_type = ftype

        # Only report a failure type if similarity exceeds a minimum
        if best_sim < 0.3:
            return "nominal"

        return best_type

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def get_threshold(self) -> float:
        return self._threshold

    def set_threshold(self, threshold: float) -> None:
        self._threshold = float(threshold)

    def generate_nominal_data(
        self,
        n_steps: int = 1000,
        noise_std: float = 0.02,
    ) -> np.ndarray:
        """Generate synthetic nominal engine telemetry for testing.

        Returns (n_steps, num_channels) array with:
        ch0: turbopump RPM  (~36000, small oscillation)
        ch1: chamber pressure (~9.7 MPa)
        ch2-4: vibration XYZ (near zero, small noise)
        ch5: temperature (~3600 K)
        """
        t = np.arange(n_steps, dtype=np.float64)
        data = np.zeros((n_steps, max(self.num_channels, 6)))

        # Turbopump RPM (with slow drift and fast oscillation)
        data[:, 0] = 36000.0 + 50.0 * np.sin(2 * np.pi * t / 200) + \
                      self.rng.randn(n_steps) * noise_std * 36000.0

        # Chamber pressure
        data[:, 1] = 9.7e6 + 1e4 * np.sin(2 * np.pi * t / 150) + \
                      self.rng.randn(n_steps) * noise_std * 9.7e6

        # Vibration (XYZ)
        for ch in range(2, min(5, self.num_channels)):
            data[:, ch] = self.rng.randn(n_steps) * noise_std * 10.0

        # Temperature
        if self.num_channels > 5:
            data[:, 5] = 3600.0 + 5.0 * np.sin(2 * np.pi * t / 300) + \
                          self.rng.randn(n_steps) * noise_std * 3600.0

        return data[:, :self.num_channels]
