from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

class _BaseTf:
    def fit(self, s: pd.Series): ...
    def transform(self, s: pd.Series) -> np.ndarray: ...
    def inverse(self, v: np.ndarray) -> pd.Series: ...


class _StdTf(_BaseTf):
    def fit(self, s):
        self.mean, self.std = s.mean(), s.std()
        return self
    def transform(self, s):
        return (s - self.mean) / self.std
    def inverse(self, v):
        return pd.Series(v * self.std + self.mean)


class _MinMaxTf(_BaseTf):
    """Scale to [–1,1] by true min/max—preserves heavy tails."""
    def fit(self, s: pd.Series):
        self.min_, self.max_ = float(s.min()), float(s.max())
        return self
    def transform(self, s: pd.Series) -> np.ndarray:
        r = (s.astype(float) - self.min_) / (self.max_ - self.min_)
        return 2 * r - 1
    def inverse(self, v: np.ndarray) -> pd.Series:
        r = (v + 1) / 2
        raw = r * (self.max_ - self.min_) + self.min_
        return pd.Series(raw)


class _LogTf(_BaseTf):
    """Handles heavy-tailed distributions using log transform with offset handling"""

    def fit(self, s: pd.Series):
        # Determine the minimum value to handle zero/negative values
        self.min_val = float(s.min())
        self.offset = 1.0 if self.min_val >= 0.0 else 1.0 - self.min_val

        # Apply log transform
        log_vals = np.log1p(s + self.offset - self.min_val)
        self.log_min = float(log_vals.min())
        self.log_max = float(log_vals.max())
        return self

    def transform(self, s: pd.Series) -> np.ndarray:
        log_vals = np.log1p(s + self.offset - self.min_val)
        # Scale to [-1, 1] range for GAN
        return 2 * ((log_vals - self.log_min) / (self.log_max - self.log_min)) - 1

    def inverse(self, v: np.ndarray) -> pd.Series:
        # Convert back from [-1, 1] to log space
        log_vals = (v + 1) / 2 * (self.log_max - self.log_min) + self.log_min
        # Convert from log space to original scale
        raw = np.expm1(log_vals) + self.min_val - self.offset
        return pd.Series(raw)


class _ZeroInflatedTf(_BaseTf):
    """Specialized for data with many zeros followed by positive values"""

    def fit(self, s: pd.Series):
        # Identify zero vs non-zero
        self.zero_mask = (s == 0)
        self.zero_rate = self.zero_mask.mean()

        # Handle non-zero part with a standard approach
        non_zero = s[~self.zero_mask]
        if len(non_zero) > 0:
            # Use log transform for the positive values
            log_vals = np.log1p(non_zero)
            self.log_min = float(log_vals.min())
            self.log_max = float(log_vals.max())
        else:
            self.log_min, self.log_max = 0.0, 1.0
        return self

    def transform(self, s: pd.Series) -> np.ndarray:
        # Initialize with a special value for zeros
        result = np.zeros(len(s))

        # Transform the non-zero values
        non_zero_mask = (s != 0)
        if non_zero_mask.any():
            non_zero = s[non_zero_mask]
            log_vals = np.log1p(non_zero)
            # Scale to [0, 1] range
            scaled = (log_vals - self.log_min) / (self.log_max - self.log_min)
            result[non_zero_mask] = scaled

        # Use -1 for zeros, positive values for non-zeros
        result = result * 2 - 1  # Scale to [-1, 1]
        result[s == 0] = -1  # Set zeros to -1

        return result

    def inverse(self, v: np.ndarray) -> pd.Series:
        # Initialize result with zeros
        result = np.zeros_like(v, dtype=float)

        # Apply threshold to determine which values should be non-zero
        # Higher values in the output are more likely to be non-zero
        non_zero_probs = (v + 1) / 2  # Convert from [-1, 1] to [0, 1]

        # Use a probability threshold based on real data's zero rate
        non_zero_mask = non_zero_probs > self.zero_rate

        # Convert non-zero values back to original scale
        if non_zero_mask.any():
            # Rescale [0, 1] to log space
            log_vals = non_zero_probs[non_zero_mask] * (self.log_max - self.log_min) + self.log_min
            # Convert from log space
            result[non_zero_mask] = np.expm1(log_vals)

        return pd.Series(result)


class _BoundedTf(_BaseTf):
    """For variables with natural bounds like ages"""

    def fit(self, s: pd.Series):
        # Auto-detect boundaries
        # For integer columns, round to nearest 5 or 10
        if s.dtype in (int, np.int64, np.int32):
            min_val = max(0, int(np.floor(s.min() / 5) * 5))
            max_val = int(np.ceil(s.max() / 5) * 5)
        else:
            # For float columns, just use actual min/max
            min_val = float(s.min())
            max_val = float(s.max())

        self.min_val = min_val
        self.max_val = max_val

        # Detect modality (unimodal, bimodal, etc)
        self.mean = float(s.mean())
        self.std = float(s.std())

        # Save quantiles for better distribution matching
        self.quantiles = np.quantile(s, np.linspace(0.05, 0.95, 19))
        return self

    def transform(self, s: pd.Series) -> np.ndarray:
        # Clip values to bounds
        s_clipped = s.clip(self.min_val, self.max_val)

        # Scale to [-1, 1]
        normalized = (s_clipped - self.min_val) / (self.max_val - self.min_val)
        return 2 * normalized - 1

    def inverse(self, v: np.ndarray) -> pd.Series:
        # Convert back from [-1, 1] to [0, 1]
        norm = (v + 1) / 2

        # Apply distribution shape correction (optional)
        # Map uniform quantiles to empirical distribution
        if hasattr(self, 'quantiles'):
            quantile_indices = np.floor(norm * 20).clip(0, 19).astype(int)
            lower_q = np.zeros_like(norm)
            upper_q = np.ones_like(norm) * self.max_val

            # Apply quantile mapping for values that fall within our saved quantiles
            mask = (quantile_indices > 0) & (quantile_indices < 19)
            if mask.any():
                lower_q[mask] = self.quantiles[quantile_indices[mask] - 1]
                upper_q[mask] = self.quantiles[quantile_indices[mask]]

                # Interpolate between quantiles
                alpha = (norm[mask] * 20) % 1
                norm[mask] = lower_q[mask] + alpha * (upper_q[mask] - lower_q[mask])

        # Scale back to original range
        raw = norm * (self.max_val - self.min_val) + self.min_val
        return pd.Series(raw)

class _ContTf(_BaseTf):
    """Rank‑gauss with light tail‑clipping for stability."""

    def fit(self, s):
        q_low, q_hi = s.quantile([0.0025, 0.9975])
        self.sorted_ = np.sort(s.clip(q_low, q_hi).to_numpy(copy=True))
        return self

    def transform(self, s):
        u = rankdata(s, method="average") / (len(s) + 1)
        return norm.ppf(u)

    def inverse(self, v):
        u = norm.cdf(v).clip(0, 1)
        idx = np.floor(u * (len(self.sorted_) - 1)).astype(int)
        return pd.Series(self.sorted_[idx])


class _DtTf(_ContTf):
    def __init__(self, fmt): self.fmt = fmt
    def fit(self, s):  return super().fit(self._to_sec(s))
    def transform(self, s): return super().transform(self._to_sec(s))
    def inverse(self, v):
        return pd.to_datetime(super().inverse(v), unit="s").dt.strftime(self.fmt)
    def _to_sec(self, s):
        return pd.to_datetime(s, format=self.fmt, errors="coerce").astype("int64") // 10**9