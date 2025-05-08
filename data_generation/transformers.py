from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata


class _BaseTf:
    def fit(self, s: pd.Series): ...
    def transform(self, s: pd.Series) -> np.ndarray: ...
    def inverse(self, v: np.ndarray) -> pd.Series: ...

    # scikit-learn style alias:
    inverse_transform = inverse


class _ContTf(_BaseTf):
    """Rank‑gauss with light tail‑clipping for stability."""

    def __init__(self):
        self.sorted_ = None

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
    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt

    def fit(self, s):
        return super().fit(self._to_sec(s))

    def transform(self, s):
        return super().transform(self._to_sec(s))

    def inverse(self, v):
        return pd.to_datetime(super().inverse(v), unit="s").dt.strftime(self.fmt)

    def _to_sec(self, s):
        return (
            pd.to_datetime(s, format=self.fmt, errors="coerce").astype("int64") // 10**9
        )
