"""Unit tests for the Variational Auto‑Encoder (VAE) pipeline.

The suite focuses on fast, deterministic checks that **only** rely on CPU
execution and tiny in‑memory DataFrames, so it is suitable for CI pipelines.
"""

from __future__ import annotations

import unittest
from datetime import datetime
from typing import Final

import numpy as np
import pandas as pd
import torch

from vae.pipeline import DEVICE, VAE
from vae.dataclasses.training import VaeConfig
from models.enums import DataType
from models.field_metadata import FieldMetadata


def _make_numeric_vae() -> tuple[VAE, pd.DataFrame]:
    """Return a minimal VAE instance with **only** numeric columns."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.5, 3.5, 4.5]})
    meta = {
        "x": FieldMetadata(data_type=DataType.DECIMAL, decimal_places=1),
        "y": FieldMetadata(data_type=DataType.DECIMAL, decimal_places=1),
    }
    return VAE(df, meta, VaeConfig()), df


def _make_integer_vae() -> tuple[VAE, pd.DataFrame]:
    """Return a VAE with a single integer column – used to test rounding."""
    df = pd.DataFrame({"a": [1, 2, 3, 4]})
    meta = {"a": FieldMetadata(data_type=DataType.INTEGER)}
    return VAE(df, meta, VaeConfig()), df


class VAETests(unittest.TestCase):
    """Unit tests for key VAE helpers and public API points."""

    RNG_SEED: Final = 2025

    def test_encode_returns_expected_shape(self) -> None:
        """_encode must output shape (n_rows, num_dim + sum of cat_dims)."""
        vae, df = _make_numeric_vae()
        encoded = vae._encode(df)
        expected_dim = vae.training_data.num_dim + sum(vae.training_data.cat_dims)

        self.assertIsInstance(encoded, torch.Tensor)
        self.assertEqual(encoded.shape, (len(df), expected_dim))

    def test_beta_schedule_monotonic_and_clipped(self) -> None:
        """Beta should ramp from *0* → *kl_max* and then stay constant."""
        vae, _ = _make_numeric_vae()
        kl_max = vae.config.model.kl_max
        warmup_ep = vae.config.model.beta_warmup

        beta0 = vae._calculate_beta(0)
        beta_warmup = vae._calculate_beta(warmup_ep)
        beta_late = vae._calculate_beta(warmup_ep + 50)

        self.assertAlmostEqual(beta0, 0.0, places=6)
        self.assertAlmostEqual(beta_warmup, kl_max, places=6)
        self.assertAlmostEqual(beta_late, kl_max, places=6)
        # Ensure monotonic increase within warm‑up
        betas = [vae._calculate_beta(ep) for ep in range(1, warmup_ep + 1)]
        self.assertTrue(all(b1 <= b2 for b1, b2 in zip(betas[:-1], betas[1:])))

    def test_correlation_loss_single_dim_zero(self) -> None:
        """Loss should be 0 if matrix has only 1 feature column."""
        vae, _ = _make_numeric_vae()
        one_col = torch.randn(8, 1, device=DEVICE)
        loss = vae._calculate_correlation_loss(one_col)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_correlation_loss_non_negative(self) -> None:
        """Loss for multi‑dim input must be >= 0."""
        vae, df = _make_numeric_vae()
        two_cols = vae._encode(df)  # shape (n, 2)
        loss = vae._calculate_correlation_loss(two_cols)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_categorical_loss_zero_for_perfect_logits(self) -> None:
        """A perfect prediction should give near‑zero categorical loss."""
        vae, _ = _make_numeric_vae()
        # Inject a dummy categorical feature with two classes
        vae.schema.cat_cols = ["c"]
        vae.training_data.cat_dims = [2]
        vae.training_data.num_dim = 2

        # Construct a batch with true one‑hot targets after the numerics
        x = torch.tensor([[1.0, 2.0, 1.0, 0.0], [3.0, 4.0, 0.0, 1.0]], device=DEVICE)
        logits_good = [torch.tensor([[10.0, -10.0], [-10.0, 10.0]], device=DEVICE)]
        loss_good = vae._calculate_categorical_loss(x, logits_good)
        self.assertAlmostEqual(loss_good.item(), 0.0, places=4)

        logits_bad = [torch.tensor([[-10.0, 10.0], [10.0, -10.0]], device=DEVICE)]
        loss_bad = vae._calculate_categorical_loss(x, logits_bad)
        self.assertGreater(loss_bad.item(), 0.0)

    def test_real_corr_matches_numpy(self) -> None:
        """The pre‑computed real correlation matrix must match NumPy’s value."""
        vae, df = _make_numeric_vae()
        corr = vae.training_data.real_corr.cpu().numpy()
        expected = np.corrcoef(df.values.T)
        np.testing.assert_array_almost_equal(corr, expected, decimal=5)

    def test_generate_dict_output(self) -> None:
        """With _cpu=False the method should return a dict with keys df and cont"""
        vae, _ = _make_numeric_vae()
        out = vae.generate(7, temperature=0.9, _cpu=False)
        self.assertIn("df", out)
        self.assertIn("cont", out)
        self.assertIsInstance(out["df"], pd.DataFrame)
        self.assertIsInstance(out["cont"], torch.Tensor)
        self.assertEqual(len(out["df"]), 7)
        self.assertEqual(out["cont"].shape[0], 7)

    def test_generate_dataframe_columns(self) -> None:
        """Default call (_cpu=True) must return a DataFrame with original columns."""
        vae, df = _make_numeric_vae()
        gen_df = vae.generate(5)
        self.assertIsInstance(gen_df, pd.DataFrame)
        self.assertSetEqual(set(gen_df.columns), set(df.columns))
        self.assertEqual(len(gen_df), 5)

    def test_generate_respects_integer_metadata(self) -> None:
        """INTEGER fields should be emitted with integer dtype and valid range."""
        vae, df = _make_integer_vae()
        gen_df = vae.generate(6)
        self.assertTrue(pd.api.types.is_integer_dtype(gen_df["a"]))
        self.assertTrue(gen_df["a"].between(df["a"].min(), df["a"].max()).all())

    def test_generate_datetime_format(self) -> None:
        """Datetime columns must conform to the specified datetime_format"""
        dates = pd.date_range("2020-01-01", periods=4, freq="D")
        df = pd.DataFrame({"dt": dates.strftime("%Y-%m-%d")})
        meta = {
            "dt": FieldMetadata(data_type=DataType.DATETIME, datetime_format="%Y-%m-%d")
        }
        vae = VAE(df, meta, VaeConfig())
        gen_df = vae.generate(3)
        for value in gen_df["dt"]:
            # will raise ValueError if format mismatch
            datetime.strptime(value, "%Y-%m-%d")


if __name__ == "__main__":
    unittest.main()
