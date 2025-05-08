"""Unit tests for the VAE pipeline."""

import unittest
import numpy as np
import pandas as pd
import torch
from datetime import datetime

from vae.pipeline import VAE, DEVICE
from vae.dataclasses.training import VaeConfig
from models.field_metadata import FieldMetadata
from models.enums import DataType


def create_simple_vae():
    """Helper to instantiate a VAE with only numeric data."""
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [2.5, 3.5, 4.5, 5.5],
        }
    )
    meta = {
        "x": FieldMetadata(data_type=DataType.DECIMAL, decimal_places=1),
        "y": FieldMetadata(data_type=DataType.DECIMAL, decimal_places=1),
    }
    vae = VAE(df, meta, VaeConfig())
    return vae, df


def create_int_vae():
    """Helper to instantiate a VAE with an integer column."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    meta = {"a": FieldMetadata(data_type=DataType.INTEGER)}
    vae = VAE(df, meta, VaeConfig())
    return vae, df


class VAETests(unittest.TestCase):
    """Unit tests for VAE pipeline components."""

    def test_encode_shape(self):
        """_encode should return a tensor of shape (n_samples, num_dim)."""
        vae, df = create_simple_vae()
        enc = vae._encode(df)
        self.assertIsInstance(enc, torch.Tensor)
        self.assertEqual(enc.shape, (len(df), vae.training_data.num_dim))

    def test_calculate_beta_bounds(self):
        """_calculate_beta should return values within [0, kl_max]."""
        vae, _ = create_simple_vae()
        kl_max = vae.config.model.kl_max
        betas = [
            vae._calculate_beta(ep)
            for ep in [0, vae.config.training.epochs // 2, vae.config.training.epochs]
        ]
        for beta in betas:
            self.assertGreaterEqual(beta, 0.0)
            self.assertLessEqual(beta, kl_max)

    def test_beta_extremes(self):
        """Beta schedule should start at kl_max and hit zero at one cycle."""
        vae, _ = create_simple_vae()
        cfg = vae.config.model
        beta_start = vae._calculate_beta(0)
        self.assertAlmostEqual(beta_start, cfg.kl_max)
        epoch_cycle = vae.config.training.epochs // cfg.n_cycles
        beta_cycle = vae._calculate_beta(epoch_cycle)
        self.assertAlmostEqual(beta_cycle, 0.0, places=4)

    def test_correlation_loss_behavior(self):
        """_calculate_correlation_loss returns zero for single-dimension"""
        vae, df = create_simple_vae()
        # Single dimension yields zero
        cont1 = torch.randn(5, 1, device=DEVICE)
        loss1 = vae._calculate_correlation_loss(cont1)
        self.assertAlmostEqual(loss1.item(), 0.0)
        # Identical two-dim data: loss should be non-negative
        cont2 = vae._encode(df)
        loss2 = vae._calculate_correlation_loss(cont2)
        self.assertIsInstance(loss2, torch.Tensor)
        self.assertGreaterEqual(loss2.item(), 0.0)

    def test_categorical_loss_perfect_and_bad(self):
        """_calculate_categorical_loss is zero for perfect logits and positive otherwise."""
        vae, _ = create_simple_vae()
        # Inject one categorical dim
        vae.schema.cat_cols = ["c"]
        vae.training_data.cat_dims = [2]
        vae.training_data.num_dim = 2
        # Create x with one-hot at positions 2 & 3
        x = torch.tensor([[1.0, 2.5, 1.0, 0.0], [3.0, 4.5, 0.0, 1.0]], device=DEVICE)
        # Perfect logits: high score for true class
        logits_perfect = [torch.tensor([[10.0, -10.0], [-10.0, 10.0]], device=DEVICE)]
        loss_perfect = vae._calculate_categorical_loss(x, logits_perfect)
        self.assertAlmostEqual(loss_perfect.item(), 0.0, places=4)
        # Bad logits: opposite assignment
        logits_bad = [torch.tensor([[-10.0, 10.0], [10.0, -10.0]], device=DEVICE)]
        loss_bad = vae._calculate_categorical_loss(x, logits_bad)
        self.assertGreater(loss_bad.item(), 0.0)

    def test_real_corr_initialization(self):
        """training_data.real_corr is correctly computed from input DataFrame."""
        vae, df = create_simple_vae()
        corr = vae.training_data.real_corr.cpu().numpy()
        # Compute expected correlation
        arr = np.stack([df["x"].values, df["y"].values], axis=1)
        expected = np.corrcoef(arr, rowvar=False)
        np.testing.assert_array_almost_equal(corr, expected, decimal=5)

    def test_generate_shapes_cpu_false(self):
        """generate with _cpu=False returns dict with 'df' and 'cont' of correct shapes."""
        vae, df = create_simple_vae()
        out = vae.generate(7, temperature=1.0, _cpu=False)
        self.assertIsInstance(out, dict)
        # Expect DataFrame and tensor keys
        self.assertIn("df", out)
        self.assertIn("cont", out)
        gen_df = out["df"]
        cont = out["cont"]
        self.assertIsInstance(gen_df, pd.DataFrame)
        self.assertEqual(set(gen_df.columns), set(df.columns))
        self.assertEqual(len(gen_df), 7)
        self.assertIsInstance(cont, torch.Tensor)
        self.assertEqual(cont.shape, (7, vae.training_data.num_dim))

    def test_generate_dataframe_cpu_true(self):
        """generate with default _cpu returns pandas DataFrame with expected columns."""
        vae, df = create_simple_vae()
        gen_df = vae.generate(5)
        self.assertIsInstance(gen_df, pd.DataFrame)
        self.assertEqual(set(gen_df.columns), set(df.columns))
        self.assertEqual(len(gen_df), 5)

    def test_generate_integer_rounding(self):
        """generate respects integer metadata by producing integer dtype for INTEGER columns."""
        vae, df = create_int_vae()
        gen_df = vae.generate(6)
        self.assertTrue(pd.api.types.is_integer_dtype(gen_df["a"]))
        # Values within original range
        self.assertTrue(gen_df["a"].between(df["a"].min(), df["a"].max()).all())

    def test_generate_datetime_format(self):
        """Generated datetime columns adhere to the specified datetime_format."""
        dates = pd.date_range("2020-01-01", periods=4, freq="D")
        df = pd.DataFrame({"dt": dates.strftime("%Y-%m-%d")})
        meta = {
            "dt": FieldMetadata(data_type=DataType.DATETIME, datetime_format="%Y-%m-%d")
        }
        vae = VAE(df, meta, VaeConfig())
        gen_df = vae.generate(3)
        for val in gen_df["dt"]:
            datetime.strptime(val, "%Y-%m-%d")


if __name__ == "__main__":
    unittest.main()
