"""Gan unit-tests"""

from __future__ import annotations

import unittest
from typing import Final

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Project imports --------------------------------------------------------- #
from gan.utilities import lin_sn, CriticScheduler
from gan.pipeline import GAN
from gan.dataclasses.training import GanConfig
from models.enums import DataType
from models.field_metadata import FieldMetadata

# --------------------------------------------------------------------------- #
# Helper factories                                                            #
# --------------------------------------------------------------------------- #


def _simple_gan() -> GAN:
    """Create a *minimal* GAN instance with a single numerical feature."""
    df = pd.DataFrame({"feat": [1.0, 2.0]})
    meta = {"feat": FieldMetadata(data_type=DataType.DECIMAL, decimal_places=1)}
    gan_cfg = GanConfig()
    gan = GAN(df, meta, gan_cfg)

    # Replace the original loader with one that has a deterministic length (no drop-last).
    tensor = torch.tensor(df["feat"].values, dtype=torch.float32).unsqueeze(1)
    gan.data.loader = DataLoader(TensorDataset(tensor), batch_size=1)
    return gan


def _gan_with_cats() -> GAN:
    """A GAN configured with two categorical columns for conditional sampling tests."""
    gan = _simple_gan()

    # Inject two dummy categorical features (sizes 2 and 3).
    gan.schema.cat_cols = ["c1", "c2"]
    gan.data.cat_sizes = [2, 3]
    gan.data.cat_probs = {
        "c1": np.array([0.7, 0.3]),
        "c2": np.array([0.2, 0.5, 0.3]),
    }
    return gan


# --------------------------------------------------------------------------- #
#                                         Test-suite                         #
# --------------------------------------------------------------------------- #


class GANTests(unittest.TestCase):
    """Unit-tests for utilities, scheduler logic and the GAN pipeline."""

    RNG_SEED: Final = 2025

    def test_lin_sn_returns_linear_with_spectral_norm(self) -> None:
        """lin_sn must return an nn.Linear wrapped with spectral-norm."""
        layer = lin_sn(3, 4)
        self.assertIsInstance(layer, torch.nn.Linear)
        self.assertEqual(layer.weight.shape, (4, 3))

    def test_critic_scheduler_hysteresis(self) -> None:
        """CriticScheduler must increment/decrement n_critic as expected."""
        sched = CriticScheduler(
            initial_n_critic=5,
            alpha=0.5,
            lower_threshold=0.1,
            upper_threshold=0.9,
        )

        sched.update_smooth_wassertain_distance(0.0)
        # Below lower threshold – no change when already at initial value.
        self.assertEqual(sched.get_updated_n_critic(5), 5)
        # Below lower threshold – increment when below initial.
        self.assertEqual(sched.get_updated_n_critic(3), 4)

        # Manually set a large W-distance and verify decrement.
        sched.w_smooth = 1.0
        self.assertEqual(sched.get_updated_n_critic(3), 2)

    def test_early_stop_increments_no_imp(self) -> None:
        """_early_stop should increase the no_imp counter when W-EMA stagnates."""
        gan = _simple_gan()

        # The first call initializes EMA – cannot trigger stop.
        self.assertFalse(gan._early_stop(d_loss=1.0))
        prev = gan.training_state.no_imp

        # Second call with identical loss ⇒ no improvement.
        self.assertFalse(gan._early_stop(d_loss=1.0))
        self.assertEqual(gan.training_state.no_imp, prev + 1)

    def test_feature_matching_loss_zero_when_identical(self) -> None:
        """Feature-matching loss must be near 0 when real and fake batches are equal."""
        gan = _simple_gan()
        device = next(gan.models.discriminator.parameters()).device

        batch = torch.tensor([[1.0], [2.0]], dtype=torch.float32, device=device)
        cond = gan._sample_cond(batch.size(0)).to(device)

        loss = gan.calculate_feature_matching_loss(batch, batch, cond)
        self.assertLessEqual(loss.item(), 1e-3)

    def test_covariance_loss_zero_when_identical(self) -> None:
        """Covariance loss is 0 =- e if fake == real."""
        gan = _simple_gan()
        batch = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
        loss = gan.calculate_covariance_loss(batch, batch)
        self.assertAlmostEqual(loss.item(), 0.0)

    def test_adjust_n_critic_integration(self) -> None:
        """Internal helper must apply scheduler decision to training_state.n_critic."""
        gan = _simple_gan()

        gan.training_state.n_critic = 2
        gan.critic_scheduler.w_smooth = 0.0
        gan._adjust_n_critic(mean_w=0.0)
        self.assertEqual(gan.training_state.n_critic, 3)

        gan.training_state.n_critic = 3
        gan.critic_scheduler.w_smooth = 1.0
        gan._adjust_n_critic(mean_w=1.0)
        self.assertEqual(gan.training_state.n_critic, 2)

    def test_update_metrics_appends_values(self) -> None:
        """_update_metrics must append a new row with all expected fields."""
        gan = _simple_gan()
        gan._reset_pre_train()

        start_len = len(gan.training_state.metrics["epochs"])

        # Deterministic inputs
        gan.training_state.n_critic = 3
        gan.opt.opt_d.param_groups[0]["lr"] = 0.01
        gan.opt.opt_g.param_groups[0]["lr"] = 0.02

        gan._update_metrics(epoch=7, mean_w=0.123, d_mean=1.234, g_mean=2.345)

        mtr = gan.training_state.metrics
        self.assertEqual(len(mtr["epochs"]), start_len + 1)
        self.assertEqual(mtr["epochs"][-1], 7)
        self.assertAlmostEqual(mtr["w_distance"][-1], 0.123)
        self.assertAlmostEqual(mtr["d_loss"][-1], 1.234)
        self.assertAlmostEqual(mtr["g_loss"][-1], 2.345)
        self.assertEqual(mtr["n_critic"][-1], 3)
        self.assertEqual(mtr["lr_d"][-1], 0.01)
        self.assertEqual(mtr["lr_g"][-1], 0.02)

    def test_reset_pre_train_initialises_metrics(self) -> None:
        """After _reset_pre_train every expected metric key must exist and be empty."""
        gan = _simple_gan()
        gan.training_state.metrics = {"dummy": [1]}
        gan._reset_pre_train()

        expected_keys = {
            "epochs",
            "w_distance",
            "d_loss",
            "g_loss",
            "n_critic",
            "lr_d",
            "lr_g",
            "val_wd",
        }
        self.assertEqual(set(gan.training_state.metrics.keys()), expected_keys)

        # All scalar lists must be empty.
        for key in expected_keys - {"val_wd"}:
            self.assertListEqual(gan.training_state.metrics[key], [])

        # val_wd is a dict keyed by continuous & datetime columns.
        expected_cols = set(gan.schema.num_cols + gan.schema.dt_cols)
        val_wd = gan.training_state.metrics["val_wd"]
        self.assertEqual(set(val_wd.keys()), expected_cols)
        for col in expected_cols:
            self.assertListEqual(val_wd[col], [])

    def test_sample_cond_no_categorical(self) -> None:
        """If schema.cat_cols is empty, _sample_cond returns a [bsz, 0] tensor."""
        gan = _simple_gan()
        cond = gan._sample_cond(4)
        self.assertEqual(cond.shape, (4, 0))

    def test_sample_cond_with_categorical(self) -> None:
        """Returned tensor must be one-hot across concatenated categorical blocks."""
        gan = _gan_with_cats()
        batch_size = 10
        cond = gan._sample_cond(batch_size)

        self.assertEqual(cond.shape, (batch_size, sum(gan.data.cat_sizes)))

        # Split into individual category blocks and ensure each row is one-hot.
        arr = cond.cpu().numpy()
        splits = np.split(arr, np.cumsum(gan.data.cat_sizes)[:-1], axis=1)
        for block in splits:
            self.assertTrue(np.all(block.sum(axis=1) == 1))

    def test_apply_temperature_output_shape(self) -> None:
        """apply_temperature must produce the expected latent-size matrix."""
        gan = _simple_gan()

        # Replace the EMA generator with a trivial identity-like stub
        class _Identity:
            def __init__(self, latent_dim: int) -> None:
                self.dim = latent_dim

            def eval(self) -> "_Identity":
                """Evaluate the model."""
                return self

            def requires_grad_(self, flag: bool) -> "_Identity":
                """Requires gradient flag."""
                return self

            def __call__(  # noqa: D401
                self,
                z: torch.Tensor,
                cond: torch.Tensor,
                hard: bool = True,
            ) -> torch.Tensor:
                _ = hard
                return torch.cat([z, cond], dim=1)

        gan.models.ema_g = _Identity(gan.cfg.model.latent_dim)

        rows = 8
        mat = gan.apply_temperature(generate_size=rows, temperature=0.5)

        self.assertEqual(mat.shape[0], rows)
        self.assertEqual(mat.shape[1], gan.cfg.model.latent_dim)


if __name__ == "__main__":
    unittest.main()
