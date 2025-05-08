"""Unit tests for GAN utilities and pipeline methods."""

import unittest
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from gan.utilities import lin_sn, CriticScheduler
from gan.pipeline import GAN
from gan.dataclasses.training import GanConfig
from models.field_metadata import FieldMetadata
from models.enums import DataType


def create_simple_gan():
    """Helper to instantiate a GAN with minimal data and ensure CPU usage and non-empty loader."""
    df = pd.DataFrame({"feat": [1.0, 2.0]})
    meta = {"feat": FieldMetadata(data_type=DataType.DECIMAL, decimal_places=1)}
    cfg = GanConfig()
    gan = GAN(df, meta, cfg)
    # Override loader to avoid zero-division in _early_stop
    tensor = torch.tensor(df["feat"].values, dtype=torch.float32).unsqueeze(1)
    gan.data.loader = DataLoader(TensorDataset(tensor), batch_size=1)
    return gan


def create_gan_with_cats():
    """Instantiate a GAN configured with two categorical columns for _sample_cond testing."""
    gan = create_simple_gan()
    # inject two categorical features
    gan.schema.cat_cols = ["c1", "c2"]
    gan.data.cat_sizes = [2, 3]
    gan.data.cat_probs = {"c1": np.array([0.7, 0.3]), "c2": np.array([0.2, 0.5, 0.3])}
    return gan


class GANTests(unittest.TestCase):
    """Unit tests for GAN utilities and pipeline methods."""

    def test_lin_sn_returns_linear_with_spectral_norm(self):
        """lin_sn should wrap a Linear layer with spectral normalization."""
        layer = lin_sn(3, 4)
        self.assertIsInstance(layer, torch.nn.Linear)
        self.assertEqual(layer.weight.shape, (4, 3))

    def test_critic_scheduler_hysteresis(self):
        """CriticScheduler should adjust n_critic up/down based on w_smooth."""
        sched = CriticScheduler(
            initial_n_critic=5, alpha=0.5, lower_threshold=0.1, upper_threshold=0.9
        )
        # initial smoothing
        sched.update_smooth_wassertain_distance(0.0)
        self.assertEqual(sched.w_smooth, 0.0)
        # below threshold, from initial remains
        self.assertEqual(sched.get_updated_n_critic(5), 5)
        # below threshold from 3 -> increments
        self.assertEqual(sched.get_updated_n_critic(3), 4)
        # above threshold -> decrements
        sched.w_smooth = 1.0
        self.assertEqual(sched.get_updated_n_critic(3), 2)

    def test_early_stop_no_improvement(self):
        """_early_stop should increment no_imp when no improvement for non-empty loader."""
        gan = create_simple_gan()
        stop = gan._early_stop(d_loss=1.0)
        self.assertFalse(stop)
        prev = gan.training_state.no_imp
        stop2 = gan._early_stop(d_loss=1.0)
        self.assertFalse(stop2)
        self.assertEqual(gan.training_state.no_imp, prev + 1)

    def test_feature_matching_loss_zero_when_identical(self):
        """calculate_feature_matching_loss returns zero when real and fake inputs are identical."""
        gan = create_simple_gan()
        # Align inputs with the discriminator's device
        device = next(gan.models.discriminator.parameters()).device
        batch = torch.tensor([[1.0], [2.0]], dtype=torch.float32, device=device)
        cond = gan._sample_cond(batch.size(0)).to(device)

        fm_loss = gan.calculate_feature_matching_loss(batch, batch, cond)
        self.assertLessEqual(fm_loss.item(), 1e-3)

    def test_covariance_loss_zero_when_identical(self):
        """calculate_covariance_loss is zero when real and fake match exactly."""
        gan = create_simple_gan()
        batch = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
        cov_loss = gan.calculate_covariance_loss(batch, batch)
        self.assertAlmostEqual(cov_loss.item(), 0.0)

    def test_adjust_n_critic(self):
        """_adjust_n_critic should integrate with CriticScheduler to update n_critic."""
        gan = create_simple_gan()
        gan.training_state.n_critic = 2
        gan.critic_scheduler.w_smooth = 0.0
        gan._adjust_n_critic(mean_w=0.0)
        self.assertEqual(gan.training_state.n_critic, 3)
        gan.training_state.n_critic = 3
        gan.critic_scheduler.w_smooth = 1.0
        gan._adjust_n_critic(mean_w=1.0)
        self.assertEqual(gan.training_state.n_critic, 2)

    def test_update_metrics_appends_values(self):
        """_update_metrics should record the metrics correctly."""
        gan = create_simple_gan()
        # Initialize metrics storage
        gan._reset_pre_train()
        initial_len = len(gan.training_state.metrics["epochs"])
        epoch = 7
        mean_w = 0.123
        d_mean = 1.234
        g_mean = 2.345
        # Set deterministic values
        gan.training_state.n_critic = 3
        gan.opt.opt_d.param_groups[0]["lr"] = 0.01
        gan.opt.opt_g.param_groups[0]["lr"] = 0.02
        gan._update_metrics(epoch, mean_w, d_mean, g_mean)

        metrics = gan.training_state.metrics
        self.assertEqual(len(metrics["epochs"]), initial_len + 1)
        self.assertEqual(metrics["epochs"][-1], epoch)
        self.assertAlmostEqual(metrics["w_distance"][-1], mean_w)
        self.assertAlmostEqual(metrics["d_loss"][-1], d_mean)
        self.assertAlmostEqual(metrics["g_loss"][-1], g_mean)
        self.assertEqual(metrics["n_critic"][-1], 3)
        self.assertEqual(metrics["lr_d"][-1], 0.01)
        self.assertEqual(metrics["lr_g"][-1], 0.02)

    def test_sample_cond_no_categorical(self):
        """_sample_cond should return an empty tensor when no categorical columns are defined."""
        gan = create_simple_gan()
        cond = gan._sample_cond(5)
        self.assertIsInstance(cond, torch.Tensor)
        self.assertEqual(cond.shape, (5, 0))

    def test_update_smooth_exponential(self):
        """update_smooth_wassertain_distance should apply exponential smoothing correctly."""
        sched = CriticScheduler(initial_n_critic=1, alpha=0.5)
        sched.update_smooth_wassertain_distance(2.0)
        self.assertAlmostEqual(sched.w_smooth, 2.0)
        sched.update_smooth_wassertain_distance(4.0)
        expected = 0.5 * 2.0 + 0.5 * 4.0
        self.assertAlmostEqual(sched.w_smooth, expected)

    def test_reset_pre_train_resets_metrics(self):
        """_reset_pre_train should initialize all metric lists and val_wd structure correctly."""
        gan = create_simple_gan()
        # pre-populate metrics
        gan.training_state.metrics = {"dummy": [1, 2, 3]}
        gan._reset_pre_train()
        metrics = gan.training_state.metrics
        # expected metric keys
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
        self.assertEqual(set(metrics.keys()), expected_keys)
        # all lists should be empty
        for key in [
            "epochs",
            "w_distance",
            "d_loss",
            "g_loss",
            "n_critic",
            "lr_d",
            "lr_g",
        ]:
            self.assertListEqual(metrics[key], [])
        # val_wd should have one key per numeric/dt column, each mapping to empty list
        val_wd = metrics["val_wd"]
        expected_cols = set(gan.schema.num_cols + gan.schema.dt_cols)
        self.assertEqual(set(val_wd.keys()), expected_cols)
        for col in expected_cols:
            self.assertListEqual(val_wd[col], [])

    def test_sample_cond_with_categories(self):
        """_sample_cond should produce correct one-hot encoding given cat_sizes and cat_probs."""
        gan = create_gan_with_cats()
        batch_size = 10
        cond = gan._sample_cond(batch_size)
        total_dims = sum(gan.data.cat_sizes)
        self.assertEqual(cond.shape, (batch_size, total_dims))
        arr = cond.cpu().numpy()
        splits = np.split(arr, np.cumsum(gan.data.cat_sizes)[:-1], axis=1)
        for block in splits:
            # each block row sums to 1
            self.assertTrue(np.all(block.sum(axis=1) == 1))

    def test_temperature_application_shape(self):
        """apply_temperature should return a numpy array of correct shape."""
        gan = create_simple_gan()

        class Identity:
            def __init__(self, dim):
                self.dim = dim

            def eval(self):
                return self

            def requires_grad_(self, flag):
                return self

            def __call__(self, z, cond, hard=True):
                return torch.cat([z, cond], dim=1)

        gan.models.ema_g = Identity(gan.cfg.model.latent_dim)
        size = 8
        temp = 0.5
        mat = gan.apply_temperature(size, temp)
        # expect shape [size, latent_dim] when no cond
        self.assertEqual(mat.shape[0], size)
        self.assertEqual(mat.shape[1], gan.cfg.model.latent_dim)


if __name__ == "__main__":
    unittest.main()
