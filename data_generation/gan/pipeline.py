"""Defines the GAN class for training and generating synthetic data
using Generative Adversarial Networks (GANs)"""

from __future__ import annotations

import copy
import logging
import math
from typing import Callable, Dict
import numpy as np
import pandas as pd

# plotting and statistical utilities
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import wasserstein_distance

import torch
from torch.nn.functional import mse_loss, relu
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from faker import Faker

from sklearn.model_selection import train_test_split

from data_generation.gan.dataclasses.training import GanConfig
from data_generation.gan.dataclasses.pipeline import (
    DataSchema,
    DataPipeline,
    ModelContainer,
    Optimizers,
    TrainingState,
)
from data_generation.gan.models import _Generator, _Discriminator
from data_generation.gan.utilities import set_seed, amp_autocast, CriticScheduler

# project files
from models.enums import DataType
from models.field_metadata import FieldMetadata

LOGGER = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GAN:
    """Generative Adversarial Network (GAN) for synthetic data generation."""

    def __init__(
        self, real: pd.DataFrame, meta: Dict[str, FieldMetadata], cfg: GanConfig
    ):
        self.cfg = cfg
        set_seed(cfg.training.seed)

        self.schema = DataSchema.from_dataframe(real, meta)
        self.data = DataPipeline.from_schema(self.schema, cfg)
        self._setup_models()
        self._setup_optimizers()
        self.training_state = TrainingState(
            n_critic=self.cfg.optimizer.n_critic_initial
        )

        self.critic_scheduler = CriticScheduler(
            initial_n_critic=self.cfg.optimizer.n_critic_initial
        )

    def _setup_models(self) -> None:
        self.models = ModelContainer(
            generator=_Generator(
                self.cfg,
                len(self.schema.num_cols) + len(self.schema.dt_cols),
                self.data.cat_sizes,
                self.cfg.training.epochs * math.ceil(len(self.data.loader)),
            ).to(DEVICE),
            discriminator=_Discriminator(
                self.data.loader, sum(self.data.cat_sizes), self.cfg
            ).to(DEVICE),
            ema_g=None,
        )
        self.models.ema_g = (
            copy.deepcopy(self.models.generator).eval().requires_grad_(False)
        )

    def _setup_optimizers(self) -> None:
        """Create optimizers, schedulers, and gradient scalers."""
        opt_g = torch.optim.Adam(
            self.models.generator.parameters(),
            lr=self.cfg.optimizer.g_lr,
            betas=(0.0, 0.9),
        )
        opt_d = torch.optim.Adam(
            self.models.discriminator.parameters(),
            lr=self.cfg.optimizer.d_lr,
            betas=(0.0, 0.9),
        )

        self.opt = Optimizers(
            opt_g=opt_g,
            opt_d=opt_d,
            sch_g=CosineAnnealingLR(
                opt_g,
                self.cfg.training.epochs,
                eta_min=self.cfg.optimizer.g_lr * self.cfg.optimizer.lr_min_ratio,
            ),
            sch_d=CosineAnnealingLR(
                opt_d,
                self.cfg.training.epochs,
                eta_min=self.cfg.optimizer.d_lr * self.cfg.optimizer.lr_min_ratio,
            ),
            scaler_g=torch.amp.GradScaler("cuda", enabled=True),
            scaler_d=torch.amp.GradScaler("cuda", enabled=True),
        )

    def _update_metrics(self, epoch, mean_w, d_mean, g_mean) -> None:
        """Update the training metrics after each epoch."""
        lr_discriminator = self.opt.opt_d.param_groups[0]["lr"]
        lr_generator = self.opt.opt_g.param_groups[0]["lr"]

        self.training_state.metrics["epochs"].append(epoch)
        self.training_state.metrics["w_distance"].append(mean_w)
        self.training_state.metrics["d_loss"].append(d_mean)
        self.training_state.metrics["g_loss"].append(g_mean)
        self.training_state.metrics["n_critic"].append(self.training_state.n_critic)
        self.training_state.metrics["lr_d"].append(lr_discriminator)
        self.training_state.metrics["lr_g"].append(lr_generator)

        if self.cfg.training.verbose:
            LOGGER.info(
                "Ep %03d | W %.4f | D %.4f | G %.4f | n_c %d | lr_d %.6f | lr_g %.6f",
                epoch,
                mean_w,
                d_mean,
                g_mean,
                self.training_state.n_critic,
                lr_discriminator,
                lr_generator,
            )

    def _reset_pre_train(self) -> None:
        """Reset the training state before training"""
        dataset: TensorDataset = self.data.loader.dataset
        x_all = dataset.tensors[0]
        n = x_all.size(0)
        idx = np.arange(n)
        train_idx, val_idx = train_test_split(
            idx, test_size=0.2, random_state=self.cfg.training.seed
        )

        # rebuild a loader that only yields the train split
        x_train = x_all[train_idx]
        self.data.loader = DataLoader(
            TensorDataset(x_train),
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            drop_last=True,
        )

        # validation set
        self.training_state.val_df = self.schema.real_df.iloc[val_idx]

        # reset any early‑stop counters
        self.training_state.best_w = -float("inf")
        self.training_state.no_imp = 0
        self.training_state.w_ema = None

        # Reset metrics
        self.training_state.metrics = {
            "epochs": [],
            "w_distance": [],
            "d_loss": [],
            "g_loss": [],
            "n_critic": [],
            "lr_d": [],
            "lr_g": [],
            "val_wd": {c: [] for c in (self.schema.num_cols + self.schema.dt_cols)},
        }

        self.critic_scheduler = CriticScheduler(
            initial_n_critic=self.cfg.optimizer.n_critic_initial
        )

    def _train_epoch(self, epoch: int):
        d_loss_sum: float = 0.0
        g_loss_sum: float = 0.0
        w_loss_sum: float = 0.0
        for i, (real_batch,) in enumerate(self.data.loader, 1):
            real_batch = real_batch.to(DEVICE)
            cond = self._sample_cond(real_batch.size(0)).to(DEVICE)

            # train D
            w_est, d_loss = self._train_d(real_batch, cond)
            w_loss_sum += w_est
            d_loss_sum += d_loss

            # train G every n_critic steps
            if i % self.training_state.n_critic == 0:
                g_loss = self._train_g(real_batch, cond, epoch)
                g_loss_sum += g_loss

        return d_loss_sum, g_loss_sum, w_loss_sum

    def _early_stop(self, d_loss: float) -> bool:
        """Early stop training if no improvement in W-EMA"""
        w_est = -(d_loss / len(self.data.loader))
        self.training_state.w_ema = (
            w_est
            if self.training_state.w_ema is None
            else 0.9 * self.training_state.w_ema + 0.1 * w_est
        )
        if self.training_state.w_ema > self.training_state.best_w + 1e-4:
            self.training_state.best_w, self.training_state.no_imp = (
                self.training_state.w_ema,
                0,
            )
            self.training_state.best_state = copy.deepcopy(
                self.models.generator.state_dict()
            )
        else:
            self.training_state.no_imp += 1

        if self.training_state.no_imp >= self.cfg.scheduler.patience:
            if self.cfg.training.verbose:
                LOGGER.info(
                    "Early stop after %d epochs without improvement",
                    self.cfg.scheduler.patience,
                )
            return True
        return False

    def fit(self):
        """Method for training the GAN model"""
        self._reset_pre_train()

        # the usual GAN training loop
        for epoch in range(1, self.cfg.training.epochs + 1):
            d_loss, g_loss, w_loss = self._train_epoch(epoch)

            # adapt n_critic, decay LRs, EMA‑stop
            mean_w = w_loss / len(self.data.loader)
            self._adjust_n_critic(mean_w)
            self.opt.sch_g.step()
            self.opt.sch_d.step()

            self._update_metrics(
                epoch,
                mean_w,
                d_loss / len(self.data.loader),
                g_loss / max(1, len(self.data.loader) / self.training_state.n_critic),
            )

            if (
                self.training_state.val_df is not None
                and epoch % self.cfg.scheduler.val_interval == 0
            ):
                syn = self.generate(len(self.training_state.val_df))
                val_wd_epoch = {}

                # numeric features
                for c in self.schema.num_cols:
                    real_vals = self.training_state.val_df[c].astype(float).values
                    syn_vals = syn[c].astype(float).values
                    wd = wasserstein_distance(real_vals, syn_vals)
                    val_wd_epoch[c] = wd
                    self.training_state.metrics["val_wd"][c].append((epoch, wd))
                    if self.cfg.training.verbose:
                        LOGGER.info("VAL WD (num) %30s: %.4f", c, wd)

                # datetime features to epoch seconds
                for c in self.schema.dt_cols:
                    real_ts = (
                        pd.to_datetime(
                            self.training_state.val_df[c],
                            format=self.schema.metadata[c].datetime_format,
                        ).astype("int64")
                        // 10**9
                    ).values.astype(float)
                    syn_ts = (
                        pd.to_datetime(
                            syn[c], format=self.schema.metadata[c].datetime_format
                        ).astype("int64")
                        // 10**9
                    ).values.astype(float)
                    wd = wasserstein_distance(real_ts, syn_ts)
                    val_wd_epoch[c] = wd
                    self.training_state.metrics["val_wd"][c].append((epoch, wd))
                    if self.cfg.training.verbose:
                        LOGGER.info("VAL WD (dt)  %30s: %.4f", c, wd)

            if self._early_stop(d_loss):
                break

        # Final visualization after training
        self.plot_training_metrics()

        return self.training_state.metrics

    def generate(
        self, generate_size: int, use_best_model=False, temperature: float = 1.0
    ) -> pd.DataFrame:
        """Generate a synthetic data frame of size *n*."""
        # Use the best model if available and requested
        orig_state = None
        if use_best_model and self.training_state.best_state is not None:
            orig_state = copy.deepcopy(self.models.generator.state_dict())
            self.models.generator.load_state_dict(self.training_state.best_state)
            self.models.ema_g = copy.deepcopy(self.models.generator)

        self.models.ema_g.eval()

        mat = self.apply_temperature(generate_size, temperature)

        # Restore the original model if we switched
        if (
            use_best_model
            and self.training_state.best_state is not None
            and orig_state is not None
        ):
            self.models.generator.load_state_dict(orig_state)
            self.models.ema_g = copy.deepcopy(self.models.generator)

        ptr, data = 0, {}
        for c in self.schema.num_cols:
            inv = self.data.tfs[c].inverse(mat[:, ptr])
            ptr += 1

            # round to integer or fixed decimals
            if self.schema.metadata[c].data_type is DataType.Integer:
                inv = np.round(inv).astype(int)
            else:
                inv = np.round(inv, self.schema.metadata[c].decimal_places)

            data[c] = inv
        for c in self.schema.dt_cols:
            data[c] = self.data.tfs[c].inverse(mat[:, ptr])
            ptr += 1
        for c, k in zip(self.schema.cat_cols, self.data.cat_sizes):
            idx = mat[:, ptr : ptr + k].argmax(1)
            ptr += k
            inv = {v: k for k, v in self.data.cat_maps[c].items()}
            data[c] = [inv[int(i)] for i in idx]
        for c in self.schema.str_cols:
            fn: Callable = self.schema.metadata[c].faker_method or Faker().word
            data[c] = [
                fn(**(self.schema.metadata[c].faker_args or {}))
                for _ in range(generate_size)
            ]

        return pd.DataFrame(data)[list(self.schema.metadata.keys())]

    def apply_temperature(self, generate_size: int, temperature: float):
        """Apply temperature to the generated data."""

        rows = []

        if isinstance(temperature, (list, tuple, np.ndarray)):
            temp_vec = torch.tensor(temperature, device=DEVICE).float()
            assert len(temp_vec) == len(self.schema.num_cols) + len(
                self.schema.dt_cols
            ), "temperature list length mismatch"
        else:
            temp_vec = None

        with torch.no_grad():
            for _ in range(0, generate_size, self.cfg.training.batch_size):
                cur = min(self.cfg.training.batch_size, generate_size - len(rows))
                # Apply temperature to noise distribution
                z = torch.randn(cur, self.cfg.model.latent_dim, device=DEVICE) * (
                    temperature if not temp_vec else 1.0
                )
                out = self.models.ema_g(z, self._sample_cond(cur).to(DEVICE), hard=True)
                if temp_vec is not None:
                    out[:, : len(temp_vec)] *= temp_vec
                rows.append(out.cpu().numpy())

        mat = np.vstack(rows)[:generate_size]

        return mat

    def _train_d(self, real, cond):
        """Train the discriminator on a batch of real data"""
        self.opt.opt_d.zero_grad(set_to_none=True)
        z = torch.randn(real.size(0), self.cfg.model.latent_dim, device=DEVICE)
        fake = self.models.generator(z, cond).detach()
        with amp_autocast():
            d_real = self.models.discriminator(real, cond)
            d_fake = self.models.discriminator(fake, cond)
            # always compute W‐dist estimate for logging
            w_est = d_real.mean() - d_fake.mean()

            if self.cfg.loss.regularization.use_hinge:
                loss = relu(1.0 - d_real).mean() + relu(1.0 + d_fake).mean()
            else:
                gp = self._gp(real, fake, cond)
                loss = -w_est + self.cfg.loss.gp.gp_weight * gp
                loss = loss + self.cfg.loss.gp.drift_epsilon * (
                    d_real.pow(2).mean() + d_fake.pow(2).mean()
                )

        self.opt.scaler_d.scale(loss).backward()
        self.opt.scaler_d.step(self.opt.opt_d)
        self.opt.scaler_d.update()
        return w_est.item(), loss.item()

    def _train_g(self, real, cond, epoch):
        """Train the generator on a batch of real data"""
        self.opt.opt_g.zero_grad(set_to_none=True)
        z = torch.randn(real.size(0), self.cfg.model.latent_dim, device=DEVICE)
        with amp_autocast():
            fake = self.models.generator(z, cond)
            loss_adv = -self.models.discriminator(fake, cond).mean()

            # optional Δ-loss
            if self.cfg.loss.delta.use_delta_loss and self.data.delta_real is not None:
                fake_cont = fake[
                    :, : (len(self.schema.num_cols) + len(self.schema.dt_cols))
                ]
                delta_fake = fake_cont[1:] - fake_cont[:-1]
                # EM distance between distributions (one-sample approx)
                loss_delta = mse_loss(delta_fake.mean(0), self.data.delta_real.mean(0))
                current_w = self.cfg.loss.delta.delta_w * min(
                    1, epoch / self.cfg.loss.delta.delta_warmup
                )
                loss = loss_adv + current_w * loss_delta
            else:
                loss = loss_adv

            if self.cfg.loss.feature_matching.use_fm_loss:
                loss += self.calculate_feature_matching_loss(real, fake, cond)

            if self.cfg.loss.covariance.use_cov_loss:
                loss += self.calculate_covariance_loss(real, fake)
        self.opt.scaler_g.scale(loss).backward()
        self.opt.scaler_g.step(self.opt.opt_g)
        self.opt.scaler_g.update()

        with torch.no_grad():
            for p, p_ema in zip(
                self.models.generator.parameters(), self.models.ema_g.parameters()
            ):
                p_ema.mul_(self.cfg.ema.beta).add_(p.data, alpha=1 - self.cfg.ema.beta)

        return loss.item()

    def calculate_feature_matching_loss(self, real, fake, cond):
        """Calculate the feature matching loss"""
        fr = self.models.discriminator.feature_map(real, cond)
        ff = self.models.discriminator.feature_map(fake, cond)
        fm = mse_loss(ff.mean(0), fr.mean(0))
        return self.cfg.loss.feature_matching.fm_w * fm

    def calculate_covariance_loss(self, real, fake) -> torch.Tensor:
        """Calculate the covariance loss"""
        cont_fake = fake[:, : len(self.schema.num_cols)]
        cont_real = real[:, : len(self.schema.num_cols)]

        # center each feature
        f_centered = cont_fake - cont_fake.mean(dim=0, keepdim=True)
        r_centered = cont_real - cont_real.mean(dim=0, keepdim=True)

        # covariance = (Xᵀ X) / (N - 1)
        cov_f = (f_centered.T @ f_centered) / (cont_fake.size(0) - 1)
        cov_r = (r_centered.T @ r_centered) / (cont_real.size(0) - 1)

        # detach real‐cov so gradient flows only through cov_f
        cov_loss = mse_loss(cov_f, cov_r.detach())
        return self.cfg.loss.covariance.cov_w * cov_loss

    def _sample_cond(self, bsz):
        """Sample a batch of categorical codes for the generator"""
        if not self.schema.cat_cols:
            return torch.zeros(bsz, 0, device=DEVICE)
        cond = torch.zeros(bsz, sum(self.data.cat_sizes), device=DEVICE)
        off = 0
        for c, k in zip(self.schema.cat_cols, self.data.cat_sizes):
            if c in self.data.cat_probs:
                # empirical sampling
                p = self.data.cat_probs[c]
                idx = np.random.choice(k, size=bsz, p=p)
            else:
                # uniform fallback
                idx = np.random.randint(k, size=bsz)
            cond[range(bsz), off + idx] = 1.0
            off += k
        return cond

    def _gp(self, real, fake, cond):
        alpha = torch.rand(real.size(0), 1, device=DEVICE)
        mix = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        with amp_autocast():
            score = self.models.discriminator(mix, cond)
        grad = torch.autograd.grad(score.sum(), mix, create_graph=True)[0]
        return ((grad.view(grad.size(0), -1).norm(2, 1) - 1) ** 2).mean()

    def _adjust_n_critic(self, mean_w: float):
        """Update self.w_smooth and adapt self.n_critic via three‑level hysteresis."""
        self.critic_scheduler.update_smooth_wassertain_distance(mean_w)
        self.training_state.n_critic = self.critic_scheduler.get_updated_n_critic(
            self.training_state.n_critic
        )

    def plot_training_metrics(self, figsize=(15, 10)):
        """
        Plot training metrics including Wasserstein distance, losses, and other training parameters.
        """
        if not self.training_state.metrics["epochs"]:
            LOGGER.warning("No training metrics available to plot")
            return

        plt.figure(figsize=figsize)

        # Create a 2x2 subplot layout
        plt.subplot(2, 2, 1)
        plt.plot(
            self.training_state.metrics["epochs"],
            self.training_state.metrics["w_distance"],
            "b-",
            label="Wasserstein Distance",
        )
        plt.title("Wasserstein Distance")
        plt.xlabel("Epoch")
        plt.ylabel("W-Distance")
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.subplot(2, 2, 2)
        plt.plot(
            self.training_state.metrics["epochs"],
            self.training_state.metrics["d_loss"],
            "r-",
            label="Discriminator Loss",
        )
        plt.plot(
            self.training_state.metrics["epochs"],
            self.training_state.metrics["g_loss"],
            "g-",
            label="Generator Loss",
        )
        plt.title("Training Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.subplot(2, 2, 3)
        plt.plot(
            self.training_state.metrics["epochs"],
            self.training_state.metrics["n_critic"],
            "k-",
        )
        plt.title("n_critic Adaptation")
        plt.xlabel("Epoch")
        plt.ylabel("n_critic")
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.subplot(2, 2, 4)
        plt.plot(
            self.training_state.metrics["epochs"],
            self.training_state.metrics["lr_d"],
            "r-",
            label="D Learning Rate",
        )
        plt.plot(
            self.training_state.metrics["epochs"],
            self.training_state.metrics["lr_g"],
            "g-",
            label="G Learning Rate",
        )
        plt.title("Learning Rates")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()

        plt.show()
