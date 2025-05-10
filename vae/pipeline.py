"""
vae.py

Implements a Variational Autoencoder (VAE) for learning latent representations of tabular data
and generating synthetic samples. Includes model definition, training loop, and data handling:

- VAE: encoder–decoder network with configurable latent dimension.
- Loss components: reconstruction loss (numeric and categorical), KL divergence, corr penalty.
- Data pipeline: apply continuous transformers, one‐hot encode categoricals, and batch sampling.
- Utilities: model checkpointing, sample generation, and training progress logging.
"""

from __future__ import annotations
import logging
import math
import itertools
from typing import Union, Dict, Any

import torch
from torch.nn import functional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.utilities import set_seed
from common.dataclasses import DataSchema
from common.transformers import ContTf, DtTf
from vae.models import Encoder, Decoder
from vae.dataclasses.training import (
    VaeConfig,
)
from vae.dataclasses.pipeline import (
    TrainingDataContainer,
    ModelContainer,
    TrainingState,
    DataPipeline,
)
from vae.utilities import sliced_wasserstein, batch_corr
from models.enums import DataType
from models.field_metadata import FieldMetadata

LOGGER = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAE:
    """Variational Autoencoder for tabular data."""

    def __init__(
        self, real_df: pd.DataFrame, meta: dict[str, FieldMetadata], config: VaeConfig
    ):
        self.config = config
        self.schema = DataSchema.from_dataframe(real_df, meta)

        set_seed(self.config.training.seed)

        self.training_data = TrainingDataContainer(
            tf_num={
                c: ContTf().fit(self.schema.real_df[c]) for c in self.schema.num_cols
            },
            tf_dt={
                c: DtTf(meta[c].datetime_format).fit(self.schema.real_df[c])
                for c in self.schema.dt_cols
            },
            cat_maps={
                c: pd.Categorical(self.schema.real_df[c]).categories
                for c in self.schema.cat_cols
            },
            num_dim=len(self.schema.num_cols + self.schema.dt_cols),
            cat_dims=None,
            real_corr=None,
            cat_targets=[],
        )

        self.training_data.cat_dims = [
            len(self.training_data.cat_maps[c]) for c in self.schema.cat_cols
        ]

        # pre-compute corpus probabilities for every categorical
        for c in self.schema.cat_cols:
            # align counts with the fixed category order in cat_maps
            counts = (
                self.schema.real_df[c]
                .value_counts(sort=False)
                .reindex(self.training_data.cat_maps[c], fill_value=0)
                .values
            )
            probs = torch.tensor(
                counts / len(self.schema.real_df), dtype=torch.float32, device=DEVICE
            )
            self.training_data.cat_targets.append(probs)

        real_cont = np.stack(
            [
                self.training_data.tf_num[c].transform(self.schema.real_df[c])
                for c in self.schema.num_cols
            ]
            + [
                self.training_data.tf_dt[c].transform(self.schema.real_df[c])
                for c in self.schema.dt_cols
            ],
            1,
        )
        self.training_data.real_corr = torch.tensor(
            np.corrcoef(real_cont, rowvar=False), dtype=torch.float32, device=DEVICE
        )

        self.models = ModelContainer(
            encoder=Encoder(
                self.training_data.num_dim + sum(self.training_data.cat_dims),
                self.config.model.latent_dim,
            ).to(DEVICE),
            decoder=Decoder(
                self.training_data.num_dim,
                self.training_data.cat_dims,
                self.config.model.latent_dim,
            ).to(DEVICE),
            optimizer=None,
            scheduler=None,
        )

        self.models.optimizer = torch.optim.AdamW(
            itertools.chain(
                self.models.encoder.parameters(), self.models.decoder.parameters()
            ),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )

        self.models.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.models.optimizer,
            lambda ep: (
                (ep + 1) / 10
                if ep < 10
                else 0.5
                * (
                    1
                    + math.cos(math.pi * (ep - 10) / (self.config.training.epochs - 10))
                )
            ),
        )

        self.training_state = TrainingState()
        self.data_pipeline = DataPipeline.from_schema(self.schema, self.config)

    def _encode(self, df: pd.DataFrame) -> torch.Tensor:
        """Encodes a DataFrame into a tensor."""
        num = torch.stack(
            [
                torch.tensor(
                    self.training_data.tf_num[c].transform(df[c]).copy(),
                    dtype=torch.float32,
                )
                for c in self.schema.num_cols
            ]
            + [
                torch.tensor(
                    self.training_data.tf_dt[c].transform(df[c]).copy(),
                    dtype=torch.float32,
                )
                for c in self.schema.dt_cols
            ],
            1,
        )
        cats = []
        for c, dim in zip(self.schema.cat_cols, self.training_data.cat_dims):
            idx = pd.Categorical(df[c], categories=self.training_data.cat_maps[c]).codes
            cats.append(torch.eye(dim, device=DEVICE)[idx])
        return torch.cat([num.to(DEVICE)] + cats, 1)

    def _reset_pretrain(self):
        """Reset training state"""
        self.training_state = TrainingState()
        self.data_pipeline = DataPipeline.from_schema(self.schema, self.config)

    def _calculate_beta(self, epoch: int) -> float:
        """Linear warm-up of the KL weight until `beta_warmup`, then constant."""
        w = self.config.model.beta_warmup
        if epoch <= w:
            return self.config.model.kl_max * epoch / w
        return self.config.model.kl_max

    def _calculate_categorical_loss(self, x, logits):
        """Calculate categorical loss"""
        if not self.training_data.cat_dims:
            return torch.tensor(0.0, device=DEVICE)

        loss_cat = 0
        start = self.training_data.num_dim
        for l, dim in zip(logits, self.training_data.cat_dims):
            tgt = x[:, start : start + dim].argmax(1)
            loss_cat += functional.cross_entropy(l, tgt)
            start += dim
        return loss_cat / len(self.training_data.cat_dims)

    def _calculate_correlation_loss(self, synthetic_cont):
        """Calculate correlation loss"""
        if synthetic_cont.size(1) <= 1:
            return torch.tensor(0.0, device=synthetic_cont.device)

        corr_mat = batch_corr(synthetic_cont)
        return functional.l1_loss(corr_mat, self.training_data.real_corr)

    def _calculate_kl_loss(self, logits):
        """Calculate kl loss"""
        cat_freq_loss = 0.0
        for l, target in zip(logits, self.training_data.cat_targets):
            cat_freq_loss += functional.kl_div(
                l.softmax(dim=1).mean(0).log(), target, reduction="batchmean"
            )
        return cat_freq_loss

    def _train_epoch(self, epoch, beg, beta) -> None:
        batch = self.schema.real_df.iloc[
            self.data_pipeline.training_df[beg : beg + self.config.training.batch_size]
        ].reset_index(drop=True)
        x = self._encode(batch)
        z, mu, logv = self.models.encoder(x)
        z_cat, z_num = torch.chunk(z, 2, 1)

        num_hat, logits, _ = self.models.decoder(
            z_cat,
            z_num,
            tau=max(0.2, math.exp(-0.01 * epoch)),
            hard=max(0.2, math.exp(-0.01 * epoch)) <= 0.3,
        )

        # numerical loss
        loss = functional.mse_loss(num_hat, x[:, : self.training_data.num_dim])
        # categorical loss
        loss += 2 * self._calculate_categorical_loss(x, logits)

        loss += beta * (-0.5 * (1 + logv - mu.pow(2) - logv.exp()).mean())

        with torch.no_grad():
            synthetic_cont = self.generate(len(batch), temperature=0.5, _cpu=False)[
                "cont"
            ]
        # wasserstain distance
        loss += (5 * beta) * sliced_wasserstein(
            synthetic_cont, x[:, : self.training_data.num_dim]
        )
        # correlation loss
        loss += (10 * beta) * self._calculate_correlation_loss(synthetic_cont)
        # kl loss
        loss += (2.0 * beta) * self._calculate_kl_loss(logits)

        self.models.optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            itertools.chain(
                self.models.encoder.parameters(), self.models.decoder.parameters()
            ),
            max_norm=5.0,
        )

        self.models.optimizer.step()

    def fit(self) -> None:
        """Trains the VAE model on the provided DataFrame."""
        self._reset_pretrain()

        for epoch in range(1, self.config.training.epochs + 1):
            np.random.shuffle(self.data_pipeline.training_df)
            beta = self._calculate_beta(epoch)

            for beg in range(
                0, len(self.data_pipeline.training_df), self.config.training.batch_size
            ):
                self._train_epoch(epoch, beg, beta)

            self.models.scheduler.step()

            # validation, early stop
            if epoch % 5 == 0 or epoch == self.config.training.epochs:
                val_fake = self.generate(
                    len(self.data_pipeline.validation_df), temperature=0.5, _cpu=False
                )["cont"]
                val_real = self._encode(
                    self.schema.real_df.iloc[self.data_pipeline.validation_df]
                )[:, : self.training_data.num_dim]
                swd_val = sliced_wasserstein(val_fake, val_real).item()
                if self.config.training.verbose:
                    LOGGER.info("Ep %03d  β=%0.3f  SWD_val=%0.4f", epoch, beta, swd_val)

                if self.config.training.verbose:
                    # record for final plot
                    self.training_state.history_ep.append(epoch)
                    self.training_state.history_swd.append(swd_val)
                    self.training_state.history_beta.append(beta)
                if swd_val < self.training_state.best - 1e-4:
                    self.training_state.best, self.training_state.no_improvements = (
                        swd_val,
                        0,
                    )
                    torch.save(
                        {
                            "enc": self.models.encoder.state_dict(),
                            "dec": self.models.decoder.state_dict(),
                        },
                        "vae_best.pt",
                    )
                else:
                    self.training_state.no_improvements += 1
                    if self.training_state.no_improvements > 20:
                        if self.config.training.verbose:
                            LOGGER.info("Early stop")
                        break
        self.models.encoder.load_state_dict(torch.load("vae_best.pt")["enc"])
        self.models.decoder.load_state_dict(torch.load("vae_best.pt")["dec"])

        if self.config.training.verbose:
            fig, ax = plt.subplots()
            ax.plot(
                self.training_state.history_ep,
                self.training_state.history_swd,
                "-o",
                label="SWD_val",
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("SWD_val", color="C0")
            ax.tick_params(axis="y", labelcolor="C0")
            ax2 = ax.twinx()
            ax2.plot(
                self.training_state.history_ep,
                self.training_state.history_beta,
                "-s",
                color="C1",
                label="β",
            )
            ax2.set_ylabel("β (KL weight)", color="C1")
            ax2.tick_params(axis="y", labelcolor="C1")
            fig.tight_layout()
            plt.title("VAE training: SWD vs β")
            plt.show()

    def generate(
        self, generate_size: int, temperature: float = 0.8, _cpu: bool = True
    ) -> Union[pd.DataFrame, Dict[str, Any]]:
        """Generate synthetic samples from the VAE model."""
        self.models.decoder.eval()
        cont, cats = self._generate_step(generate_size, temperature)

        # inverse-transform continuous
        data, k = self._generate_numeric(cont)

        datetime_dict = self._generate_datetime(cont, k)
        if datetime_dict:
            data |= datetime_dict
            k += len(datetime_dict)

        # inverse-transform categoricals / booleans
        start = 0
        for c, dim in zip(self.schema.cat_cols, self.training_data.cat_dims):
            idx = cats[:, start : start + dim].argmax(1).cpu().numpy()
            data[c] = self.training_data.cat_maps[c][idx]
            start += dim

        data |= self._generate_string(generate_size)

        df = pd.DataFrame(data)
        return {"df": df, "cont": cont} if not _cpu else df

    def _generate_step(self, generate_size: int, temperature: float):
        with torch.no_grad():
            out_num, out_cat = [], []
            bs = 1024
            # start is how many samples have already been handled
            for start in range(0, generate_size, bs):
                cur = min(bs, generate_size - start)
                z = torch.randn(cur, self.config.model.latent_dim, device=DEVICE)
                z_cat, z_num = torch.chunk(z, 2, 1)
                num, _, cond = self.models.decoder(
                    z_cat, z_num, tau=temperature, hard=True
                )
                out_num.append(num)
                out_cat.append(cond)
            # now we have exactly n samples
            cont = torch.cat(out_num, dim=0).detach()
            cats = torch.cat(out_cat, dim=0).detach()

        return cont, cats

    def _generate_string(self, generate_size: int):
        data = {}
        # generate strings with Faker or empirical sampling
        for c in self.schema.str_cols:
            m = self.schema.metadata[c]
            if m.faker_method is not None:
                data[c] = [
                    m.faker_method(**getattr(m, "faker_args", {}))
                    for _ in range(generate_size)
                ]
            else:
                # fall back to empirical sampling
                data[c] = (
                    self.schema.real_df[c].sample(generate_size, replace=True).to_list()
                )
        return data

    def _generate_numeric(self, cont: torch.Tensor):
        data, k = {}, 0
        for c in self.schema.num_cols:
            vals = self.training_data.tf_num[c].inverse(cont[:, k].cpu().numpy())
            data[c] = (
                np.round(vals).astype(int)
                if self.schema.metadata[c].data_type == DataType.INTEGER
                else np.round(vals, self.schema.metadata[c].decimal_places or 2)
            )
            k += 1
        return data, k

    def _generate_datetime(self, cont: torch.Tensor, cur_k: int):
        data = {}
        for c in self.schema.dt_cols:
            data[c] = self.training_data.tf_dt[c].inverse(cont[:, cur_k].cpu().numpy())
            cur_k += 1
        return data
