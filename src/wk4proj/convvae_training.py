"""Training utilities for learning a latent representation of Pacific SST."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import xarray as xr
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    """Seed supported random number generators for reproducible experiments."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class ConvVAE(nn.Module):
    """Convolutional VAE for 256 x 256 single-channel SST fields."""

    def __init__(self, latent_dim: int = 16) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.mu = nn.Linear(512 * 8 * 8, latent_dim)
        self.logvar = nn.Linear(512 * 8 * 8, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 512 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                512, 256, 3, stride=2, padding=1, output_padding=1
            ),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                256, 128, 3, stride=2, padding=1, output_padding=1
            ),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                128, 64, 3, stride=2, padding=1, output_padding=1
            ),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                64, 32, 3, stride=2, padding=1, output_padding=1
            ),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.encoder(x).flatten(1)
        return self.mu(hidden), self.logvar(hidden)

    def decode(self, z: Tensor) -> Tensor:
        hidden = self.decoder_input(z).view(-1, 512, 8, 8)
        return self.decoder(hidden)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return self.decode(z), mu, logvar


@dataclass
class Standardizer:
    """Training-set normalization parameters."""

    mean: float
    std: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


def prepare_sst(
    dataset: xr.Dataset,
    *,
    centre_longitude: float = 200,
    centre_latitude: float = -2,
    latitude_size: int = 128,
    longitude_size: int = 256,
    latitude_coarsen: int = 2,
    longitude_coarsen: int = 4,
) -> xr.DataArray:
    """Crop, coarsen, and remove the monthly SST climatology."""
    sst = dataset["sst"].sel(
        latitude=slice(
            centre_latitude + latitude_size / 2,
            centre_latitude - latitude_size / 2,
        ),
        longitude=slice(
            centre_longitude - longitude_size / 2,
            centre_longitude + longitude_size / 2,
        ),
    )
    sst = sst.coarsen(
        latitude=latitude_coarsen,
        longitude=longitude_coarsen,
        boundary="trim",
    ).mean()
    anomaly = sst.groupby("time.month") - sst.groupby("time.month").mean("time")
    return anomaly.transpose("time", "latitude", "longitude")


def make_dataloaders(
    data: xr.DataArray,
    *,
    validation_fraction: float = 0.2,
    batch_size: int = 16,
    seed: int = 0,
) -> tuple[DataLoader[tuple[Tensor]], DataLoader[tuple[Tensor]], Standardizer, int]:
    """Create chronological train/validation loaders and fit normalization on train."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if data.dims != ("time", "latitude", "longitude"):
        raise ValueError("data must have time, latitude, and longitude dimensions")

    values = np.asarray(data.values, dtype=np.float32)
    split = int(values.shape[0] * (1 - validation_fraction))
    if split < 1 or split == values.shape[0]:
        raise ValueError("data must contain enough time steps for both splits")
    training_values = values[:split]
    finite_training_values = training_values[np.isfinite(training_values)]
    if finite_training_values.size == 0:
        raise ValueError("training data contains no finite SST values")
    mean = float(finite_training_values.mean())
    std = float(finite_training_values.std())
    if not np.isfinite(std) or std == 0:
        raise ValueError(
            "training data must have finite SST values with non-zero variation"
        )
    standardizer = Standardizer(mean, std)
    normalized = standardizer.transform(values)
    # Land cells are commonly NaN in SST products. Zero is the climatological
    # anomaly after standardization and keeps those cells from poisoning loss.
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32
    )
    tensors = torch.from_numpy(normalized[:, None])
    train = TensorDataset(tensors[:split])
    validation = TensorDataset(tensors[split:])
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, generator=generator),
        DataLoader(validation, batch_size=batch_size, shuffle=False),
        standardizer,
        split,
    )


def vae_loss(
    reconstruction: Tensor,
    target: Tensor,
    mu: Tensor,
    logvar: Tensor,
    *,
    beta: float = 1e-3,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return total, reconstruction, and KL losses averaged per sample."""
    reconstruction_loss = nn.functional.mse_loss(reconstruction, target, reduction="mean")
    kl_loss = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(dim=1).mean()
    return reconstruction_loss + beta * kl_loss, reconstruction_loss, kl_loss


def _run_epoch(
    model: ConvVAE,
    loader: Iterable[tuple[Tensor]],
    device: torch.device,
    *,
    beta: float,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0}
    samples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for (batch,) in loader:
            batch = batch.to(device)
            mu, logvar = model.encode(batch)
            if training:
                std = torch.exp(0.5 * logvar)
                latent = mu + torch.randn_like(std) * std
            else:
                latent = mu
            reconstruction = model.decode(latent)
            loss, reconstruction_loss, kl_loss = vae_loss(
                reconstruction, batch, mu, logvar, beta=beta
            )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            count = batch.shape[0]
            samples += count
            totals["loss"] += loss.item() * count
            totals["reconstruction"] += reconstruction_loss.item() * count
            totals["kl"] += kl_loss.item() * count
    return {name: value / samples for name, value in totals.items()}


def train_conv_vae(
    model: ConvVAE,
    train_loader: DataLoader[tuple[Tensor]],
    validation_loader: DataLoader[tuple[Tensor]],
    *,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    beta: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 15,
    min_delta: float = 1e-5,
    seed: int | None = None,
    device: str | torch.device | None = None,
) -> list[dict[str, float]]:
    """Train a VAE, stopping and restoring the best validation checkpoint."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if patience < 1:
        raise ValueError("patience must be positive")
    if min_delta < 0:
        raise ValueError("min_delta must not be negative")
    if seed is not None:
        set_seed(seed)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(target_device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")
    best_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.state_dict().items()
    }
    epochs_without_improvement = 0
    for epoch in range(epochs):
        train_metrics = _run_epoch(
            model, train_loader, target_device, beta=beta, optimizer=optimizer
        )
        validation_metrics = _run_epoch(
            model, validation_loader, target_device, beta=beta, optimizer=None
        )
        history.append(
            {
                "epoch": float(epoch + 1),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
            }
        )
        if validation_metrics["loss"] < best_validation_loss - min_delta:
            best_validation_loss = validation_metrics["loss"]
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(
            f"epoch {epoch + 1:03d}/{epochs}: "
            f"train={train_metrics['loss']:.5f}, "
            f"validation={validation_metrics['loss']:.5f}"
        )
        if epochs_without_improvement >= patience:
            print(f"early stopping after {epoch + 1} epochs")
            break
    model.load_state_dict(best_state)
    return history


@torch.no_grad()
def encode_latents(
    model: ConvVAE,
    data: xr.DataArray,
    standardizer: Standardizer,
    *,
    batch_size: int = 32,
    device: str | torch.device | None = None,
) -> xr.Dataset:
    """Encode every SST field and return latent means indexed by timestamp."""
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(target_device)
    model.eval()
    values = standardizer.transform(np.asarray(data.values, dtype=np.float32))
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(values[:, None])), batch_size=batch_size)
    latent_means: list[np.ndarray] = []
    for (batch,) in loader:
        mu, _ = model.encode(batch.to(target_device))
        latent_means.append(mu.cpu().numpy())
    latent = np.concatenate(latent_means, axis=0)
    coordinates = {"time": data.time, "latent": np.arange(model.latent_dim)}
    return xr.Dataset(
        {"latent_mean": (("time", "latent"), latent)},
        coords=coordinates,
    )
