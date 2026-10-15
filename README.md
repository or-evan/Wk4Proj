# SST convVAE

The notebook contains the exploratory preprocessing and model definition. The
training utilities live in
[`src/wk4proj/convvae_training.py`](src/wk4proj/convvae_training.py), so the
same training and latent extraction code can be used from a notebook or a
script.

```python
import xarray as xr
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import numpy as np

from wk4proj.convvae_training import (
    ConvVAE,
    encode_latents,
    make_dataloaders,
    prepare_sst,
    set_seed,
    train_conv_vae,
)

seed = 42
set_seed(seed)
ds = xr.open_dataset("data/data.grib")
sst_anomaly = prepare_sst(ds)
train_loader, validation_loader, standardizer, split = make_dataloaders(
    sst_anomaly, batch_size=16, validation_fraction=0.2, seed=seed
)

model = ConvVAE(latent_dim=16)
history = train_conv_vae(
    model,
    train_loader,
    validation_loader,
    epochs=100,
    beta=1e-3,
    patience=15,
    seed=seed,
)
latents = encode_latents(model, sst_anomaly, standardizer)

# Match latent months to the categories in data/oni.csv.
oni_by_date = {}
with Path("data/oni.csv").open(newline="") as oni_file:
    for row in csv.DictReader(oni_file):
        oni_by_date[row["date"]] = row["category"]

latent_dates = latents.time.dt.strftime("%Y-%m-%d").values
categories = np.array(
    [oni_by_date.get(date, "Unlabelled") for date in latent_dates]
)
matched = categories != "Unlabelled"

category_colours = {
    "VSE": "#b2182b", "SE": "#ef8a62", "ME": "#fddbc7",
    "N": "#bdbdbd", "ML": "#d9f0d3", "WL": "#7fbf7b",
    "SL": "#1b7837", "VSL": "#276419", "WE": "#5aae61",
}

from itertools import combinations
from matplotlib.patches import Patch

latent_values = latents.latent_mean.values[matched]
latent_dimension = latent_values.shape[1]
fig, axes = plt.subplots(
    latent_dimension, latent_dimension, figsize=(16, 16),
    sharex="col", sharey="row", squeeze=False,
)
for row, column in combinations(range(latent_dimension), 2):
    ax = axes[column, row]
    for category, colour in category_colours.items():
        selected = categories[matched] == category
        ax.scatter(
            latent_values[selected, row], latent_values[selected, column],
            color=colour, s=4, alpha=0.7, edgecolors="none",
            rasterized=True,
        )
    ax.set_xlabel(f"z{row}", fontsize=7)
    ax.set_ylabel(f"z{column}", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.2)
for index in range(latent_dimension):
    axes[index, index].text(
        0.5, 0.5, f"z{index}", ha="center", va="center",
        fontsize=8, transform=axes[index, index].transAxes,
    )
    axes[index, index].set_axis_off()
for row in range(latent_dimension):
    for column in range(row + 1, latent_dimension):
        axes[row, column].set_visible(False)
fig.legend(
    handles=[Patch(facecolor=colour, label=category)
             for category, colour in category_colours.items()],
    title="ONI category", loc="upper center", ncol=5,
    fontsize=8, title_fontsize=9,
)
fig.suptitle("Pairwise SST latent spaces coloured by ONI category", y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.97), pad=0.3)
plt.show()
```

The split is chronological: the first 80% of months are used for training and
the final 20% for validation. Normalization statistics are fitted only on the
training period. `latents` contains the posterior means (`latent_mean`) for
each timestamp; use the time coordinate to group or annotate points by year.
Missing SST cells, such as land points represented by `NaN`, are excluded from
the normalization statistics and encoded as zero anomaly before training.
Validation uses the latent mean rather than a random latent sample, making its
loss stable between epochs. Training uses early stopping and restores the
lowest-validation-loss model; `weight_decay` can be increased if the training
and validation losses continue to diverge.
The VAE is unsupervised, so identify El Nino and La Nina by comparing the
latent trajectory with an external ENSO index such as NOAA ONI rather than
assuming that either latent coordinate has a fixed physical meaning.
Set the same `seed` on each machine before constructing the model. This makes
the Python, NumPy, PyTorch, CUDA, and data-loader randomness repeatable.
PyTorch can still produce small differences across different hardware,
driver, or library versions.