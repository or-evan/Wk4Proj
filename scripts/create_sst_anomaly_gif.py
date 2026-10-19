"""Create clean Pacific SST anomaly animations for PowerPoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from wk4proj.convvae_training import prepare_sst


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "data.grib"
GIF_PATH = ROOT / "pacific_sst_anomaly.gif"
VIDEO_PATH = ROOT / "pacific_sst_anomaly.mp4"
ANIMATION_START = np.datetime64("2010-01")
ANIMATION_END = np.datetime64("2017-01")
FPS = 12
ANOMALY_LIMIT = 2.2
GIF_DPI = 80
VIDEO_DPI = 120

BACKGROUND = "#ffffff"
LAND = "#eef2f5"
GRID = "#9fb0bf"
TEXT = "#15263a"

FIGURE_WIDTH = 16
FIGURE_HEIGHT = 9
MAP_TOP = 0.925
MAP_BOTTOM = 0.115
MAP_HEIGHT = MAP_TOP - MAP_BOTTOM


def coordinate_extent(coordinates: np.ndarray) -> tuple[float, float]:
    """Return the outer cell edges of a regular, possibly descending coordinate."""
    step = float(np.diff(coordinates).mean())
    lower = float(coordinates.min()) - abs(step) / 2
    upper = float(coordinates.max()) + abs(step) / 2
    return lower, upper


def build_animations(input_path: Path, gif_path: Path, video_path: Path) -> None:
    """Render the 2010-2016 Pacific SST anomaly sequence with date and legend."""
    if not input_path.is_file():
        raise FileNotFoundError(f"SST data file not found: {input_path}")
    if not FFMpegWriter.isAvailable():
        raise RuntimeError("FFmpeg is required to create the PowerPoint MP4 video.")

    with xr.open_dataset(
        input_path, engine="cfgrib", backend_kwargs={"indexpath": ""}
    ) as dataset:
        anomaly = prepare_sst(dataset).load()

    values = np.asarray(anomaly.values, dtype=np.float32)
    dates = anomaly.time.values
    longitude = anomaly.longitude.values
    latitude = anomaly.latitude.values
    frames = np.flatnonzero((dates >= ANIMATION_START) & (dates < ANIMATION_END))
    if values.ndim != 3 or frames.size == 0:
        raise ValueError(
            "Expected SST anomaly data with monthly samples for 2010 through 2016."
        )

    lon_min, lon_max = coordinate_extent(longitude)
    lat_min, lat_max = coordinate_extent(latitude)
    data_aspect = (lon_max - lon_min) / (lat_max - lat_min)
    map_width = MAP_HEIGHT * FIGURE_HEIGHT * data_aspect / FIGURE_WIDTH
    map_left = (1 - map_width) / 2

    colormap = LinearSegmentedColormap.from_list(
        "ssta",
        ["#2455a4", "#69b9d0", "#d9edf0", "#fffdf5", "#f5ae78", "#b63c45"],
    )
    figure = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), facecolor=BACKGROUND)
    map_ax = figure.add_axes(
        [map_left, MAP_BOTTOM, map_width, MAP_HEIGHT], facecolor=LAND
    )
    mesh = map_ax.pcolormesh(
        longitude,
        latitude,
        values[frames[0]],
        shading="auto",
        cmap=colormap,
        vmin=-ANOMALY_LIMIT,
        vmax=ANOMALY_LIMIT,
        rasterized=True,
    )
    map_ax.set_xlim(lon_min, lon_max)
    map_ax.set_ylim(lat_min, lat_max)
    map_ax.set_aspect("equal", adjustable="box")
    map_ax.vlines(
        [120, 160, 200, 240, 280],
        lat_min,
        lat_max,
        colors=GRID,
        linewidth=0.45,
        alpha=0.35,
        zorder=2,
    )
    map_ax.hlines(
        [-60, -30, 0, 30, 60],
        lon_min,
        lon_max,
        colors=GRID,
        linewidth=0.45,
        alpha=0.35,
        zorder=2,
    )
    map_ax.set_axis_off()

    date_label = figure.text(
        map_left,
        0.985,
        "",
        color=TEXT,
        fontsize=24,
        fontweight="bold",
        ha="left",
        va="top",
    )

    colorbar_ax = figure.add_axes([0.35, 0.045, 0.30, 0.016])
    colorbar = figure.colorbar(
        mesh,
        cax=colorbar_ax,
        orientation="horizontal",
        ticks=[-2, -1, 0, 1, 2],
    )
    colorbar.ax.tick_params(colors=TEXT, labelsize=9, length=0, pad=5)
    colorbar.outline.set_visible(False)
    figure.text(
        0.665,
        0.053,
        "°C",
        color=TEXT,
        fontsize=9,
        ha="left",
        va="center",
    )

    def update(frame: int) -> tuple[object, ...]:
        mesh.set_array(values[frame].ravel())
        date_label.set_text(np.datetime_as_string(dates[frame], unit="M"))
        return mesh, date_label

    animation = FuncAnimation(
        figure, update, frames=frames, interval=1000 / FPS, blit=False, repeat=True
    )
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(gif_path, writer=PillowWriter(fps=FPS), dpi=GIF_DPI)
    compress_gif(gif_path, colormap)
    animation.save(
        video_path,
        writer=FFMpegWriter(
            fps=FPS,
            codec="libx264",
            bitrate=-1,
            extra_args=["-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        ),
        dpi=VIDEO_DPI,
    )
    plt.close(figure)
    print(f"Saved {gif_path} and {video_path} ({len(frames)} frames, 2010-2016)")


def compress_gif(path: Path, colormap: LinearSegmentedColormap) -> None:
    """Use one shared palette to keep the high-resolution GIF compact."""
    with Image.open(path) as source:
        frame_count = source.n_frames
        gradient_colors = np.rint(
            colormap(np.linspace(0, 1, 224))[:, :3] * 255
        ).astype(np.uint8)
        interface_colors = np.array(
            [(255, 255, 255), (238, 242, 245), (159, 176, 191), (21, 38, 58)],
            dtype=np.uint8,
        )
        palette_colors = np.concatenate((gradient_colors, interface_colors))
        palette_image = Image.new("P", (1, 1))
        palette_image.putpalette(
            [int(value) for value in palette_colors.ravel()]
            + [0] * (768 - palette_colors.size)
        )
        frames = []
        durations = []
        for index in range(frame_count):
            source.seek(index)
            frame = source.convert("RGB")
            frames.append(
                frame.quantize(palette=palette_image, dither=Image.Dither.NONE)
            )
            durations.append(source.info.get("duration", round(1000 / FPS)))

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH, help="Input GRIB file")
    parser.add_argument(
        "--gif-output", type=Path, default=GIF_PATH, help="Output GIF file"
    )
    parser.add_argument(
        "--video-output", type=Path, default=VIDEO_PATH, help="Output MP4 file"
    )
    args = parser.parse_args()
    build_animations(args.input, args.gif_output, args.video_output)


if __name__ == "__main__":
    main()
