#!/usr/bin/env python3

# Create the three speaker-cohort 3D t-SNE RTF landscapes from the screen log.

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT / "Temp/synthetic_capacity_race_20260910_v2/progress.jsonl"
sys.path.insert(0, str(PROJECT / "Plots"))
import theme  # noqa: E402

DEFAULT_OUTPUT = theme.OUTPUTS


# Read one JSON record per completed screen measurement.
def load_records(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


# Keep one screen value for each unique thread allocation.
def load_points(path: Path, speakers: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    fields = ["lseend", "campplus", "asr"] if speakers == 1 else [
        "lseend", "separation", "campplus", "asr"
    ]
    values_by_configuration = {}
    for record in load_records(path):
        if record.get("scenario") != speakers or record.get("phase", "").startswith("validation"):
            continue
        report = record.get("report")
        if not report or record.get("returncode") not in (None, 0):
            continue
        configuration = record["configuration"]
        key = tuple(configuration[field] for field in fields)
        values_by_configuration[key] = report["pipeline_rtf"]
    if len(values_by_configuration) < 10:
        raise ValueError(f"Only {len(values_by_configuration)} {speakers}-speaker points found")
    configurations = np.asarray(list(values_by_configuration), dtype=float)
    rtf = np.asarray(list(values_by_configuration.values()), dtype=float)
    return configurations, rtf, fields


# Draw one cohort's t-SNE points, interpolated surface, and lowest-RTF star.
def plot_cohort(data_path: Path, output_dir: Path, speakers: int, seed: int) -> Path:
    configurations, rtf, fields = load_points(data_path, speakers)
    embedded = TSNE(
        n_components=2,
        perplexity=min(8.0, len(rtf) - 1.0),
        init="pca",
        learning_rate="auto",
        max_iter=2000,
        random_state=seed,
    ).fit_transform(StandardScaler().fit_transform(configurations))
    x, y = embedded[:, 0], embedded[:, 1]
    winner = int(np.argmin(rtf))

    grid_x, grid_y = np.meshgrid(
        np.linspace(x.min(), x.max(), 120),
        np.linspace(y.min(), y.max(), 120),
    )
    measured_xy = np.column_stack((x, y))
    grid_xy = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    surface = RBFInterpolator(
        measured_xy, rtf, kernel="thin_plate_spline", smoothing=10.0
    )(grid_xy).reshape(grid_x.shape)
    inside = Delaunay(measured_xy).find_simplex(grid_xy).reshape(grid_x.shape) >= 0
    surface[~inside] = np.nan

    normalizer = Normalize(vmin=float(rtf.min()), vmax=float(rtf.max()))
    colour_map = plt.get_cmap("viridis_r")
    shaded = LightSource(azdeg=315, altdeg=38).shade(
        np.ma.masked_invalid(surface).filled(float(np.nanmean(rtf))),
        cmap=colour_map,
        norm=normalizer,
        vert_exag=2.0,
        blend_mode="soft",
    )
    shaded[..., 3] = np.where(np.isfinite(surface), 0.72, 0.0)

    theme.apply()
    views = ((24, -58), (31, 35), (62, -112))
    figure = plt.figure(figsize=(18.5, 6.2))
    for index, (elevation, azimuth) in enumerate(views, start=1):
        axis = figure.add_subplot(1, 3, index, projection="3d")
        axis.plot_surface(
            grid_x, grid_y, surface, facecolors=shaded, linewidth=0,
            antialiased=True, shade=False, rcount=120, ccount=120,
        )
        axis.scatter(
            x, y, rtf, c=rtf, cmap=colour_map, norm=normalizer,
            s=48, edgecolor="black", linewidth=0.45, depthshade=False,
        )
        axis.scatter(
            [x[winner]], [y[winner]], [rtf[winner]], marker="*", s=280,
            facecolor=theme.NEUTRAL, edgecolor="black", linewidth=0.9,
            depthshade=False,
        )
        axis.set_xlabel("t-SNE dimension 1", labelpad=8, fontsize=11)
        axis.set_ylabel("t-SNE dimension 2", labelpad=8, fontsize=11)
        axis.set_zlabel("RTF (lower is better)", labelpad=9, fontsize=11)
        axis.set_title(f"Elevation {elevation}°, azimuth {azimuth}°", fontsize=13)
        axis.view_init(elev=elevation, azim=azimuth)
        axis.tick_params(labelsize=9)
        axis.grid(True, linewidth=0.4, alpha=0.45)

    # Give the shared scale its own fixed axis.  Letting Matplotlib infer a
    # colourbar position from three 3D axes places it over the final view.
    colour_axis = figure.add_axes((0.895, 0.21, 0.012, 0.56))
    figure.colorbar(
        ScalarMappable(norm=normalizer, cmap=colour_map), cax=colour_axis,
    ).set_label("RTF")
    labels = {
        "lseend": "LS-EEND", "separation": "ConvTasNet",
        "campplus": "CAM++", "asr": "ASR",
    }
    allocation = ", ".join(
        f"{labels[field]}={int(value)}"
        for field, value in zip(fields, configurations[winner])
    )
    figure.suptitle(
        f"{speakers} speaker t-SNE RTF landscape ({len(rtf)} configurations)",
        fontsize=18, y=0.98,
    )
    figure.text(
        0.5, 0.035,
        f"Star: lowest measured RTF ({allocation}, RTF={rtf[winner]:.3f}). "
        "Surface: interpolation inside the measured convex hull.",
        ha="center", fontsize=11,
    )
    figure.subplots_adjust(left=0.01, right=0.87, top=0.90, bottom=0.10, wspace=0.02)

    output = output_dir / f"{speakers}spk_tsne.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    for speakers in (1, 2, 3):
        print(plot_cohort(args.data, args.output_dir, speakers, args.seed))


if __name__ == "__main__":
    main()
