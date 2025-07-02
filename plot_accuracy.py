#!/usr/bin/env python3
"""
plot_accuracy.py
================

Quick start::

    # Plot every CSV into **one** combined figure
    python plot_accuracy.py results/ -o plots

Filename pattern ("a" & "b" are optional)::

    acc_list_metric={metric}[_a={a}_b={b}]_model={model}_{disease}_{optional_suffix}.csv

* Each CSV must contain **one column** of floating‑point accuracies; the row
  number is treated as the time step.
* The script collects **all** CSVs found in the provided paths (recursively
  with `--recursive` if requested) and draws them in a single line plot.
* **Legend rule** — when the model segment is literally `No`, the legend label
  is just the disease string (everything between `No_` and the suffix). For
  any other model value, the label follows: `<model> [| a=<a>,b=<b>] [| <suffix>]`.
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
})

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def load_series(path: Path) -> pd.Series:
    """Load a single‑column CSV and return it as a ``pd.Series`` of floats."""
    df = pd.read_csv(path, header=None)
    if df.shape[1] != 1:
        raise ValueError(f"Expected single column in {path} (got {df.shape[1]} cols).")
    series = df.iloc[:, 0].astype(float)
    series.index.name = "time"
    series.name = "accuracy"
    return series


def make_label(meta: dict[str, str | None]) -> str:
    """Return the legend label string according to the user‑defined rule."""
    # If model is literally "No", use only the disease string
    return str(meta["disease"]).replace("_", " ")


def plot_all(
    entries: list[tuple[dict[str, str | None], pd.Series]],
    outdir: Path,
    stem: str = "accuracy_combined",
    dark: bool = False,
) -> None:
    """
    Plot multiple accuracy curves.  When two final points overlap in y,
    only annotate the curve whose final x is largest (right-most).
    """
    if not entries:
        raise ValueError("No series to plot.")

    plt.style.use("seaborn-v0_8-dark" if dark else "seaborn-v0_8-paper")

    fig, ax = plt.subplots()
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)

    ax.tick_params(left=False, bottom=False)
    annotated: dict[float, tuple[float, str]] = {}  # y → (x, text)
    y_tol = 0.02  # “too close” threshold in data units

    # ------------------------------------------------------------------
    # 1.  Draw all curves first, remembering their last points.
    #     Sort DESC by length so the right-most curve is handled first.
    # ------------------------------------------------------------------
    last_pts: list[tuple[dict[str, str | None], float, float]] = []
    for meta, series in sorted(entries, key=lambda e: len(e[1]), reverse=True):
        ax.plot(series.index, series.values, marker=".", label=make_label(meta))
        last_pts.append((meta, series.index[-1], series.values[-1]))

    # ------------------------------------------------------------------
    # 2.  Annotate, preferring the right-most (already ordered that way).
    # ------------------------------------------------------------------
    for meta, last_x, last_y in last_pts:
        if any(abs(last_y - y0) < y_tol for y0 in annotated):
            continue  # a longer series at ~this y already claimed the label

        offset = (4, 0)  # 4 px to the right of the point
        ax.annotate(f"{last_y:.0%}",
                    (last_x, last_y),
                    xytext=offset,
                    textcoords="offset points",
                    va="center", ha="left",
                    fontsize=8)
        annotated[last_y] = (last_x, f"{last_y:.0%}")

    ax.set_title("MIMIC-CXR Label Accuracy per Retrieval (Ours Prototypical)")
    ax.set_xlabel("Number of Retrievals")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=False, ncol=2, loc=4)
    fig.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{stem}.{ext}")
    plt.close(fig)



# Fallback in case you don’t already have one.
def make_label(meta: dict[str, str | None]) -> str:
    """Simple label builder: use the disease name, plus any extra info."""
    disease = meta.get("disease", "unknown")
    extra   = meta.get("extra", "")
    return f"{disease} {extra}".strip()

# ------------------------------------------------------------------

def build_entries(df: pd.DataFrame) -> List[Tuple[dict[str, str | None], pd.Series]]:
    """
    Convert every column of `df` into the (meta, series) tuples
    that `plot_all` expects.
    """
    entries: List[Tuple[dict[str, str | None], pd.Series]] = []

    for col in df.columns:
        # Drop NaNs so ragged columns are OK.
        series = df[col].dropna()

        # Re-index 1, 2, … so the x-axis is “retrieval #”.
        series.index = range(1, len(series) + 1)

        meta = {"disease": col}
        entries.append((meta, series))

    return entries


def main(csv_path: Path, outdir: Path, stem: str, dark: bool) -> None:
    df = pd.read_csv(csv_path)
    entries = build_entries(df)

    # Hand off to your existing plotting helper.
    plot_all(entries, outdir, stem=stem, dark=dark)
    print(f"✓ Plots written to {outdir / (stem + '.png')} and .pdf")


if __name__ == "__main__":
    main(Path("/vol/ideadata/ce90tate/knowledge_graph_distance/retrieved_final/acc_list_metric=sym_tversky_a=0_b=1_model=No_Atelectasis_Cardiomegaly_Consolidation_Edema_No Finding_Pleural Effusion_Pneumonia_Pneumothorax_all_anomalies.csv"),
         Path("/vol/ideadata/ce90tate/knowledge_graph_distance/plots/"),
         "tverskyfinal",
         False)
