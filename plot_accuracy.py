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

# -----------------------------------------------------------------------------
# Regex to parse filenames
# -----------------------------------------------------------------------------
FILENAME_RE = re.compile(
    r"acc_list_metric=(?P<metric>[^_]+(?:_[^_]+)*?)"        # metric (underscores ok)
    r"(?:_a=(?P<a>[^_]+)_b=(?P<b>[^_]+))?"                 # optional a & b
    r"_model=(?P<model>[^_]+(?:_[^_]+)*?)"                 # model (underscores ok)
    r"_(?P<disease>[^_]+)"                                 # disease
    r"(?:_(?P<suffix>[^.]+))?"                             # optional suffix
    r"\.csv$"
)

mpl.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
})

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def parse_filename(path: Path) -> dict[str, str | None]:
    """Extract metadata from *path* according to ``FILENAME_RE``."""
    m = FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Filename '{path.name}' does not match required pattern.")
    info = m.groupdict()
    info.setdefault("a", None)
    info.setdefault("b", None)
    return info


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

# -----------------------------------------------------------------------------
# File discovery
# -----------------------------------------------------------------------------

def discover_csvs(paths: Iterable[Path], recursive: bool) -> List[Path]:
    csvs: list[Path] = []
    for p in paths:
        if any(ch in str(p) for ch in "*?["):
            csvs.extend(p.parent.glob(p.name))
            continue
        if p.is_dir():
            csvs.extend(p.rglob("*.csv") if recursive else p.glob("*.csv"))
        elif p.is_file():
            csvs.append(p)
        else:
            print(f"⚠️ Warning: {p} not found", file=sys.stderr)
    return sorted(csvs)

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_all(
    entries: List[Tuple[dict[str, str | None], pd.Series]],
    outdir: Path,
    stem: str = "accuracy_combined",
    dark: bool = False,
) -> None:
    if not entries:
        raise ValueError("No series to plot.")

    plt.style.use("seaborn-v0_8-dark" if dark else "seaborn-v0_8-paper")

    fig, ax = plt.subplots()
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)

    ax.tick_params(left=False, bottom=False)
    annotated_y = []
    y_tol = 0.02  # “too close” threshold (in data units, tweak as needed)
    # ------------------------------------------

    for meta, series in entries:
        ax.plot(series.index, series.values, marker=".", label=make_label(meta))

        # last point
        last_x = series.index[-1]
        last_y = series.values[-1]

        # decide offset: default → right; if too close → left
        offset = (4, 0)  # 4 px to the right
        if any(abs(last_y - y0) < y_tol for y0 in annotated_y):
            continue

        ax.annotate(f"{last_y:.0%}",
                    (last_x, last_y),
                    xytext=offset,
                    textcoords="offset points",
                    va="center", ha="left" if offset[0] > 0 else "right",
                    fontsize=8)

        annotated_y.append(last_y)  # remember position for next curves

    ax.set_title("MIMIC-CXR Label Accuracy per Retrieval (Jaccard)")
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

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            Combine accuracy‑over‑time curves from CSV files into one plot.

            Positional *paths* may be directories, individual CSV files, or globs. All
            matching CSVs are plotted in a single figure. Use `--recursive` to search
            sub‑folders.
            """
        ),
    )

    parser.add_argument("paths", nargs="+", type=Path, help="Input directories/files/globs.")
    parser.add_argument("-o", "--outdir", type=Path, default=Path("plots"), help="Output directory.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search directories.")
    parser.add_argument("--dark", action="store_true", help="Use dark theme for the plot.")
    return parser

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    csv_paths = discover_csvs(args.paths, recursive=args.recursive)
    if not csv_paths:
        parser.error("No CSV files found matching provided paths.")

    entries: List[Tuple[dict[str, str | None], pd.Series]] = []
    for path in csv_paths:
        try:
            meta = parse_filename(path)
            series = load_series(path)
        except ValueError as e:
            print(f"⚠️ Skipping {path}: {e}", file=sys.stderr)
            continue
        entries.append((meta, series))

    if not entries:
        parser.error("No valid CSVs matched the naming pattern.")

    plot_all(entries, args.outdir, dark=args.dark)
    print(f"✅ Combined plot saved to '{args.outdir.resolve()}'")


if __name__ == "__main__":
    main()
