"""Cetagostini visual identity for Quarto articles.

Usage
-----
In a notebook cell::

    from cetagostini.style import setup_notebook, COLORS, PALETTE, make_rng

    seed, rng = make_rng("my article title")

The ``setup_notebook()`` call configures ArviZ, matplotlib rcParams,
retina rendering, and autoreload in one shot.  ``COLORS`` and ``PALETTE``
are dicts/lists ready for plotting.  ``make_rng(phrase)`` returns a
deterministic ``(seed, rng)`` pair from a human-readable phrase.
"""

from __future__ import annotations

import warnings
from typing import Tuple

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Color palette ──────────────────────────────────────────────────

COLORS: dict[str, str] = {
    "primary": "#778873",
    "secondary": "#A1BC98",
    "accent": "#DCCFC0",
    "bg": "#FDF6ED",
    "ink": "#2B2A26",
    "ink_muted": "#6B665C",
    "green_strong": "#4F6B4A",
    "brown": "#6B5A48",
    "line": "#E6DFD2",
    "surface_alt": "#F2EDE3",
}

PALETTE: list[str] = [
    "#778873", "#A1BC98", "#DCCFC0", "#4F6B4A", "#6B5A48", "#6B665C",
]

# ── rcParams ───────────────────────────────────────────────────────

_RCPARAMS: dict = {
    "figure.facecolor": COLORS["bg"],
    "axes.facecolor": COLORS["bg"],
    "axes.edgecolor": COLORS["line"],
    "axes.labelcolor": COLORS["ink"],
    "text.color": COLORS["ink"],
    "xtick.color": COLORS["ink_muted"],
    "ytick.color": COLORS["ink_muted"],
    "grid.color": COLORS["line"],
    "grid.alpha": 0.6,
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Manrope", "Helvetica Neue", "Arial"],
    "axes.titleweight": "semibold",
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.facecolor": COLORS["bg"],
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
    "figure.constrained_layout.use": True,
}

# ── Public helpers ─────────────────────────────────────────────────


def setup_notebook(
    *,
    figsize: tuple[int, int] = (8, 4),
    warnings_filter: str = "ignore",
) -> None:
    """Configure ArviZ, matplotlib, and IPython for a Quarto article.

    Call once in the first code cell of every notebook.
    """
    if warnings_filter:
        warnings.filterwarnings(warnings_filter)

    az.style.use("arviz-darkgrid")
    plt.rcParams["figure.figsize"] = list(figsize)
    mpl.rcParams.update(_RCPARAMS)

    # IPython magic — silently no-op outside Jupyter
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is not None:
            shell.run_line_magic("load_ext", "autoreload")
            shell.run_line_magic("autoreload", "2")
            shell.run_line_magic("config", "InlineBackend.figure_format = 'retina'")
    except ImportError:
        pass


def make_rng(phrase: str) -> Tuple[int, np.random.Generator]:
    """Return a deterministic ``(seed, rng)`` pair from a human-readable phrase.

    Parameters
    ----------
    phrase : str
        A descriptive string (usually the article title in lowercase).
        ``seed`` is computed as ``sum(map(ord, phrase))``.

    Returns
    -------
    seed : int
    rng  : numpy.random.Generator
    """
    seed: int = sum(map(ord, phrase))
    rng: np.random.Generator = np.random.default_rng(seed=seed)
    return seed, rng


def article_table(
    frame: pd.DataFrame,
    caption: str,
    formats: dict[str, str] | None = None,
):
    """Render a compact, left-aligned, index-free styled table.

    Parameters
    ----------
    frame : DataFrame
    caption : str
    formats : dict, optional
        Column-name → format-string mapping passed to ``Styler.format()``.
    """
    styled = (
        frame.style
        .hide(axis="index")
        .set_caption(caption)
        .set_properties(**{"text-align": "left"})
        .set_table_styles([
            {"selector": "th", "props": [("text-align", "left")]},
            {"selector": "td", "props": [("text-align", "left")]},
        ])
    )
    return styled.format(formats) if formats else styled
