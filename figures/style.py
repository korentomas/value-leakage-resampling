"""Leakage-probing figure style: sans-serif, white ground, three world colours (shared with html/figstyle.css).

Palette extracted from the approved moodboard
(~/Downloads/Gemini_Generated_Image_k98c48k98c48k98c.jpg):
warm off-white ground, near-black ink, burnt-sienna reserved for
"influenced/steered", blue-grey for "neutral", pale grey for spaghetti.

Usage in a notebook cell:
    import style
    style.use()
    fig, ax = style.plate(figsize=(7, 4.2))
    ...
    style.annotate(ax, "Betley et al.'s published 0.62", xy=(0, .62), xytext=(0.12, .8))
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

BG = "#ffffff"        # white (TK 2026-08-20: no warm paper tint on plots)
INK = "#2b3a4f"       # slate ink (matches html/figstyle.css --ink)
STEERED = "#b5552d"   # sienna — ONLY for the swapped/influenced arm (--sie)
NEUTRAL = "#6f7f92"   # blue-grey — no-bet/neutral (--neu)
FAINT = "#b9bcc0"     # spaghetti / de-emphasized (--faint)
FAINT_STEERED = "#d9a88f"   # washed sienna for steered spaghetti
GRID = "#e3e3e3"

SERIF = ["STIXGeneral", "DejaVu Serif", "Georgia", "serif"]
SANS = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans", "sans-serif"]


def use():
    mpl.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "font.family": "sans-serif",
        "font.sans-serif": SANS,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Helvetica Neue", "mathtext.it": "Helvetica Neue:italic", "mathtext.bf": "Helvetica Neue:bold",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.grid": False,
        "legend.frameon": False,
        "svg.fonttype": "path",          # glyphs as paths: renders identically everywhere (text-as-text clipped in browsers lacking STIX)
        "figure.dpi": 130,
    })


def plate(figsize=(7, 4.2), **kw):
    """Figure+axes with the plate look (spines trimmed, ink on paper)."""
    fig, ax = plt.subplots(figsize=figsize, **kw)
    trim(ax)
    return fig, ax


def trim(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    return ax


def annotate(ax, text, xy, xytext, color=INK, fontsize=10, curve=0.18):
    """Direct on-figure annotation with a light curved arrow, no boxes."""
    ax.annotate(
        text, xy=xy, xytext=xytext,
        fontsize=fontsize, color=color,
        arrowprops=dict(
            arrowstyle="-", lw=0.7, color=color, shrinkA=2, shrinkB=3,
            connectionstyle=f"arc3,rad={curve}",
        ),
    )


def label_halfplane(ax, text, x, y, color=INK, fontsize=10, style="italic"):
    ax.text(x, y, text, fontsize=fontsize, color=color, style=style,
            ha="center", va="center")


def save(fig, name, outdir="out"):
    import os
    for ext in ("svg", "png"):
        os.makedirs(f"{outdir}/{ext}", exist_ok=True)
        fig.savefig(f"{outdir}/{ext}/{name}.{ext}", bbox_inches="tight",
                    pad_inches=0.15, dpi=450 if ext == "png" else None)
    print(f"saved {outdir}/svg/{name}.svg + {outdir}/png/{name}.png")


def tangle(ax, cx, cy, r, color, n=14, seed=0, lw=1.0, alpha=0.9):
    """Decorative reasoning-tangle (for schematic panels), smooth random loops."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        t = np.linspace(0, 2 * np.pi, 200)
        a, b = rng.uniform(0.3, 1.0, 2) * r
        ph = rng.uniform(0, 2 * np.pi, 3)
        f1, f2 = rng.integers(1, 4), rng.integers(2, 6)
        x = cx + a * np.cos(f1 * t + ph[0]) + 0.25 * r * np.sin(f2 * t + ph[1])
        y = cy + b * np.sin(f1 * t + ph[2]) + 0.25 * r * np.cos(f2 * t + ph[0])
        ax.plot(x, y, color=color, lw=lw, alpha=alpha, solid_capstyle="round")
