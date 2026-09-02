"""Leakage-probing figure style: SciencePlots base, TK's semantic palette on top.

Layering (v2, 2026-08-31)
------------------------
`use()` applies `plt.style.use(["science", "no-latex"])` first — SciencePlots
supplies tick direction, minor ticks, sizing and legend defaults — then
overrides it with this project's palette, white ground and sans-serif face.
`no-latex` is mandatory: this repo must not acquire a TeX dependency.

Precedence, in order of increasing authority:
    matplotlib defaults  <  science + no-latex  <  the overrides below.

The palette is semantic and beats anything SciencePlots would pick:
warm white ground, slate ink, burnt-sienna reserved for "influenced/steered",
blue-grey for "neutral/no-bet", pale grey for spaghetti. Shared with
html/figstyle.css so the HTML schematic (F1) matches the plotted figures.

Spines stay open (top/right hidden) via `plate()`/`trim()` — that is this
project's house look and every figure script relies on it. Ticks point inward,
which is SciencePlots' contribution and is what the open-spine look wants.

Sizes are final printed sizes. Build at the size the figure will be shown at;
never build wide and shrink. `WIDE` (6.5in) is the LessWrong post column;
`SINGLE` (5.5x3.4) is the single-panel default.

Usage:
    import style
    style.use()
    fig, ax = style.plate(figsize=style.SINGLE)
    ...
    style.save(fig, "F2_final")     # writes svg + pdf + png, prints an audit line
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  (registers the "science" style with matplotlib)

# ---- palette (semantic; do not reassign these to new meanings) --------------
BG = "#ffffff"        # white (TK 2026-08-20: no warm paper tint on plots)
INK = "#2b3a4f"       # slate ink (matches html/figstyle.css --ink)
STEERED = "#b5552d"   # sienna — ONLY for the swapped/influenced arm (--sie)
NEUTRAL = "#6f7f92"   # blue-grey — no-bet/neutral (--neu)
FAINT = "#b9bcc0"     # spaghetti / de-emphasized (--faint)
FAINT_STEERED = "#d9a88f"   # washed sienna for steered spaghetti
GRID = "#e3e3e3"

SERIF = ["STIXGeneral", "DejaVu Serif", "Georgia", "serif"]
SANS = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans", "sans-serif"]

# ---- final printed sizes (inches) ------------------------------------------
WIDE = 6.5            # LessWrong post column, maximum usable width
SINGLE = (5.5, 3.4)   # single-panel default
TEXT_PT_FLOOR = 5.0   # anything smaller than this is unreadable in print


def use():
    """SciencePlots base, then this project's overrides."""
    plt.style.use(["science", "no-latex"])
    mpl.rcParams.update({
        # --- ground and ink (override SciencePlots) ---
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        # --- sans-serif face (SciencePlots defaults to a serif) ---
        "font.family": "sans-serif",
        "font.sans-serif": SANS,
        # dejavusans, not a custom Helvetica map: SciencePlots turns on
        # axes.formatter.use_mathtext, which sends TICK LABELS through the math
        # font. A custom fontset cannot resolve a weight there and matplotlib
        # logs "Failed to find font weight normal" on every render. DejaVu Sans
        # is a complete sans math set that is always present.
        "mathtext.fontset": "dejavusans",
        # --- sizes, tuned for a 6.5in web column read on screen ---
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        # --- open spines: house look, reinforced per-axes by trim() ---
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.top": False,
        "ytick.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "axes.grid": False,
        "legend.frameon": False,
        # --- vector output ---
        "svg.fonttype": "path",   # glyphs as paths: identical in browsers lacking STIX
        "pdf.fonttype": 42,       # embed TrueType, not Type 3 — required by most venues
        "ps.fonttype": 42,
        "figure.dpi": 130,
    })


def plate(figsize=SINGLE, **kw):
    """Figure+axes with the plate look (spines trimmed, ink on paper)."""
    fig, ax = plt.subplots(figsize=figsize, **kw)
    trim(ax)
    return fig, ax


def trim(ax):
    """Hide top/right spines. Tick direction is left to rcParams (SciencePlots: in)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def annotate(ax, text, xy, xytext, color=INK, fontsize=8, curve=0.18, **kw):
    """Direct on-figure annotation with a light curved arrow, no boxes."""
    ax.annotate(
        text, xy=xy, xytext=xytext,
        fontsize=fontsize, color=color,
        arrowprops=dict(
            arrowstyle="-", lw=0.6, color=color, shrinkA=2, shrinkB=3,
            connectionstyle=f"arc3,rad={curve}",
        ),
        **kw,
    )


def label_halfplane(ax, text, x, y, color=INK, fontsize=8, style="italic"):
    ax.text(x, y, text, fontsize=fontsize, color=color, style=style,
            ha="center", va="center")


def no_minor(ax, axis="y"):
    """Kill minor ticks on a categorical axis.

    SciencePlots turns minor ticks on globally, which is right for a continuous
    axis and wrong for one whose ticks are category names: AutoMinorLocator
    interpolates between the categories and draws marks at positions that mean
    nothing.
    """
    from matplotlib.ticker import NullLocator
    (ax.yaxis if axis == "y" else ax.xaxis).set_minor_locator(NullLocator())
    return ax


def panel_label(ax, letter, x=-0.02, y=1.04):
    """(a)/(b) panel letter at the top-left of an axes, in axes coordinates."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", color=INK, ha="right", va="bottom")


# ---- output -----------------------------------------------------------------

def save(fig, name, outdir="out", grey=True):
    """Write svg + pdf + png(2x), plus a greyscale proof, and print an audit line."""
    paths = {}
    for ext in ("svg", "pdf", "png"):
        os.makedirs(f"{outdir}/{ext}", exist_ok=True)
        p = f"{outdir}/{ext}/{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.12,
                    dpi=200 if ext == "png" else None)
        paths[ext] = p
    if grey:
        _greyscale_proof(paths["png"], f"{outdir}/grey/{name}.png")
    print(f"saved {name}: " + " ".join(paths.values()))
    print("  audit: " + audit(fig))
    return paths


def _greyscale_proof(png_in, png_out):
    """Luminance render of the PNG, so contrast can be checked without colour."""
    from PIL import Image
    os.makedirs(os.path.dirname(png_out), exist_ok=True)
    Image.open(png_in).convert("L").save(png_out)


def audit(fig):
    """Report printed width, smallest text, and any text below the legibility floor."""
    fig.canvas.draw()
    w, h = fig.get_size_inches()
    sizes = [t.get_fontsize() for t in _all_text(fig) if t.get_text().strip()]
    problems = []
    if w > WIDE + 0.01:
        problems.append(f"width {w:.2f}in exceeds the {WIDE}in column")
    small = [s for s in sizes if s < TEXT_PT_FLOOR]
    if small:
        problems.append(f"{len(small)} text objects below {TEXT_PT_FLOOR}pt")
    status = "clean" if not problems else "; ".join(problems)
    return f"{w:.2f}x{h:.2f}in, min text {min(sizes):.1f}pt, {status}"


def _all_text(fig):
    out = list(fig.texts)
    for ax in fig.axes:
        out += [ax.title, ax.xaxis.label, ax.yaxis.label]
        out += ax.get_xticklabels() + ax.get_yticklabels() + list(ax.texts)
        leg = ax.get_legend()
        if leg is not None:
            out += leg.get_texts()
    return out


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
