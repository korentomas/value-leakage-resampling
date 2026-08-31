"""Figure styling for the paper set, modelled on Betley et al.'s shared/plot_style.py.

The blog figures in style.py are heavily styled: bespoke palette, open spines,
no gridlines, prose annotations with curved leader arrows. None of that appears
in the papers this work sits next to. Betley et al.'s own style module sets
font sizes and nothing else, leaving matplotlib's defaults in place; Bigelow et
al. publish no plotting code at all, and their figures are plain stacked panels
with the explanation carried entirely by the caption.

So this module deliberately does almost nothing. It resets to matplotlib
defaults and sets font sizes. Boxed spines, DejaVu Sans and the tab10 cycle are
all inherited on purpose.

No gridlines. Betley et al. put a y-grid on essentially every axes, but neither
Bigelow paper nor Thought Branches uses one, and this work is positioned against
the forking-paths line, so the plots follow those.

Arm colours are pinned here so the F1 schematic and the plots agree. Green and
red are avoided, following Betley et al., because on this task those hues read
as the good and bad cause.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

TICK_FS = 9
AXIS_LABEL_FS = 10
TITLE_FS = 10
SUPTITLE_FS = 11
LEGEND_FS = 9
ANNOT_FS = 8

# tab10, pinned by arm. Shared with html/figstyle.css for F1.
ORIG = "#1f77b4"     # tab10 blue
NEUTRAL = "#7f7f7f"  # tab10 grey
SWAP = "#ff7f0e"     # tab10 orange
REF = "#d62728"      # tab10 red, reference lines only (Betley use red dashed)

WIDE = (5.5, 2.6)     # full text width of the ICLR single-column layout
HALF = (2.7, 2.4)     # side-by-side pair inside that width


def use():
    mpl.rcdefaults()
    mpl.rcParams.update({
        "xtick.labelsize": TICK_FS,
        "ytick.labelsize": TICK_FS,
        "axes.labelsize": AXIS_LABEL_FS,
        "axes.titlesize": TITLE_FS,
        "figure.titlesize": SUPTITLE_FS,
        "legend.fontsize": LEGEND_FS,
        "figure.dpi": 130,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def below_legend(fig, ax, ncol=3, y=-0.02, title=None):
    """Legend under the axes, which is what both Bigelow papers and Thought
    Branches do. Keeps the plotting area for data."""
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
               frameon=False, title=title)
    return fig


def save(fig, name, outdir="out/paper"):
    """PDF for the paper, PNG alongside it.

    Betley et al. write PDF only, straight into Overleaf. We need the PNG too:
    the same figures are used in the LessWrong write-up, and neither GitHub
    markdown nor LessWrong renders a PDF inline.
    """
    import os
    os.makedirs(outdir, exist_ok=True)
    p = f"{outdir}/{name}.pdf"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(f"{outdir}/{name}.png", bbox_inches="tight", dpi=200)
    w, h = fig.get_size_inches()
    print(f"saved {p}  ({w:.2f}x{h:.2f} in)")
    plt.close(fig)
    return p
