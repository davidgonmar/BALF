from matplotlib.lines import Line2D
import matplotlib.pyplot as plt


DPI = 300

FIGSIZE_MAIN = (3.6, 2.55)
FIGSIZE_WIDE = (4.8, 2.8)
FIGSIZE_APPENDIX = (4.8, 3.0)

COLORS = {
    "blue": "#2B6EA6",
    "orange": "#E07821",
    "green": "#3A9142",
    "red": "#C43E3E",
    "purple": "#7B61A8",
    "brown": "#8A5A44",
    "teal": "#2F8F8C",
    "gray": "0.25",
    "light_gray": "0.88",
    "black": "0.05",
}

METHOD_COLORS = {
    "balf_f": COLORS["blue"],
    "balf_p": COLORS["orange"],
    "energy": COLORS["green"],
    "energy_aa": COLORS["purple"],
    "uniform": COLORS["red"],
    "uniform_aa": COLORS["brown"],
}

RATIO_COLORS = [
    COLORS["blue"],
    COLORS["orange"],
    COLORS["green"],
    COLORS["red"],
    COLORS["purple"],
    COLORS["brown"],
    COLORS["teal"],
]


def apply_paper_style():
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 6.7,
            "legend.title_fontsize": 6.7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1.25,
            "lines.markersize": 3.0,
            "figure.dpi": DPI,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axes(ax, grid_axis="y"):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis=grid_axis, linestyle="-", linewidth=0.45, color=COLORS["light_gray"])
    ax.tick_params(axis="both", length=3, width=0.8)


def save_pdf(fig, path):
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.02)


def paper_line_kwargs(color, marker=None, linestyle="-", alpha=1.0):
    return {
        "color": color,
        "marker": marker,
        "linestyle": linestyle,
        "markerfacecolor": "none",
        "markeredgewidth": 0.8,
        "alpha": alpha,
        "zorder": 3,
    }


def line_handle(color, label, marker=None, linestyle="-", linewidth=1.25):
    return Line2D(
        [0],
        [0],
        color=color,
        marker=marker,
        linestyle=linestyle,
        markerfacecolor="none",
        markeredgewidth=0.8,
        linewidth=linewidth,
        label=label,
    )
