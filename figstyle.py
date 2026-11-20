"""Shared figure style for Paper/ notebooks.

Call ``apply()`` once, then use the helpers so later figures match. Lithology
and continuous fields use curlew's colour ramp (``ramp()`` falls back to a
built-in copy, so this module needs only numpy and matplotlib), and maps keep
the full evaluation-grid extent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patheffects import withStroke
from matplotlib.transforms import ScaledTranslation

# Seaborn colorblind; used in order when a horizon name is not in the dict.
FORMATION_PALETTE = (
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
)
# Paul Tol bright palette, ordered for categorical horizons in section figures.
# The colours remain distinct under common colour-vision deficiencies and avoid
# the very pale yellow used by several default qualitative maps.
HORIZON_PALETTE = (
    "#4477AA",
    "#66CCEE",
    "#228833",
    "#CCBB44",
    "#EE6677",
    "#AA3377",
)
FORMATION_COLORS = {
    "rock1": FORMATION_PALETTE[0],
    "rock2": FORMATION_PALETTE[1],
}
NORMAL_COLOR = "0.2"
YOUNGING_LW = 1.1

# Figures are drawn at their final print size: Nature's double column is
# 180 mm and its single column 88 mm, with 5-7 pt text. Include them in LaTeX
# at natural size (no width scaling) so these sizes survive.
PRINT_WIDTH = 180.0 / 25.4
PRINT_WIDTH_SINGLE = 88.0 / 25.4

# Helvetica.ttc first face is Regular (unlike Avenir Next, which is Heavy).
_HELVETICA = Path("/System/Library/Fonts/Helvetica.ttc")
_ARIAL = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
_ARIAL_BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")

FONT: FontProperties | None = None
FONT_BOLD: FontProperties | None = None

# Nominal sizes in this module (10 for labels, 9 for ticks, 11 for titles)
# map to 7, 6.3 and 7.7 pt in print.
TEXT_SCALE = 0.7


def _text_size(points: float) -> float:
    """Apply the paper-wide text scale to a point size."""
    return float(points) * TEXT_SCALE


def pt(points: float) -> float:
    """Print size of a nominal point size, for ``fontsize=`` in notebooks."""
    return _text_size(points)


RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.weight": "normal",
    "font.size": _text_size(10),
    "axes.titlesize": _text_size(11),
    "axes.labelsize": _text_size(10),
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "axes.titlepad": 4,
    "axes.linewidth": 0.5,
    "axes.edgecolor": "0.25",
    "axes.labelcolor": "0.15",
    "axes.unicode_minus": True,
    "axes.formatter.useoffset": False,
    "xtick.labelsize": _text_size(9),
    "ytick.labelsize": _text_size(9),
    "xtick.color": "0.25",
    "ytick.color": "0.25",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": False,
    "ytick.right": False,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "legend.fontsize": _text_size(9),
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "legend.borderpad": 0.3,
    "legend.labelspacing": 0.35,
    "legend.columnspacing": 1.2,
    "legend.handletextpad": 0.45,
    "lines.linewidth": 1.0,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "stixsans",
    "mathtext.default": "it",
    "text.usetex": False,
}


def _helvetica_bold() -> Path | None:
    """Helvetica Bold as a standalone file, extracted from the system .ttc.

    Matplotlib reads only the first face of a font collection, which for
    Helvetica.ttc is Regular, so bold text would silently render Regular.
    """
    if not _HELVETICA.exists():
        return None
    import matplotlib

    out = Path(matplotlib.get_cachedir()) / "Helvetica-Bold.ttf"
    if not out.exists():
        try:
            from fontTools.ttLib import TTCollection

            for face in TTCollection(str(_HELVETICA)).fonts:
                if face["name"].getDebugName(4) == "Helvetica Bold":
                    face.save(str(out))
                    break
        except Exception:
            return None
    return out if out.exists() else None


def _register_fonts() -> str:
    for path in (_HELVETICA, _helvetica_bold(), _ARIAL, _ARIAL_BOLD):
        if path is None:
            continue
        if path.exists():
            try:
                font_manager.fontManager.addfont(str(path))
            except (OSError, ValueError):
                continue
    if _HELVETICA.exists():
        return "Helvetica"
    if _ARIAL.exists():
        return "Arial"
    return "DejaVu Sans"


def apply() -> None:
    global FONT, FONT_BOLD
    family = _register_fonts()
    params = dict(RCPARAMS)
    params["font.sans-serif"] = [family, "Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams.update(params)
    if _HELVETICA.exists():
        FONT = FontProperties(fname=str(_HELVETICA))
        bold = _helvetica_bold()
        FONT_BOLD = FontProperties(fname=str(bold if bold is not None else _HELVETICA))
    elif _ARIAL.exists():
        FONT = FontProperties(fname=str(_ARIAL))
        FONT_BOLD = FontProperties(fname=str(_ARIAL_BOLD) if _ARIAL_BOLD.exists() else str(_ARIAL))
    else:
        FONT = FontProperties(family=family)
        FONT_BOLD = FontProperties(family=family, weight="bold")


def _fp(bold: bool = False, size: float | None = None) -> FontProperties:
    base = FONT_BOLD if bold and FONT_BOLD is not None else FONT
    scaled_size = None if size is None else _text_size(size)
    if base is None:
        return FontProperties(
            family="sans-serif",
            weight="bold" if bold else "normal",
            size=scaled_size,
        )
    fp = base.copy()
    if scaled_size is not None:
        fp.set_size(scaled_size)
    return fp


# Layout inches for comparison_grid at print size. ``_TIGHT_COL_GAP`` is the
# map-to-map spacing once a column's colour-bar slot has been reclaimed.
_CBAR_W = 0.06
_CBAR_PAD = 0.06
_COL_GAP = 0.50
_TIGHT_COL_GAP = 0.14
_LEFT = 0.42            # y tick labels and the z label
_ROW_LABEL_ROOM = 0.18  # extra left margin for row_label()
_RIGHT_CBAR = 0.46      # right margin when the last column keeps its colour bar
_RIGHT_BARE = 0.06
_BOTTOM = 0.34
_TOP = 0.26
_CBAR_SHRINK = 0.90


class ComparisonGrid:
    """Maps in a grid that always spans ``width`` inches.

    Panel heights are shared and widths follow each column's aspect. When a
    colour-bar column is collapsed (see ``hide_cbar``) the layout is solved
    again, so the reclaimed width goes to the maps instead of to white space.
    """

    def __init__(
        self,
        fig,
        axes,
        cbar_axes,
        nrows: int,
        ncols: int,
        aspect: float,
        *,
        column_aspects,
        width: float,
        row_gap: float,
        left: float,
        right: float | None = None,
        cbar_w: float = _CBAR_W,
        cbar_pad: float = _CBAR_PAD,
        col_gap: float = _COL_GAP,
        tight_col_gap: float = _TIGHT_COL_GAP,
    ):
        self.fig = fig
        self.axes = list(axes)
        self.cbar_axes = list(cbar_axes)
        self.nrows = nrows
        self.ncols = ncols
        self.aspect = aspect
        self.column_aspects = [max(float(a), 1e-6) for a in column_aspects]
        self.width = float(width)
        self.row_gap = float(row_gap)
        self.left = float(left)
        self.right = None if right is None else float(right)
        self.cbar_w = float(cbar_w)
        self.cbar_pad = float(cbar_pad)
        self.col_gap = float(col_gap)
        self.tight_col_gap = float(tight_col_gap)
        self._collapsed_cols: set[int] = set()
        for cax in self.cbar_axes:
            cax._comparison_grid = self
        fig._print_width = self.width
        self.layout()

    def layout(self) -> None:
        """Place every map and colour-bar slot so the grid spans ``width``."""
        slots = [c not in self._collapsed_cols for c in range(self.ncols)]
        gaps = [
            self.col_gap if slots[c] else self.tight_col_gap
            for c in range(self.ncols - 1)
        ]
        right = _RIGHT_CBAR if slots[-1] else _RIGHT_BARE
        if self.right is not None:
            right = self.right
        fixed = (
            self.left + right + sum(gaps)
            + sum(self.cbar_pad + self.cbar_w for c in range(self.ncols) if slots[c])
        )
        plot_h = (self.width - fixed) / sum(self.column_aspects)
        if plot_h <= 0:
            raise ValueError("comparison_grid: margins leave no room for the maps")
        widths = [plot_h * a for a in self.column_aspects]
        fig_w = self.width
        fig_h = _BOTTOM + _TOP + self.nrows * plot_h + (self.nrows - 1) * self.row_gap
        self.fig.set_size_inches(fig_w, fig_h)
        self.plot_h = plot_h
        for r in range(self.nrows):
            y = _BOTTOM + (self.nrows - 1 - r) * (plot_h + self.row_gap)
            x = self.left
            for c in range(self.ncols):
                j = r * self.ncols + c
                self.axes[j].set_position(
                    (x / fig_w, y / fig_h, widths[c] / fig_w, plot_h / fig_h)
                )
                self.cbar_axes[j].set_position((
                    (x + widths[c] + self.cbar_pad) / fig_w,
                    (y + plot_h * (1.0 - _CBAR_SHRINK) / 2.0) / fig_h,
                    self.cbar_w / fig_w,
                    (plot_h * _CBAR_SHRINK) / fig_h,
                ))
                x += widths[c]
                if c < self.ncols - 1:
                    x += (self.cbar_pad + self.cbar_w if slots[c] else 0.0) + gaps[c]

    def __getitem__(self, i):
        return self.axes[i]

    def __len__(self):
        return len(self.axes)

    def __iter__(self):
        return iter(self.axes)


def view_aspect(xlim, zlim) -> float:
    """Geographic aspect Δx / Δz of a map view."""
    dx = abs(float(xlim[1]) - float(xlim[0]))
    dz = abs(float(zlim[1]) - float(zlim[0]))
    if dz < 1e-12:
        return 1.0
    return dx / dz


def _panel_wh(aspect: float, panel: float = 3.05, min_panel: float = 2.0) -> tuple[float, float]:
    """Map box in inches so width/height = geographic aspect."""
    aspect = max(float(aspect), 1e-6)
    if aspect >= 1.0:
        plot_w = float(panel)
        plot_h = plot_w / aspect
        if plot_h < min_panel:
            plot_h = min_panel
            plot_w = plot_h * aspect
    else:
        plot_h = float(panel)
        plot_w = plot_h * aspect
        if plot_w < min_panel:
            plot_w = min_panel
            plot_h = plot_w / aspect
    return plot_w, plot_h


def comparison_grid(
    nrows: int = 2,
    ncols: int = 3,
    *,
    aspect: float = 1.0,
    xlim=None,
    zlim=None,
    column_aspects=None,
    width: float = PRINT_WIDTH,
    row_gap: float = 0.40,
    row_labels: bool = False,
    right: float | None = None,
    panel: float | None = None,
    max_figwidth: float | None = None,
):
    """Maps with a vertical colour-bar slot to the right of each.

    The grid spans ``width`` inches (180 mm by default), and panel sizes are
    solved to fill it. Panel boxes match the view aspect (Δx/Δz) so
    ``set_aspect("equal")`` fills the axes instead of letterboxing inside a
    square slot. Pass ``xlim`` and ``zlim`` (or ``aspect``) from the
    evaluation grid. ``column_aspects`` may supply one geographic aspect per
    column when columns use different coordinate limits. ``row_labels`` makes
    room on the left for ``row_label``. ``right`` overrides the right margin
    (inches), for a last panel that carries its axis on the right.
    ``panel`` and ``max_figwidth`` are
    accepted for older notebooks and ignored: the width sets the panel size.
    """
    if xlim is not None and zlim is not None:
        aspect = view_aspect(xlim, zlim)
    aspect = max(float(aspect), 1e-6)
    if column_aspects is None:
        column_aspects = [aspect] * ncols
    elif len(column_aspects) != ncols:
        raise ValueError("column_aspects must have one value per column")
    left = _LEFT + (_ROW_LABEL_ROOM if row_labels else 0.0)

    fig = plt.figure(figsize=(width, width), facecolor="white")
    axes, cbar_axes = [], []
    for _ in range(nrows * ncols):
        axes.append(fig.add_axes((0.0, 0.0, 0.1, 0.1)))
        cbar_axes.append(fig.add_axes((0.0, 0.0, 0.01, 0.1)))
    return fig, ComparisonGrid(
        fig, axes, cbar_axes, nrows, ncols, aspect,
        column_aspects=column_aspects, width=width, row_gap=row_gap, left=left,
        right=right,
    )


def print_subplots(nrows: int = 1, ncols: int = 1, *, height: float,
                   width: float = PRINT_WIDTH, **kw):
    """``plt.subplots`` at print width; ``save`` keeps the file this wide."""
    fig, axs = plt.subplots(nrows, ncols, figsize=(width, height), **kw)
    fig._print_width = width
    return fig, axs


def print_figure(*, height: float, width: float = PRINT_WIDTH):
    """Empty figure at print width, for layouts placed in inches."""
    fig = plt.figure(figsize=(width, height), facecolor="white")
    fig._print_width = width
    return fig


def row_label(ax, text: str) -> None:
    """Bold row name, rotated, to the left of the y label of ``ax``."""
    ax.annotate(
        text,
        xy=(0.0, 0.5), xycoords="axes fraction",
        xytext=(-30.0, 0.0), textcoords="offset points",
        rotation=90, ha="right", va="center",
        fontproperties=_fp(bold=True, size=11), color="0.1",
        annotation_clip=False,
    )


def bottom_cbar(ax, mappable, label: str, *, drop: float = 0.36, **kw):
    """Horizontal colour bar centred under ``ax``, its top ``drop`` inches below."""
    fig = ax.figure
    fig_h = fig.get_figheight()
    pos = ax.get_position()
    cax = fig.add_axes((
        pos.x0 + 0.15 * pos.width,
        pos.y0 - (drop + 0.06) / fig_h,
        0.7 * pos.width,
        0.06 / fig_h,
    ))
    return add_cbar(cax, mappable, label, **kw)


def bottom_legend(ax, handles, *, drop: float = 0.36, **kw):
    """Figure legend centred under ``ax``, its top ``drop`` inches below."""
    fig = ax.figure
    pos = ax.get_position()
    kw.setdefault("frameon", False)
    return fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(pos.x0 + 0.5 * pos.width, pos.y0 - drop / fig.get_figheight()),
        **kw,
    )


def section_figure(xlim, zlim, *, width: float = 14.0):
    """One map whose box matches the view aspect (for diagnostic plots)."""
    plot_w, plot_h = _panel_wh(view_aspect(xlim, zlim), panel=width)
    left, right, bottom, top = 0.72, 0.35, 0.52, 0.42
    fig_w = left + plot_w + right
    fig_h = bottom + plot_h + top
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    ax = fig.add_axes((left / fig_w, bottom / fig_h, plot_w / fig_w, plot_h / fig_h))
    ax.set_xlim(float(xlim[0]), float(xlim[1]))
    ax.set_ylim(float(zlim[0]), float(zlim[1]))
    ax.set_aspect("equal", adjustable="box")
    return fig, ax


def panel_label(ax, letter: str) -> None:
    """Panel letter in the margin outside the axes (Nature-style)."""
    label = letter.strip().lower()
    fp = _fp(bold=True, size=8.0 / TEXT_SCALE)   # 8 pt bold in print
    pe = [withStroke(linewidth=2.2, foreground="white")]
    if getattr(ax, "name", "") == "3d":
        ax.text2D(
            0.035, 0.965, label,
            transform=ax.transAxes,
            fontproperties=fp, va="top", ha="left", color="0.08",
            path_effects=pe, zorder=10, clip_on=False,
        )
        return
    # Anchor the label to the axes, not to its current figure coordinates.
    # This keeps it attached when tight_layout or a hidden colour-bar slot
    # subsequently moves the axes. The offset is in physical points so it is
    # consistent across single- and multi-column figures.
    offset = ScaledTranslation(
        -4.0 / 72.0, 2.0 / 72.0, ax.figure.dpi_scale_trans
    )
    ax.text(
        0.0, 1.0, label,
        transform=ax.transAxes + offset,
        fontproperties=fp, va="bottom", ha="right", color="0.08",
        path_effects=pe, zorder=10, clip_on=False,
    )


def format_section_ax(
    ax,
    *,
    xlabel: bool = True,
    ylabel: bool = True,
    title: str | None = None,
    title_label: str | None = None,
    unit: str = "m",
) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3, steps=[1, 2, 5]))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, steps=[1, 2, 5]))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.tick_params(which="both", length=2.4, width=0.5, colors="0.25", labelcolor="0.2")
    ax.set_xlabel(f"x ({unit})" if xlabel else "", fontproperties=_fp(size=10))
    ax.set_ylabel(f"z ({unit})" if ylabel else "", fontproperties=_fp(size=10))
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(_fp(size=9))
    if not xlabel:
        ax.tick_params(labelbottom=False)
    if not ylabel:
        ax.tick_params(labelleft=False)
    if title:
        ax.set_title(title, color="0.15", fontproperties=_fp(size=11))
    if title_label:
        ax.set_title(title_label, color="0.15", fontproperties=_fp(size=11), loc="left")


def set_view(ax, xlim, zlim) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.set_aspect("equal", adjustable="box")


def _horizon_color(name, index: int, colors: dict | None):
    """Solid colour for a named horizon; fall back to palette order, not C0."""
    table = colors if colors is not None else FORMATION_COLORS
    if name in table:
        return table[name]
    palette = list(table.values()) if table else list(FORMATION_PALETTE)
    if not palette:
        palette = list(FORMATION_PALETTE)
    return palette[index % len(palette)]


def draw_younging(
    ax,
    gp,
    gv,
    *,
    quiver_scale: float = 0.018,
    color: str = NORMAL_COLOR,
) -> None:
    """Way-up T-bar: cap on the bedding, stem towards younger rocks.

    ``gv`` is the signed bedding normal (younging). Drawn in data coordinates
    so the stem stays perpendicular to bedding under equal aspect.
    """
    gp = np.asarray(gp, dtype=float)
    gv = np.asarray(gv, dtype=float)
    if gp.size == 0 or gv.size == 0:
        return
    gp = np.atleast_2d(gp)[:, :2]
    gv = np.atleast_2d(gv)[:, :2]
    nlen = np.linalg.norm(gv, axis=1, keepdims=True)
    nlen = np.maximum(nlen, 1e-12)
    n = gv / nlen
    tangent = np.stack([-n[:, 1], n[:, 0]], axis=1)
    size = 1.0 / max(float(quiver_scale), 1e-8)
    bar_half = 0.42 * size
    stem = 0.48 * size
    caps = np.stack([gp - bar_half * tangent, gp + bar_half * tangent], axis=1)
    stems = np.stack([gp, gp + stem * n], axis=1)
    segs = np.concatenate([caps, stems], axis=0)
    # A row per point colours that point's cap and stem together.
    if not isinstance(color, str):
        point_colors = np.asarray(color)
        if point_colors.ndim == 2 and len(point_colors) == len(gp):
            color = np.concatenate([point_colors, point_colors], axis=0)
    ax.add_collection(
        LineCollection(
            segs, colors=color, linewidths=YOUNGING_LW, capstyle="butt",
            zorder=5, clip_on=True,
        )
    )


def overlay_constraints(
    ax,
    horizons,
    horizon_names,
    gp,
    gv,
    *,
    colors: dict | None = None,
    scatter_s: float = 8,
    quiver_scale: float = 0.018,
    show_scatter: bool = True,
    show_quiver: bool = True,
    legend: bool = False,
    included=None,
    fade_alpha: float = 0.28,
) -> None:
    if show_scatter:
        for i, (name, h) in enumerate(zip(horizon_names, horizons)):
            color = _horizon_color(name, i, colors)
            pts = np.asarray(h)
            if included is None:
                ax.scatter(
                    pts[:, 0], pts[:, 1],
                    s=scatter_s, color=color,
                    edgecolors="k", linewidths=0.3,
                    zorder=4, label=name,
                )
                continue
            keep = np.asarray(included[i], dtype=bool)
            if (~keep).any():
                ax.scatter(
                    pts[~keep, 0], pts[~keep, 1],
                    s=scatter_s, color=color, alpha=fade_alpha,
                    edgecolors="none", linewidths=0,
                    zorder=3,
                )
            if keep.any():
                ax.scatter(
                    pts[keep, 0], pts[keep, 1],
                    s=scatter_s, color=color, alpha=1.0,
                    edgecolors="k", linewidths=0.3,
                    zorder=4, label=name,
                )
    if show_quiver:
        draw_younging(ax, gp, gv, quiver_scale=quiver_scale)
    if legend:
        leg = ax.legend(
            handles=constraint_handles(colors, horizon_names),
            loc="lower right",
            borderaxespad=0.6,
            labelspacing=0.55,
            handletextpad=0.5,
            prop=_fp(size=9),
        )
        for text in leg.get_texts():
            text.set_fontproperties(_fp(size=9))


def _sample_grid(x, z, field, pts):
    """Bilinear sample of ``field[ix, iz]`` at ``pts[:, :2]``."""
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    field = np.asarray(field, dtype=float)
    pts = np.atleast_2d(np.asarray(pts, dtype=float))[:, :2]
    ix = np.clip(np.interp(pts[:, 0], x, np.arange(x.size)), 0.0, x.size - 1.001)
    iz = np.clip(np.interp(pts[:, 1], z, np.arange(z.size)), 0.0, z.size - 1.001)
    i0 = np.floor(ix).astype(np.intp)
    j0 = np.floor(iz).astype(np.intp)
    wx = ix - i0
    wz = iz - j0
    return (
        (1.0 - wx) * (1.0 - wz) * field[i0, j0]
        + wx * (1.0 - wz) * field[i0 + 1, j0]
        + (1.0 - wx) * wz * field[i0, j0 + 1]
        + wx * wz * field[i0 + 1, j0 + 1]
    )


def draw_contact_contours(
    ax,
    x,
    z,
    field,
    horizons,
    horizon_names,
    *,
    colors: dict | None = None,
    linewidth: float = 0.9,
    edgecolor: str = "white",
    edgewidth: float = 1.0,
    levels=None,
) -> None:
    """Isocontours through each contact set, coloured like the points.

    ``field`` is ``(len(x), len(z))``, matching ``G.reshape(...)`` in the
    paper notebooks. One level per horizon: ``levels[i]`` when given, which
    should be the isovalues the lithology was classified with, so that the
    lines bound the units; otherwise the median φ at the horizon's contacts.
    A white halo is drawn under the coloured line so it reads on the ramp.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    field = np.asarray(field, dtype=float)
    Z = field.T
    for i, (name, h) in enumerate(zip(horizon_names, horizons)):
        if levels is not None:
            level = float(levels[i])
        else:
            vals = _sample_grid(x, z, field, h)
            vals = np.asarray(vals, dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            level = float(np.median(vals))
        color = _horizon_color(name, i, colors)
        if edgecolor:
            ax.contour(
                x, z, Z, levels=[level], colors=[edgecolor],
                linewidths=linewidth + edgewidth, linestyles="solid",
                zorder=3,
            )
        ax.contour(
            x, z, Z, levels=[level], colors=[color],
            linewidths=linewidth, linestyles="solid",
            zorder=3.1,
        )


def draw_intermediate_contours(
    ax,
    x,
    z,
    field,
    *,
    n_levels: int = 16,
    skip_levels=(),
    color: str = "0.2",
    linewidth: float = 0.25,
    alpha: float = 0.5,
) -> None:
    """Thin isocontours at evenly spaced levels across the whole field.

    They show how the isosurfaces between and beyond the contacts close (or
    not). Levels within half a spacing of any ``skip_levels`` (the contact
    isovalues, drawn by :func:`draw_contact_contours`) are left out.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    field = np.asarray(field, dtype=float)
    lo, hi = float(np.nanmin(field)), float(np.nanmax(field))
    levels = np.linspace(lo, hi, n_levels + 2)[1:-1]
    spacing = levels[1] - levels[0]
    for skip in skip_levels:
        levels = levels[np.abs(levels - float(skip)) > 0.5 * spacing]
    ax.contour(
        x, z, field.T, levels=levels, colors=[color],
        linewidths=linewidth, linestyles="solid", alpha=alpha, zorder=2.5,
    )


def style_axes3d(ax) -> None:
    """Transparent panes with a readable coordinate grid for mplot3d."""
    ax.set_facecolor("white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("0.80")
        axis._axinfo["grid"]["color"] = (0, 0, 0, 0.16)
        axis._axinfo["grid"]["linewidth"] = 0.65
    ax.tick_params(
        axis="both", which="major",
        length=3.2, width=0.6, colors="0.25", labelcolor="0.2", pad=2.0,
    )


def format_axes3d(
    ax,
    *,
    xlabel: str = "x (m)",
    ylabel: str = "y (m)",
    zlabel: str = "z (m)",
    title: str | None = None,
) -> None:
    """Apply shared fonts and integer tick labels to a 3-D axes."""
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(ticker.MaxNLocator(nbins=4, steps=[1, 2, 5]))
        axis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    for lab in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        lab.set_fontproperties(_fp(size=9))
    ax.set_xlabel(xlabel, fontproperties=_fp(size=10), labelpad=8)
    ax.set_ylabel(ylabel, fontproperties=_fp(size=10), labelpad=8)
    # mplot3d rotates the z label edge-on by default; keep it readable and
    # leave figure margin so tight saves do not clip it.
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(zlabel, fontproperties=_fp(size=10), labelpad=12, color="0.15")
    ax.figure.subplots_adjust(left=0.02, right=0.84, bottom=0.06, top=0.94)
    if title:
        ax.set_title(title, color="0.15", fontproperties=_fp(size=11), pad=10)


def _subsample_rows(pts, rng, max_n: int):
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or len(pts) == 0 or len(pts) <= max_n:
        return pts
    return pts[rng.choice(len(pts), size=max_n, replace=False)]


def _circle_xy(center, radius: float, n: int = 48):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=True)
    x = center[0] + radius * np.cos(t)
    y = center[1] + radius * np.sin(t)
    return x, y


def _bedding_plane_corners(center, normal, half_size: float) -> np.ndarray:
    """Square corners spanning the plane perpendicular to ``normal``."""
    center = np.asarray(center, dtype=float)
    n = np.asarray(normal, dtype=float)
    n /= max(float(np.linalg.norm(n)), 1e-12)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(n, ref))) > 0.92:
        ref = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, ref)
    t1 /= max(float(np.linalg.norm(t1)), 1e-12)
    t2 = np.cross(n, t1)
    hs = float(half_size)
    return np.array([
        center + hs * (t1 + t2),
        center + hs * (-t1 + t2),
        center + hs * (-t1 - t2),
        center + hs * (t1 - t2),
    ])


def _draw_bedding_glyphs_3d(
    ax,
    gp,
    gv,
    *,
    plane_half: float,
    normal_length: float,
    plane_alpha: float = 0.26,
    upright_color: str = NORMAL_COLOR,
    inverted_color: str = FORMATION_PALETTE[3],
) -> None:
    """Local bedding patch with a normal arrow at each orientation sample."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    gp = np.asarray(gp, dtype=float)
    gv = np.asarray(gv, dtype=float)
    if gp.size == 0 or gv.size == 0:
        return
    gp = np.atleast_2d(gp)
    gv = np.atleast_2d(gv)
    up = gv[:, 2] >= 0.0
    for mask, color in ((up, upright_color), (~up, inverted_color)):
        if not np.any(mask):
            continue
        for center, normal in zip(gp[mask], gv[mask]):
            corners = _bedding_plane_corners(center, normal, plane_half)
            ax.add_collection3d(Poly3DCollection(
                [corners],
                facecolors=(*to_rgb(color), plane_alpha),
                edgecolors=color, linewidths=0.65, zorder=5,
            ))
            n = np.asarray(normal, dtype=float)
            n /= max(float(np.linalg.norm(n)), 1e-12)
            tip = center + normal_length * n
            ax.plot(
                [center[0], tip[0]], [center[1], tip[1]], [center[2], tip[2]],
                color=color, lw=1.15, solid_capstyle="round", zorder=6,
            )
            ax.scatter(
                [tip[0]], [tip[1]], [tip[2]],
                s=14, marker="o", color=color, edgecolors="white",
                linewidths=0.35, depthshade=False, zorder=7,
            )


def _bedding_glyph_legend_handle(color: str, label: str, *, plane_alpha: float = 0.26) -> Line2D:
    return Line2D(
        [0.05, 0.55], [0.35, 0.75], color=color, lw=1.1,
        marker="s", markersize=7.5,
        markerfacecolor=(*to_rgb(color), plane_alpha),
        markeredgecolor=color, markeredgewidth=0.65,
        label=label,
    )


def _draw_xy_imprint(ax, center, radius, z_floor, *, color, alpha=0.22, lw=0.9):
    """Footprint of a vertical drillhole on the plan view at ``z_floor``."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    x, y = _circle_xy(center, radius)
    z = np.full_like(x, z_floor)
    ring = np.column_stack([x, y, z])
    ax.add_collection3d(Poly3DCollection(
        [ring], facecolors=(*to_rgb(color), alpha),
        edgecolors=color, linewidths=lw, zorder=0,
    ))
    ax.scatter(
        [center[0]], [center[1]], [z_floor],
        s=16, marker=".", color=color, alpha=0.85, depthshade=False, zorder=1,
    )


def plot_drillhole_data3d(
    ax,
    horizon_names,
    horizons_eq,
    horizons_far,
    *,
    collars=None,
    hole_radius: float = 10.0,
    gp=None,
    gv=None,
    colors: dict | None = None,
    context_max: int = 120,
    context_s: float = 4.5,
    context_alpha: float = 0.14,
    sample_s: float = 32.0,
    show_traces: bool = True,
    show_xy_imprints: bool = True,
    trace_color: str = "0.55",
    imprint_color: str = "0.45",
    imprint_alpha: float = 0.22,
    bedding_glyph_scale: float | None = None,
    bedding_plane_alpha: float = 0.26,
    seed: int = 0,
    vertical_exag: float = 1.0,
    elev: float = 22.0,
    azim: float = -58.0,
    title: str | None = None,
    legend: bool = True,
    legend_compact: bool = False,
    domain_limits: tuple | None = None,
    extra_focus=None,
    focus_pad: float = 0.12,
) -> None:
    """Sparse vertical drillholes on a thinned 3-D contact cloud.

    Context contacts are subsampled; sampled equalities are drawn as filled
    markers on grey hole traces with collar ticks. Bedding orientations are
    shown as a local plane patch with a normal arrow (``gv`` = younging).
    Each collar can leave a circular footprint on the *x*–*y* floor.
    ``vertical_exag`` scales only the display box via ``set_box_aspect``
    (1 = true proportions; >1 stretches *z* visually).
    """
    style_axes3d(ax)
    rng = np.random.default_rng(seed)
    per_h = max(1, context_max // max(len(horizon_names), 1))

    eq_pts = [np.asarray(h, dtype=float) for h in horizons_eq if len(h)]
    all_eq = np.vstack(eq_pts) if eq_pts else np.empty((0, 3))

    focus = all_eq if len(all_eq) else np.vstack(
        [np.asarray(h, dtype=float) for h in horizons_far if len(h)]
    )
    if gp is not None and len(gp):
        focus = np.vstack([focus, np.asarray(gp, dtype=float)])
    if extra_focus is not None:
        extra_focus = np.asarray(extra_focus, dtype=float)
        if extra_focus.size:
            focus = np.vstack([focus, np.atleast_2d(extra_focus)])
    if domain_limits is not None:
        xlim, ylim, zlim = domain_limits
        xlim = (float(xlim[0]), float(xlim[1]))
        ylim = (float(ylim[0]), float(ylim[1]))
        zlim = (float(zlim[0]), float(zlim[1]))
        z_floor = zlim[0]
        dx = xlim[1] - xlim[0]
        dy = ylim[1] - ylim[0]
        dz = zlim[1] - zlim[0]
    elif len(focus):
        pad = focus_pad * np.maximum(np.ptp(focus, axis=0), 1.0)
        xlim = (float(focus[:, 0].min() - pad[0]), float(focus[:, 0].max() + pad[0]))
        ylim = (float(focus[:, 1].min() - pad[1]), float(focus[:, 1].max() + pad[1]))
        zlim = (float(focus[:, 2].min() - pad[2]), float(focus[:, 2].max() + pad[2]))
        z_floor = zlim[0]
        dx = xlim[1] - xlim[0]
        dy = ylim[1] - ylim[0]
        dz = zlim[1] - zlim[0]
    else:
        xlim = ylim = zlim = None
        z_floor = 0.0
        dx = dy = dz = 1.0

    for i, (name, h) in enumerate(zip(horizon_names, horizons_far)):
        h = _subsample_rows(h, rng, per_h)
        if len(h) == 0:
            continue
        ax.scatter(
            h[:, 0], h[:, 1], h[:, 2],
            s=context_s, facecolors="none",
            edgecolors=_horizon_color(name, i, colors),
            linewidths=0.35, alpha=context_alpha, depthshade=False, zorder=1,
        )

    if show_xy_imprints and collars is not None and len(collars):
        collars_arr = np.asarray(collars, dtype=float)
        for c in collars_arr:
            _draw_xy_imprint(
                ax, c, hole_radius, z_floor,
                color=imprint_color, alpha=imprint_alpha,
            )

    if show_traces and collars is not None and len(collars):
        collars_arr = np.asarray(collars, dtype=float)
        for c in collars_arr:
            if len(all_eq):
                near = np.hypot(all_eq[:, 0] - c[0], all_eq[:, 1] - c[1]) <= hole_radius
                z_pts = all_eq[near, 2] if np.any(near) else all_eq[:, 2]
            else:
                z_pts = np.array([0.0])
            z0, z1 = float(z_pts.min()), float(z_pts.max())
            pad_z = 0.05 * max(z1 - z0, 1.0)
            ax.plot(
                [c[0], c[0]], [c[1], c[1]], [z0 - pad_z, z1 + pad_z],
                color=trace_color, lw=1.35, alpha=0.8, zorder=2,
            )
            ax.scatter(
                [c[0]], [c[1]], [z1 + pad_z],
                s=30, marker="^", color="0.35", edgecolors="white",
                linewidths=0.35, depthshade=False, zorder=6,
            )
            if show_xy_imprints:
                ax.plot(
                    [c[0], c[0]], [c[1], c[1]], [z_floor, z0 - pad_z],
                    color=imprint_color, lw=0.65, ls=":", alpha=0.55, zorder=1,
                )

    for i, (name, h) in enumerate(zip(horizon_names, horizons_eq)):
        if len(h) == 0:
            continue
        ax.scatter(
            h[:, 0], h[:, 1], h[:, 2],
            s=sample_s, color=_horizon_color(name, i, colors),
            edgecolors="white", linewidths=0.5, depthshade=False, zorder=4,
        )

    if xlim is not None:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ve = max(float(vertical_exag), 1e-3)
        ax.set_box_aspect((dx, dy, ve * dz))

    if gp is not None and gv is not None and len(gp):
        extent = max(dx, dy, dz, 1.0)
        glyph_scale = (
            float(bedding_glyph_scale)
            if bedding_glyph_scale is not None
            else 0.042 * extent
        )
        _draw_bedding_glyphs_3d(
            ax, gp, gv,
            plane_half=glyph_scale,
            normal_length=glyph_scale * 1.2,
            plane_alpha=bedding_plane_alpha,
        )

    ax.view_init(elev=elev, azim=azim)
    format_axes3d(ax, title=title)

    handles = constraint_handles(colors, horizon_names)[:-1]
    if not legend_compact:
        handles.extend([
            Line2D(
                [0], [0], marker="o", color="none", markerfacecolor="none",
                markeredgecolor="0.45", markeredgewidth=0.8, markersize=5,
                label="Context",
            ),
            Line2D([0], [0], color=trace_color, lw=1.35, label="Drillhole"),
        ])
        if show_xy_imprints:
            handles.append(
                Line2D(
                    [0], [0], marker="o", color="none",
                    markerfacecolor=(*to_rgb(imprint_color), imprint_alpha),
                    markeredgecolor=imprint_color, markeredgewidth=0.8,
                    markersize=7, label="Collar (plan)",
                )
            )
    if getattr(ax, "name", "") == "3d":
        handles.extend([
            _bedding_glyph_legend_handle(
                NORMAL_COLOR, "Younging (upright)", plane_alpha=bedding_plane_alpha,
            ),
            _bedding_glyph_legend_handle(
                FORMATION_PALETTE[3], "Younging (inverted)", plane_alpha=bedding_plane_alpha,
            ),
        ])
    else:
        handles.extend([
            Line2D(
                [0], [0], color="none", marker=_younging_marker(),
                markerfacecolor=NORMAL_COLOR, markeredgecolor=NORMAL_COLOR,
                markeredgewidth=0.0, markersize=8, label="Younging (upright)",
            ),
            Line2D(
                [0], [0], color="none", marker=_younging_marker(),
                markerfacecolor=FORMATION_PALETTE[3], markeredgecolor=FORMATION_PALETTE[3],
                markeredgewidth=0.0, markersize=8, label="Younging (inverted)",
            ),
        ])
    if legend:
        legend_kw = dict(frameon=False, borderaxespad=0.35, labelspacing=0.38, prop=_fp(size=8))
        if getattr(ax, "name", "") == "3d":
            anchor = (0.98, 0.98) if legend_compact else (0.03, 0.52)
            loc = "upper right" if legend_compact else "center left"
            ax.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, **legend_kw)
        else:
            ax.legend(
                handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.02),
                **legend_kw,
            )


def assign_litho_from_phi(phi, horizons) -> np.ndarray:
    """Bin restored depth ``phi`` into integer unit IDs (1 = shallowest)."""
    h = np.asarray(horizons, dtype=float).ravel()
    phi = np.asarray(phi, dtype=float)
    litho = np.full(phi.shape, len(h), dtype=int)
    litho[phi > h[0]] = 1
    for i in range(len(h) - 1):
        litho[(phi <= h[i]) & (phi > h[i + 1])] = i + 2
    return litho


def draw_truth_plane_3d(
    ax,
    slice_axis: int,
    v: float,
    ha,
    va,
    litho,
    cmap,
    norm,
    *,
    alpha: float = 0.42,
    stride: int = 1,
    edgecolor: str = "0.2",
    edgelw: float = 0.85,
) -> None:
    """Ground-truth lithology on a vertical plane embedded in a 3-D axes."""
    ha = np.asarray(ha, dtype=float)[::stride]
    va = np.asarray(va, dtype=float)[::stride]
    litho = np.asarray(litho, dtype=float)[::stride, ::stride]
    hh, vv = np.meshgrid(ha, va, indexing="xy")
    colors = np.asarray(cmap(norm(litho)), dtype=float)
    v = float(v)
    if slice_axis == 1:
        ax.plot_surface(
            hh, np.full_like(hh, v), vv,
            facecolors=colors, shade=False, alpha=alpha,
            rstride=1, cstride=1, linewidth=0, antialiased=False, zorder=2,
        )
        ax.plot(
            [ha[0], ha[-1], ha[-1], ha[0], ha[0]],
            [v, v, v, v, v],
            [va[0], va[0], va[-1], va[-1], va[0]],
            color=edgecolor, lw=edgelw, alpha=min(1.0, alpha + 0.35), zorder=3,
        )
    elif slice_axis == 0:
        ax.plot_surface(
            np.full_like(hh, v), hh, vv,
            facecolors=colors, shade=False, alpha=alpha,
            rstride=1, cstride=1, linewidth=0, antialiased=False, zorder=2,
        )
        ax.plot(
            [v, v, v, v, v],
            [ha[0], ha[-1], ha[-1], ha[0], ha[0]],
            [va[0], va[0], va[-1], va[-1], va[0]],
            color=edgecolor, lw=edgelw, alpha=min(1.0, alpha + 0.35), zorder=3,
        )
    else:
        raise ValueError("slice_axis must be 0 (y–z) or 1 (x–z)")


def section_plane_corners(slice_axis: int, v: float, domaxes) -> np.ndarray:
    """Four corners of a section plane at ``v`` (for 3-D axis framing)."""
    hax, vax = (0, 2) if slice_axis == 1 else (1, 2)
    pts = []
    for a in (float(domaxes[hax].min()), float(domaxes[hax].max())):
        for b in (float(domaxes[vax].min()), float(domaxes[vax].max())):
            p = np.zeros(3, dtype=float)
            p[slice_axis] = float(v)
            p[hax] = a
            p[vax] = b
            pts.append(p)
    return np.asarray(pts)


def pericline_comparison_figure(
    xz_aspect: float,
    yz_aspect: float,
    *,
    panel: float = 2.35,
    max_width: float = 18.0,
):
    """4×3 layout: 3-D block (2×2), implicit 2×2, restore 1×4."""
    plot_w = float(panel)
    h_xz = plot_w / max(float(xz_aspect), 1e-6)
    h_yz = plot_w / max(float(yz_aspect), 1e-6)
    h_row3 = max(h_xz, h_yz)
    row_gap = 0.42
    col_gap = 0.48
    left = 0.55
    right = 0.30
    bottom = 0.46
    top = 0.34
    cbar_w = 0.085
    cbar_pad = 0.07

    block_w = 2.0 * plot_w + col_gap
    span_w = 4.0 * plot_w + 3.0 * col_gap
    fig_w = left + span_w + cbar_pad + cbar_w + right
    if fig_w > max_width:
        scale = (max_width - left - right - cbar_pad - cbar_w) / span_w
        plot_w *= scale
        h_xz *= scale
        h_yz *= scale
        h_row3 *= scale
        block_w = 2.0 * plot_w + col_gap
        span_w = 4.0 * plot_w + 3.0 * col_gap
        fig_w = left + span_w + cbar_pad + cbar_w + right

    fig_h = bottom + top + h_xz + row_gap + h_yz + row_gap + h_row3
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    def _col_x(col: int) -> float:
        return (left + col * (plot_w + col_gap)) / fig_w

    x3d = left / fig_w
    w3d = block_w / fig_w
    y_r2 = bottom / fig_h
    y_r1 = (bottom + h_row3 + row_gap) / fig_h
    y_r0 = (bottom + h_row3 + row_gap + h_yz + row_gap) / fig_h
    h3d = (h_yz + row_gap + h_xz) / fig_h
    pw = plot_w / fig_w

    axes = {
        "3d": fig.add_axes((x3d, y_r1, w3d, h3d), projection="3d"),
        "imp_xz_core": fig.add_axes((_col_x(2), y_r0, pw, h_xz / fig_h)),
        "imp_xz_flank": fig.add_axes((_col_x(3), y_r0, pw, h_xz / fig_h)),
        "imp_yz_core": fig.add_axes((_col_x(2), y_r1, pw, h_yz / fig_h)),
        "imp_yz_flank": fig.add_axes((_col_x(3), y_r1, pw, h_yz / fig_h)),
        "rest_xz_core": fig.add_axes((_col_x(0), y_r2, pw, h_row3 / fig_h)),
        "rest_xz_flank": fig.add_axes((_col_x(1), y_r2, pw, h_row3 / fig_h)),
        "rest_yz_core": fig.add_axes((_col_x(2), y_r2, pw, h_row3 / fig_h)),
        "rest_yz_flank": fig.add_axes((_col_x(3), y_r2, pw, h_row3 / fig_h)),
    }
    cbar_x = (left + span_w + cbar_pad) / fig_w
    cbar_y = y_r2
    cbar_h = (h_row3 + row_gap + h_yz + row_gap + h_xz) / fig_h
    cbar_ax = fig.add_axes((cbar_x, cbar_y, cbar_w / fig_w, cbar_h))
    return fig, axes, cbar_ax


def _younging_marker() -> MplPath:
    """Filled T: bar on the bedding, stem pointing up (younging)."""
    w, t = 0.95, 0.22
    y_bar, y_top = -0.42, 0.78
    verts = [
        (-w, y_bar - t), (w, y_bar - t), (w, y_bar + t),
        (t, y_bar + t), (t, y_top), (-t, y_top),
        (-t, y_bar + t), (-w, y_bar + t), (-w, y_bar - t),
    ]
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * 7 + [MplPath.CLOSEPOLY]
    return MplPath(verts, codes)


def constraint_handles(colors=None, horizon_names=("rock1", "rock2")) -> list:
    handles = [
        Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor=_horizon_color(name, i, colors),
            markeredgecolor="white", markeredgewidth=0.3,
            markersize=4.5, label=name,
        )
        for i, name in enumerate(horizon_names)
    ]
    handles.append(
        Line2D(
            [0], [0], color="none",
            marker=_younging_marker(),
            markerfacecolor=NORMAL_COLOR,
            markeredgecolor=NORMAL_COLOR,
            markeredgewidth=0.0,
            markersize=6,
            label="Younging",
        )
    )
    return handles


def entropy_cmap():
    """Pale orange scale for lithology entropy.

    Light enough that contacts drawn over it stay legible, and warm so it
    separates from the blue horizon colours.
    """
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "entropy_pale", plt.get_cmap("Oranges")(np.linspace(0.0, 0.55, 256)),
    )


# curlew.ccramp, for kernels without curlew.
_CURLEW_RAMP = ("#A6340B", "#E35B0E", "#F39C12", "#F0C419", "#FAE8B6",
                "#8CD9E0", "#31B4C2", "#1B768F", "#054862")


def ramp():
    """Continuous colour map for scalar fields (``curlew.ccramp`` when available)."""

    try:
        from curlew import ccramp
    except ImportError:
        from matplotlib.colors import LinearSegmentedColormap
        ccramp = LinearSegmentedColormap.from_list("curlew_continuous", _CURLEW_RAMP, N=256)
    return ccramp


def litho_cmap(ids) -> tuple[ListedColormap, BoundaryNorm]:
    ccramp = ramp()
    ids = np.unique(np.asarray(ids))
    n = max(int(len(ids)), 2)
    cmap = ListedColormap(ccramp(np.linspace(0.0, 1.0, n)))
    bounds = np.concatenate([[ids[0] - 0.5], ids.astype(float) + 0.5])
    return cmap, BoundaryNorm(bounds, ncolors=cmap.N)


def _compact_e(v, _=None) -> str:
    if abs(v) < 1e-18:
        return "0"
    mant, exp = f"{v:.0e}".split("e")
    return f"{mant}e{int(exp):d}"


def _cbar_needs_scientific(mappable, ticks, threshold: float) -> bool:
    if ticks is not None:
        values = np.abs(np.asarray(ticks, dtype=float))
        return bool(np.any(np.isfinite(values)) and np.nanmax(values) >= threshold)
    clim = None
    if hasattr(mappable, "get_clim"):
        try:
            clim = mappable.get_clim()
        except (TypeError, ValueError):
            clim = None
    if clim is None:
        norm = getattr(mappable, "norm", None)
        clim = (getattr(norm, "vmin", None), getattr(norm, "vmax", None))
    finite = [abs(float(v)) for v in clim if v is not None and np.isfinite(v)]
    return bool(finite) and max(finite) >= threshold


def add_cbar(
    cax,
    mappable,
    label: str,
    *,
    ticks=None,
    scientific: bool | None = None,
    scientific_threshold: float = 1e3,
):
    pos = cax.get_position()
    horizontal = pos.width >= pos.height
    cb = cax.figure.colorbar(
        mappable, cax=cax, ticks=ticks,
        orientation="horizontal" if horizontal else "vertical",
    )
    axis = cb.ax.xaxis if horizontal else cb.ax.yaxis
    if horizontal:
        cb.set_label(label, rotation=0, labelpad=3, fontproperties=_fp(size=10), color="0.15")
    else:
        cb.set_label(label, rotation=270, labelpad=9, fontproperties=_fp(size=10), color="0.15")
    cb.ax.tick_params(length=2.0, width=0.4, labelsize=_text_size(8.5), pad=1.5, colors="0.25", labelcolor="0.2")
    cb.outline.set_linewidth(0.4)
    cb.outline.set_edgecolor("0.25")
    if scientific is None:
        scientific = _cbar_needs_scientific(mappable, ticks, scientific_threshold)
    if ticks is not None:
        cb.set_ticks(list(ticks))
        labels = [_compact_e(t) if scientific else f"{t:.0f}" for t in ticks]
        cb.set_ticklabels(labels, fontproperties=_fp(size=8.5))
    else:
        axis.set_major_locator(ticker.MaxNLocator(nbins=4))
        if scientific:
            fmt = ticker.FuncFormatter(_compact_e)
        else:
            fmt = ticker.ScalarFormatter()
            fmt.set_useOffset(False)
            fmt.set_scientific(False)
        axis.set_major_formatter(fmt)
    ticklabels = cb.ax.get_xticklabels() if horizontal else cb.ax.get_yticklabels()
    for lab in ticklabels:
        lab.set_fontproperties(_fp(size=8.5))
    return cb


def _collapse_cbar_column(grid: ComparisonGrid, col: int) -> None:
    """Drop a colour-bar column from the layout once every row has hidden it."""
    if col in grid._collapsed_cols:
        return
    grid._collapsed_cols.add(col)
    grid.layout()


def hide_cbar(cax) -> None:
    """Hide a colour-bar slot and collapse it when the whole column is unused.

    A 2-D grid keeps the slot if any row still shows a colour bar, so columns
    stay aligned. Once every row in that column is hidden, the grid is laid
    out again without it and the maps grow to fill the width.
    """
    cax.set_visible(False)
    grid = getattr(cax, "_comparison_grid", None)
    if grid is None:
        return
    try:
        idx = grid.cbar_axes.index(cax)
    except ValueError:
        return
    col = idx % grid.ncols
    if all(
        not grid.cbar_axes[r * grid.ncols + col].get_visible()
        for r in range(grid.nrows)
    ):
        _collapse_cbar_column(grid, col)


def add_colorbar(fig, mappable, ax, label: str, *, ticks=None):
    cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04, ticks=ticks)
    cb.set_label(label, fontproperties=_fp(size=10))
    cb.ax.tick_params(width=0.6, length=3.0)
    cb.outline.set_linewidth(0.6)
    return cb


def bright_rgb(amp01, val, cmap):
    """Multi-hue colour from a unit interval; lightness comes only from ``val``."""
    col = np.asarray(cmap(np.clip(amp01, 0.0, 1.0)), dtype=np.float64)[..., :3]
    mx = col.max(axis=-1, keepdims=True).clip(1e-6)
    return (col / mx) * np.clip(val, 0.0, 1.0)[..., None]


def arc_colorbar(
    ax,
    cmap,
    norm,
    ticks,
    tick_labels,
    *,
    theta0=24.0,
    theta1=58.0,
    r0=1.055,
    r1=1.20,
    val_gamma=0.6,
    amp_name=r"$|C_k|$",
    density_name="modes",
):
    """Short bivariate annulus: hue along the arc, brightness radially (inner empty)."""
    t0, t1 = np.deg2rad(float(theta0)), np.deg2rad(float(theta1))
    r0, r1 = float(r0), float(r1)
    n_th, n_r = 96, 28
    th_e = np.linspace(t0, t1, n_th + 1)
    r_e = np.linspace(r0, r1, n_r + 1)
    verts, cols = [], []
    for i in range(n_r):
        val = ((i + 0.5) / n_r) ** float(val_gamma)
        for j in range(n_th):
            amp01 = (j + 0.5) / n_th
            rgb = bright_rgb(np.array([amp01]), np.array([val]), cmap)[0]
            t00, t01 = th_e[j], th_e[j + 1]
            rr0, rr1 = r_e[i], r_e[i + 1]
            verts.append([
                (rr0 * np.cos(t00), rr0 * np.sin(t00)),
                (rr0 * np.cos(t01), rr0 * np.sin(t01)),
                (rr1 * np.cos(t01), rr1 * np.sin(t01)),
                (rr1 * np.cos(t00), rr1 * np.sin(t00)),
            ])
            cols.append(rgb)
    ax.add_collection(PolyCollection(
        verts, facecolors=cols, edgecolors="none", zorder=6, rasterized=True,
    ))
    th_line = np.linspace(t0, t1, 80)
    for rr in (r0, r1):
        ax.plot(
            rr * np.cos(th_line), rr * np.sin(th_line),
            color="0.25", lw=0.55, zorder=7, solid_capstyle="butt",
        )
    for tht in (t0, t1):
        ax.plot(
            [r0 * np.cos(tht), r1 * np.cos(tht)],
            [r0 * np.sin(tht), r1 * np.sin(tht)],
            color="0.25", lw=0.55, zorder=7, solid_capstyle="butt",
        )
    r_tick, r_lab = r1 + 0.028, r1 + 0.10
    for t, lab in zip(ticks, tick_labels):
        a01 = float(np.clip(np.ma.filled(norm(t), 0.0), 0.0, 1.0))
        tht = t0 + a01 * (t1 - t0)
        c, s = np.cos(tht), np.sin(tht)
        ax.plot(
            [r1 * c, r_tick * c], [r1 * s, r_tick * s],
            color="0.25", lw=0.5, zorder=7, solid_capstyle="butt",
        )
        ax.text(
            r_lab * c, r_lab * s, lab,
            ha="center", va="center", color="0.2",
            fontproperties=_fp(size=7), zorder=8, clip_on=False,
        )
    thm = 0.5 * (t0 + t1)
    if amp_name:
        ax.text(
            (r1 + 0.22) * np.cos(thm), (r1 + 0.22) * np.sin(thm), amp_name,
            ha="center", va="center", color="0.15",
            fontproperties=_fp(size=9), zorder=8, clip_on=False,
        )
    if density_name:
        # Just past the low end of the arc and outside it, so the label
        # never reaches the rim of the disc.
        t_lab = t0 - 0.12
        ax.text(
            (r1 + 0.05) * np.cos(t_lab),
            (r1 + 0.05) * np.sin(t_lab),
            density_name,
            ha="left", va="center", color="0.15",
            fontproperties=_fp(size=8), zorder=8, clip_on=False,
        )


def _figure_has_3d(fig) -> bool:
    return any(getattr(ax, "name", "") == "3d" for ax in fig.axes)


def save(fig, path, *, also_png: bool = True, tight: bool = True) -> list[Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(facecolor="white", dpi=300)
    print_width = getattr(fig, "_print_width", None)
    if tight and print_width is not None:
        # Crop to the content vertically but keep the full print width, so
        # the file is exactly as wide as it will be printed.
        from matplotlib.transforms import Bbox

        pad = 0.04
        # Draw first so constrained/tight layouts have placed the axes.
        fig.draw_without_rendering()
        box = fig.get_tightbbox(fig.canvas.get_renderer())
        x0 = min(0.0, box.x0 - pad)
        x1 = max(fig.get_figwidth(), box.x1 + pad)
        if x1 - x0 > fig.get_figwidth() + 1e-3:
            print(
                f"figstyle.save: {path.name} content is {25.4 * (x1 - x0):.0f} mm "
                f"wide, wider than its {25.4 * print_width:.0f} mm print width"
            )
        kw.update(bbox_inches=Bbox.from_extents(x0, box.y0 - pad, x1, box.y1 + pad))
    elif tight:
        pad = 0.32 if _figure_has_3d(fig) else 0.12
        kw.update(bbox_inches="tight", pad_inches=pad)
    pdf = path.with_suffix(".pdf")
    fig.savefig(pdf, **kw)
    written = [pdf]
    if also_png:
        png = path.with_suffix(".png")
        fig.savefig(png, **kw)
        written.append(png)
    return written
