"""Charts built to be projected in a management meeting.

Dark, high-contrast plates in the same visual language as the rest of the
deck they get pasted into: near-black ground, a bone display face for the
numbers that matter, mono for data. Bars are colored by whether they sit
in a reasonable range, not by an escalating-danger scale: gray below the
40th percentile (a plan that size is overconfident) and gray again past
P95 (padding nobody asked for), with a green ramp in between that deepens
toward P85 — the buffer worth taking to a manager. The functions return
figures; saving is the caller's business.
"""
import io
import logging
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox

matplotlib.use("Agg")

# Fonts most machines won't have are listed first anyway (see MONO_STACK /
# DISPLAY_STACK) and matplotlib falls back gracefully; that fallback is
# expected, not a problem to surface on every single text draw.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

FIGSIZE = (11.0, 5.5)
DPI = 300

# Ground and text.
INK = "#0A0D14"       # plate background
INK_2 = "#121826"     # a touch lighter, for the resting (below-P50) bars
BONE = "#EDE7DA"      # primary text and the S-curve line
BONE_DIM = "#8A8375"  # ticks, axis captions — quiet on purpose

# Status colors. Gray reads as "outside the range worth discussing" in
# either direction; green is reserved for the one number worth acting
# on. Neither is the app's brand accent (that stays orange, for chrome
# only) — status color here means confidence, nothing else.
GRAY = "#5B6773"        # P50 and P95 — same neutral, both ends of the range
GREEN_LOW = "#2F6E5C"   # start of the ramp, just past P40
GREEN_PEAK = "#4ECBA5"  # P85 — the recommended buffer
GREEN_FADE = "#3A8F76"  # easing back toward gray between P85 and P95
VIOLET = "#A78BFA"      # PLAN — the number being interrogated, not a risk tier

LEVEL_COLORS = {50: GRAY, 85: GREEN_PEAK, 95: GRAY}
LEVEL_STATUS = {85: "RECOMMENDED"}  # 50 and 95 get no status word, just their P-level
BASELINE_COLOR = VIOLET

# All three are actually installed for this build; DejaVu is the safety net
# on a machine that has none of them.
DISPLAY_STACK = ["Big Shoulders", "Archivo Expanded", "DejaVu Sans", "sans-serif"]
MONO_STACK = ["Geist Mono", "IBM Plex Mono", "PT Mono", "DejaVu Sans Mono", "monospace"]


def _level_color(level: int) -> str:
    return LEVEL_COLORS.get(level, GRAY)


def _level_tag(level: int) -> str:
    """The short label drawn on a percentile line: just its P-level, or
    "STATUS · P-level" for the one level worth flagging."""
    status = LEVEL_STATUS.get(level)
    if status:
        return "{0} · P{1}".format(status, level)
    return "P{0}".format(level)


def _lerp_hex(color_a: str, color_b: str, t: float) -> str:
    """Blend two ``#rrggbb`` colors, clamping t to [0, 1]."""
    t = min(max(t, 0.0), 1.0)
    a = tuple(int(color_a[i:i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(color_b[i:i + 2], 16) for i in (1, 3, 5))
    mixed = tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))
    return "#{0:02x}{1:02x}{2:02x}".format(*mixed)


def _zone_color(value: float, p40: float, p85: float, p95: float) -> str:
    """Gray outside [P40, P95]; a green ramp peaking at P85 inside it."""
    if value < p40 or value > p95:
        return GRAY
    if value <= p85:
        span = max(p85 - p40, 1e-9)
        return _lerp_hex(GREEN_LOW, GREEN_PEAK, (value - p40) / span)
    span = max(p95 - p85, 1e-9)
    return _lerp_hex(GREEN_PEAK, GREEN_FADE, (value - p85) / span)


def _style(axes) -> None:
    """Strip the chart down to what carries information."""
    axes.figure.set_facecolor(INK)
    axes.set_facecolor(INK)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(BONE_DIM)
    axes.tick_params(labelsize=10.5, length=3, colors=BONE_DIM,
                     labelfontfamily=MONO_STACK)
    axes.grid(False)


# Vertical spacing for staggered labels, as a fraction of the plot's own
# height (via get_xaxis_transform, so it is independent of the data scale).
_LABEL_ROW_HEIGHT = 0.15
_LABEL_BASE = 1.05
_LABEL_PAD_PX = 16  # minimum breathing room between two adjacent labels


def _place_labels(
    figure: Figure, axes, items: List[Tuple[float, str, List[str]]]
) -> int:
    """Draw each item's label above the plot, staggering rows to avoid
    overlap.

    A fixed fraction-of-axis-width threshold breaks down as soon as labels
    carry text of very different lengths — a short "P50 175" label needs
    far less berth than "BOARDING · P85 / 201 WORKING DAYS". So instead of
    guessing, this places every label provisionally, forces one draw pass
    to get each one's real rendered width from the renderer, then greedily
    reassigns rows from those actual pixel extents plus a fixed margin.

    Args:
        figure: the figure the labels belong to (must support ``.canvas``).
        axes: the axes to draw the labels on, in x-data / y-axes-fraction
            coordinates.
        items: (value, color, lines) triples; lines are stacked as one
            multi-line label centered on value.

    Returns:
        The highest row index used (0 if every label fit on one row).
    """
    FigureCanvasAgg(figure)
    transform = axes.get_xaxis_transform()

    texts = []
    for value, color, lines in items:
        text = axes.text(value, _LABEL_BASE, "\n".join(lines), transform=transform,
                         color=color, fontsize=12, fontweight="bold",
                         ha="center", va="bottom", clip_on=False,
                         fontfamily=DISPLAY_STACK, linespacing=1.5)
        texts.append((value, text))

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    order = sorted(range(len(texts)), key=lambda i: texts[i][0])
    row_right_edge: List[float] = []
    row_of = {}
    for i in order:
        _, text = texts[i]
        bbox = text.get_window_extent(renderer=renderer)
        row = 0
        while row < len(row_right_edge) and bbox.x0 - _LABEL_PAD_PX < row_right_edge[row]:
            row += 1
        if row == len(row_right_edge):
            row_right_edge.append(bbox.x1)
        else:
            row_right_edge[row] = bbox.x1
        row_of[i] = row

    max_row = max(row_of.values(), default=0)
    for i, (value, text) in enumerate(texts):
        y = _LABEL_BASE + (max_row - row_of[i]) * _LABEL_ROW_HEIGHT
        text.set_position((value, y))
    return max_row


_CURVE_LABEL_PAD_PX = 20
_CURVE_LABEL_DX_PX = 42  # clears the open marker circle at 300 dpi
_CURVE_LABEL_DY_PX = 30  # lifts the block clear of the next marker down


def _place_curve_labels(
    figure: Figure, axes, entries: List[Tuple[int, float, str, List[str]]],
) -> None:
    """Label each marker on the S-curve, keeping every label clear of the
    others.

    Each label carries an opaque background the same color as the plate,
    so whatever runs behind it — the curve, another level's horizontal
    guide line — simply disappears under the text rather than needing to
    be dodged geometrically. The one collision that background can't fix
    is two labels' own boxes landing on top of each other, so a second
    pass still nudges any label whose top edge would run into the next
    one further up — never sideways, so the reading order (bottom to top,
    P50 to P95) stays intact.

    Args:
        entries: (level, value, color, lines), already sorted ascending
            by level.
    """
    FigureCanvasAgg(figure)
    trans = axes.transData
    inv = trans.inverted()

    def shift(x_data, y_data, dx_px, dy_px):
        x_px, y_px = trans.transform((x_data, y_data))
        return inv.transform((x_px + dx_px, y_px + dy_px))

    texts = []
    for level, value, color, lines in entries:
        # Anchored above and to the right of the marker, growing upward.
        x0, y0 = shift(value, level, _CURVE_LABEL_DX_PX, _CURVE_LABEL_DY_PX)
        text = axes.text(x0, y0, "\n".join(lines), color=color, fontsize=12,
                         fontweight="bold", fontfamily=DISPLAY_STACK,
                         linespacing=1.5, va="bottom", ha="left",
                         bbox=dict(boxstyle="square,pad=0.35", facecolor=INK,
                                  edgecolor="none"))
        texts.append(text)

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    # A label can reach far enough right to land on the *next* marker up
    # (P85's box has nowhere else to go but toward P95, only sixteen days
    # away) — pull its right edge back to clear that marker before
    # stacking the boxes vertically.
    horizontal_margin_px = 22
    for i in range(len(texts) - 1):
        bbox = texts[i].get_window_extent(renderer=renderer)
        next_value, next_level = entries[i + 1][1], entries[i + 1][0]
        limit_px = trans.transform((next_value, next_level))[0] - horizontal_margin_px
        if bbox.x1 > limit_px:
            x_data, y_data = texts[i].get_position()
            x_new, _ = shift(x_data, y_data, limit_px - bbox.x1, 0)
            texts[i].set_position((x_new, y_data))

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    # Clear the label below it, bottom to top.
    prev_top_px = None
    for text in texts:
        bbox = text.get_window_extent(renderer=renderer)
        if prev_top_px is not None and bbox.y0 < prev_top_px + _CURVE_LABEL_PAD_PX:
            dy_px = (prev_top_px + _CURVE_LABEL_PAD_PX) - bbox.y0
            x_data, y_data = text.get_position()
            _, y_new = shift(x_data, y_data, 0, dy_px)
            text.set_position((x_data, y_new))
            bbox = Bbox.from_extents(
                bbox.x0, bbox.y0 + dy_px, bbox.x1, bbox.y1 + dy_px)
        prev_top_px = bbox.y1


def _draw_risk_bars(
    axes, totals: np.ndarray, p85: Optional[float], p95: Optional[float]
) -> None:
    """Color each bar by whether it falls in the range worth trusting.

    Below P40 is gray — a plan that size is overconfident, not "on track".
    From there a green ramp deepens toward P85, the buffer worth defending
    to a manager, then eases back down through P95, past which it's gray
    again: technically safer, but padding nobody asked for. P40 is derived
    straight from ``totals`` since it isn't one of the levels the rest of
    the app tracks.
    """
    counts, edges = np.histogram(totals, bins=60)
    centers = (edges[:-1] + edges[1:]) / 2.0
    width = edges[1:] - edges[:-1]

    if p85 is None or p95 is None:
        colors = [GREEN_PEAK] * len(centers)
    else:
        p40 = float(np.percentile(totals, 40))
        colors = [_zone_color(c, p40, p85, p95) for c in centers]

    axes.bar(centers, counts, width=width * 0.96, color=colors,
             edgecolor=INK, linewidth=0.6)


def histogram(
    totals: np.ndarray,
    pctls: Dict[int, float],
    baseline: Optional[float] = None,
    baseline_probability: Optional[float] = None,
    unit_label: str = "working days",
) -> Figure:
    """Where the project is likely to land.

    Args:
        totals: simulated project totals.
        pctls: level to duration, from ``stats.percentiles``.
        baseline: the plan's own duration, drawn for comparison.
        baseline_probability: the plan's chance of success, as a percentage.
        unit_label: the unit shown on the x axis.
    """
    figure = Figure(figsize=FIGSIZE, dpi=DPI)
    axes = figure.subplots()

    _draw_risk_bars(axes, totals, pctls.get(85), pctls.get(95))
    _style(axes)
    axes.set_yticks([])
    axes.set_xlabel("TOTAL PROJECT DURATION, {0}".format(unit_label.upper()),
                    fontsize=10.5, color=BONE_DIM, labelpad=12,
                    fontfamily=MONO_STACK, fontweight="medium")

    items: List[Tuple[float, str, List[str]]] = []
    if baseline is not None:
        axes.axvline(baseline, color=BASELINE_COLOR, linewidth=1.6,
                     linestyle=(0, (5, 3)))
        lines = ["PLAN {0:.0f}".format(baseline)]
        if baseline_probability is not None:
            lines.append("{0:.0f}% LIKELY".format(baseline_probability))
        items.append((baseline, BASELINE_COLOR, lines))

    for level in sorted(pctls):
        value = pctls[level]
        axes.axvline(value, color=_level_color(level), linewidth=1.8)
        items.append((value, _level_color(level),
                     [_level_tag(level),
                      "{0:.0f} {1}".format(value, unit_label).upper()]))

    max_row = _place_labels(figure, axes, items)

    # Reserve enough headroom above the plot for the tallest stack of
    # labels, regardless of how many values ended up clustered together.
    bottom = 0.16
    needed = _LABEL_BASE + max_row * _LABEL_ROW_HEIGHT + 0.13
    height = (0.97 - bottom) / needed
    figure.subplots_adjust(top=bottom + height, bottom=bottom, left=0.05, right=0.98)
    return figure


def s_curve(
    totals: np.ndarray,
    pctls: Dict[int, float],
    date_labels: Optional[Dict[int, str]] = None,
    unit_label: str = "working days",
) -> Figure:
    """Cumulative probability of finishing by a given duration.

    Args:
        totals: simulated project totals.
        pctls: level to duration, from ``stats.percentiles``.
        date_labels: optional level to formatted finish date.
        unit_label: the unit shown on the x axis.
    """
    figure = Figure(figsize=FIGSIZE, dpi=DPI)
    axes = figure.subplots()

    ordered = np.sort(np.asarray(totals, dtype=float))
    probability = np.arange(1, ordered.size + 1) / ordered.size * 100.0
    axes.plot(ordered, probability, color=BONE, linewidth=2.2, zorder=4)

    p85, p95 = pctls.get(85), pctls.get(95)
    if p85 is not None and p95 is not None:
        p40 = float(np.percentile(ordered, 40))
        below = ordered < p40
        ramp_up = (ordered >= p40) & (ordered < p85)
        ramp_down = (ordered >= p85) & (ordered < p95)
        above = ordered >= p95
        zones = ((below, GRAY), (ramp_up, GREEN_PEAK),
                 (ramp_down, GREEN_FADE), (above, GRAY))
        for mask, color in zones:
            axes.fill_between(ordered, probability, where=mask, color=color,
                              alpha=0.16, interpolate=True)
    else:
        axes.fill_between(ordered, probability, color=BONE, alpha=0.08)

    _style(axes)
    axes.set_ylim(0, 105)
    axes.set_yticks([0, 25, 50, 75, 100])
    axes.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    axes.set_xlabel("TOTAL PROJECT DURATION, {0}".format(unit_label.upper()),
                    fontsize=10.5, color=BONE_DIM, labelpad=12,
                    fontfamily=MONO_STACK, fontweight="medium")

    entries = []
    for level in sorted(pctls):
        value = pctls[level]
        color = _level_color(level)
        axes.plot([ordered[0], value], [level, level], color=color,
                  linewidth=0.9, linestyle=(0, (2, 3)))
        axes.plot([value], [level], marker="o", markersize=8, color=color,
                  markerfacecolor=INK, markeredgewidth=2.4, zorder=5)
        lines = [_level_tag(level),
                 "{0:.0f} {1}".format(value, unit_label).upper()]
        if date_labels and level in date_labels:
            lines.append(date_labels[level].upper())
        entries.append((level, value, color, lines))

    _place_curve_labels(figure, axes, entries)
    figure.tight_layout()
    return figure


def figure_to_png_bytes(figure: Figure) -> bytes:
    """Render a figure to PNG bytes for a download button."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=DPI, bbox_inches="tight",
                   facecolor=INK)
    return buffer.getvalue()
