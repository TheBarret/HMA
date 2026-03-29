"""
HMA Topology Visualizer
NATO-style cartographic output for Layer 0-3 pipeline results.

Layout:
    Left column  — four mini-plots (Layer 0, 1, 2 diagnostics + feature counts)
    Right panel  — full topology map, NATO style, Layer 0 hillshade as base
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LightSource
from scipy.ndimage import gaussian_filter
from typing import List, Dict, Optional
from pathlib import Path

from core import (
    PipelineConfig,
    PeakFeature, RidgeFeature, ValleyFeature, SaddleFeature, FlatZoneFeature,
    TerrainFeature
)

# =============================================================================
# NATO Palette
# =============================================================================

NATO = {
    'paper':        '#e8dfc8',   # map paper base
    'paper_dark':   '#d4c9a8',   # slightly darker for contrast panels
    'contour':      '#8b7355',   # brown contour lines
    'water':        '#a8c8e8',   # water bodies
    'water_deep':   '#6ba3c8',   # deeper water
    'lowland':      '#c8d8a8',   # low elevation fill
    'highland':     '#b8a878',   # high elevation fill
    'peak_marker':  '#1a1a1a',   # black peak triangle
    'ridge_line':   '#2c1810',   # dark brown ridge spine
    'valley_line':  '#4a6b8a',   # muted blue valley spine
    'saddle_mark':  '#5c3d1e',   # brown saddle marker
    'flat_fill':    '#c8d4a0',   # olive flat zone
    'flat_edge':    '#7a8c5a',   # flat zone border
    'grid':         '#b8a878',   # grid lines
    'text':         '#1a1a1a',   # primary text
    'text_muted':   '#5c4a2a',   # secondary text
    'border':       '#2c1810',   # frame border
    'danger':       '#8b1a1a',   # for cliffs / impassable
    'highlight':    '#c8781e',   # accent / callout
}

# =============================================================================
# Hillshade helper
# =============================================================================

def make_hillshade(elevation: np.ndarray, 
                   azimuth: float = 315.0,
                   altitude: float = 45.0,
                   exaggeration: float = 3.0) -> np.ndarray:
    """Generate hillshade from elevation array."""
    ls = LightSource(azdeg=azimuth, altdeg=altitude)
    return ls.hillshade(elevation, vert_exag=exaggeration, dx=1, dy=1)


def make_nato_terrain_cmap():
    """NATO-style elevation colormap: water → lowland → highland."""
    colors = [
        (0.00, NATO['water_deep']),
        (0.05, NATO['water']),
        (0.12, '#d4d8b0'),
        (0.30, NATO['lowland']),
        (0.55, '#c8b87a'),
        (0.75, NATO['highland']),
        (0.90, '#a89060'),
        (1.00, '#e8e0d0'),
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'nato_terrain',
        [(v, c) for v, c in colors]
    )
    return cmap


# =============================================================================
# Mini plot helpers (left column)
# =============================================================================

def _mini_frame(ax, title: str):
    """Apply consistent NATO mini-panel styling."""
    ax.set_facecolor(NATO['paper_dark'])
    ax.set_title(title, fontsize=7, fontweight='bold', color=NATO['text'],
                 fontfamily='monospace', pad=3)
    for spine in ax.spines.values():
        spine.set_edgecolor(NATO['border'])
        spine.set_linewidth(0.8)
    ax.tick_params(labelsize=5, colors=NATO['text_muted'])


def plot_layer0_elevation(ax, heightmap_data: np.ndarray):
    """Mini: raw elevation greyscale."""
    ax.imshow(heightmap_data, cmap='gray', interpolation='bilinear')
    _mini_frame(ax, 'L0 · ELEVATION')
    ax.set_xticks([])
    ax.set_yticks([])


def plot_layer1_slope(ax, slope: np.ndarray):
    """Mini: slope magnitude."""
    cmap = plt.cm.YlOrRd
    im = ax.imshow(slope, cmap=cmap, vmin=0, vmax=60, interpolation='bilinear')
    _mini_frame(ax, 'L1 · SLOPE °')
    ax.set_xticks([])
    ax.set_yticks([])
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(labelsize=4, colors=NATO['text_muted'])
    cb.outline.set_edgecolor(NATO['border'])


def plot_layer2_curvature(ax, mean_curvature: np.ndarray):
    """Mini: mean curvature, diverging around zero."""
    clipped = np.clip(mean_curvature, -0.02, 0.02)
    im = ax.imshow(clipped, cmap='RdBu_r', interpolation='bilinear')
    _mini_frame(ax, 'L2 · CURVATURE')
    ax.set_xticks([])
    ax.set_yticks([])
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(labelsize=4, colors=NATO['text_muted'])
    cb.outline.set_edgecolor(NATO['border'])


def plot_layer3_summary(ax, features: List[TerrainFeature]):
    """Mini: feature count bar chart."""
    from collections import Counter
    type_map = {
        PeakFeature:     'PEAKS',
        RidgeFeature:    'RIDGES',
        ValleyFeature:   'VALLEYS',
        SaddleFeature:   'SADDLES',
        FlatZoneFeature: 'FLAT',
    }
    counts = Counter()
    for f in features:
        label = type_map.get(type(f), 'OTHER')
        counts[label] += 1

    labels = ['PEAKS', 'RIDGES', 'VALLEYS', 'SADDLES', 'FLAT']
    bar_colors = ['#1a1a1a', '#2c1810', '#4a6b8a', '#5c3d1e', '#7a8c5a']
    values = [counts.get(l, 0) for l in labels]

    bars = ax.barh(labels, values, color=bar_colors, height=0.55, edgecolor=NATO['border'], linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(val), va='center', fontsize=5, color=NATO['text'],
                fontfamily='monospace')

    ax.set_facecolor(NATO['paper_dark'])
    ax.set_xlim(0, max(values) * 1.25 if values else 10)
    _mini_frame(ax, 'L3 · FEATURE COUNT')
    ax.tick_params(labelsize=5, colors=NATO['text_muted'])
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)


# =============================================================================
# Main topology map
# =============================================================================

def plot_topology_map(ax, heightmap_data: np.ndarray, features: List[TerrainFeature],
                      cell_size: float = 2.0):
    """
    Full NATO-style topology map.
    Base: hillshaded elevation with terrain colormap.
    Overlays: peaks, ridges, valleys, saddles, flat zones.
    """
    H, W = heightmap_data.shape

    # --- Base: terrain color + hillshade blend ---
    hillshade = make_hillshade(heightmap_data, exaggeration=4.0)
    nato_cmap = make_nato_terrain_cmap()
    #np.ptp(arr, ...)
    #norm_elev = (heightmap_data - heightmap_data.min()) / (heightmap_data.ptp() + 1e-8)
    norm_elev = (heightmap_data - heightmap_data.min()) / (np.ptp(heightmap_data) + 1e-8)
    terrain_rgb = nato_cmap(norm_elev)[:, :, :3]

    # Blend terrain color with hillshade (multiply mode)
    hs_3ch = np.stack([hillshade] * 3, axis=-1)
    blended = np.clip(terrain_rgb * (0.5 + 0.6 * hs_3ch), 0, 1)

    ax.imshow(blended, interpolation='bilinear', origin='upper',
              extent=[0, W * cell_size, H * cell_size, 0])

    # --- Contour lines (thin, brown, NATO style) ---
    elev_m = heightmap_data  # already in meters from pipeline
    levels = np.arange(
        np.percentile(elev_m[elev_m > 0], 5) if np.any(elev_m > 0) else elev_m.min(),
        elev_m.max(),
        5.0   # 5m contour interval
    )
    if len(levels) > 1:
        xs = np.linspace(0, W * cell_size, W)
        ys = np.linspace(0, H * cell_size, H)
        # Major every 25m
        major_levels = levels[::5]
        minor_levels = levels[~np.isin(levels, major_levels)]

        ax.contour(xs, ys, elev_m, levels=minor_levels,
                   colors=NATO['contour'], linewidths=0.2, alpha=0.45)
        ax.contour(xs, ys, elev_m, levels=major_levels,
                   colors=NATO['contour'], linewidths=0.55, alpha=0.65)

    # --- Flat zones (filled polygons, lowest layer) ---
    flat_zones  = [f for f in features if isinstance(f, FlatZoneFeature)]
    for fz in flat_zones:
        bounds = fz.metadata.get('bounds')
        if bounds:
            x0, x1, y0, y1 = bounds
            rect = mpatches.Rectangle(
                (x0 * cell_size, y0 * cell_size),
                (x1 - x0) * cell_size,
                (y1 - y0) * cell_size,
                linewidth=0.8,
                edgecolor=NATO['flat_edge'],
                facecolor=NATO['flat_fill'],
                alpha=0.28,
                linestyle='--'
            )
            ax.add_patch(rect)

    # --- Valley spines ---
    valleys = [f for f in features if isinstance(f, ValleyFeature)]
    for valley in valleys:
        if valley.spine_points and len(valley.spine_points) > 1:
            xs_v = [p[0] * cell_size for p in valley.spine_points]
            ys_v = [p[1] * cell_size for p in valley.spine_points]
            conf = valley.metadata.get('confidence', 0.5)
            lw = 0.6 + conf * 0.8
            ax.plot(xs_v, ys_v,
                    color=NATO['valley_line'], linewidth=lw,
                    alpha=0.75, solid_capstyle='round', solid_joinstyle='round',
                    zorder=3)
            # Tick marks along valley (drainage direction indicator)
            if len(valley.spine_points) > 4:
                step = max(1, len(valley.spine_points) // 4)
                for k in range(0, len(valley.spine_points) - 1, step):
                    mx = (valley.spine_points[k][0] + valley.spine_points[k+1][0]) / 2 * cell_size
                    my = (valley.spine_points[k][1] + valley.spine_points[k+1][1]) / 2 * cell_size
                    ax.plot(mx, my, 's', color=NATO['valley_line'],
                            markersize=1.0, alpha=0.6, zorder=3)

    # --- Ridge spines ---
    ridges = [f for f in features if isinstance(f, RidgeFeature)]
    for ridge in ridges:
        if ridge.spine_points and len(ridge.spine_points) > 1:
            xs_r = [p[0] * cell_size for p in ridge.spine_points]
            ys_r = [p[1] * cell_size for p in ridge.spine_points]
            conf = ridge.metadata.get('confidence', 0.5)
            lw = 0.7 + conf * 1.0
            ax.plot(xs_r, ys_r,
                    color=NATO['ridge_line'], linewidth=lw,
                    alpha=0.85, solid_capstyle='round', solid_joinstyle='round',
                    zorder=4)

    # --- Saddles (bowtie glyph) ---
    saddles = [f for f in features if isinstance(f, SaddleFeature)]
    for saddle in saddles:
        cx, cy = saddle.centroid
        conf = saddle.metadata.get('confidence', 0.3)
        size = 2.5 + conf * 2.0
        # Two small opposing triangles = bowtie
        ax.plot(cx * cell_size, cy * cell_size,
                marker=(4, 0, 45),   # rotated square = diamond
                markersize=size,
                color=NATO['saddle_mark'],
                alpha=0.7,
                zorder=5,
                markeredgewidth=0.3,
                markeredgecolor=NATO['border'])

    # --- Peaks (NATO triangle marker + prominence label) ---
    peaks = [f for f in features if isinstance(f, PeakFeature)]
    # Sort by prominence so larger peaks render on top
    peaks_sorted = sorted(peaks, key=lambda p: p.prominence)
    for peak in peaks_sorted:
        cx, cy = peak.centroid
        prom = peak.prominence
        # Size by prominence: minor=4, major=9
        size = np.clip(4.0 + (prom / 15.0) * 5.0, 4, 10)
        # Triangle up = peak
        ax.plot(cx * cell_size, cy * cell_size,
                marker='^',
                markersize=size,
                color=NATO['peak_marker'],
                zorder=6,
                markeredgewidth=0.4,
                markeredgecolor='#ffffff',
                alpha=0.92)
        # Label only significant peaks
        if prom > 20.0:
            elev = peak.metadata.get('elevation', 0)
            ax.annotate(
                f"{elev:.0f}m",
                xy=(cx * cell_size, cy * cell_size),
                xytext=(4, -6),
                textcoords='offset points',
                fontsize=4.5,
                fontfamily='monospace',
                color=NATO['text'],
                fontweight='bold',
                zorder=7
            )

    # --- Map frame and grid ---
    ax.set_facecolor(NATO['water'])
    ax.set_xlim(0, W * cell_size)
    ax.set_ylim(H * cell_size, 0)

    # Grid
    grid_step = 100  # meters
    x_ticks = np.arange(0, W * cell_size, grid_step)
    y_ticks = np.arange(0, H * cell_size, grid_step)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.grid(True, color=NATO['grid'], linewidth=0.25, alpha=0.5, linestyle=':')
    ax.tick_params(labelsize=5, colors=NATO['text_muted'])

    for spine in ax.spines.values():
        spine.set_edgecolor(NATO['border'])
        spine.set_linewidth(1.2)

    # --- Legend ---
    legend_elements = [
        mlines.Line2D([0], [0], marker='^', color='w',
                      markerfacecolor=NATO['peak_marker'], markersize=6,
                      label='Peak', markeredgecolor='white', markeredgewidth=0.3),
        mlines.Line2D([0], [0], color=NATO['ridge_line'], linewidth=1.2,
                      label='Ridge'),
        mlines.Line2D([0], [0], color=NATO['valley_line'], linewidth=1.2,
                      label='Valley'),
        mlines.Line2D([0], [0], marker=(4, 0, 45), color='w',
                      markerfacecolor=NATO['saddle_mark'], markersize=5,
                      label='Saddle'),
        mpatches.Patch(facecolor=NATO['flat_fill'], edgecolor=NATO['flat_edge'],
                       linestyle='--', linewidth=0.8, alpha=0.6, label='Flat Zone'),
    ]
    leg = ax.legend(
        handles=legend_elements,
        loc='lower right',
        fontsize=5,
        framealpha=0.88,
        facecolor=NATO['paper'],
        edgecolor=NATO['border'],
        title='FEATURES',
        title_fontsize=5,
        handlelength=1.5,
    )
    leg.get_title().set_fontfamily('monospace')
    leg.get_title().set_color(NATO['text'])


# =============================================================================
# Stamp block (bottom-right corner of topology map)
# =============================================================================

def _draw_stamp(fig, ax_topo, map_name: str, cell_size: float,
                shape: tuple, feature_count: int):
    """Draw a NATO-style map stamp in the corner of the topology axes."""
    H, W = shape
    stamp_text = (
        f"HMA · TOPOLOGICAL SURVEY\n"
        f"SCALE 1:{int(cell_size * 1000)}  CELL {cell_size}m/px\n"
        f"GRID {W*cell_size:.0f}m × {H*cell_size:.0f}m\n"
        f"FEATURES · {feature_count}  |  LAYER 3"
    )
    ax_topo.text(
        0.01, 0.01, stamp_text,
        transform=ax_topo.transAxes,
        fontsize=4.2, fontfamily='monospace',
        color=NATO['text_muted'],
        verticalalignment='bottom',
        bbox=dict(boxstyle='square,pad=0.4', facecolor=NATO['paper'],
                  edgecolor=NATO['border'], alpha=0.88, linewidth=0.7)
    )


# =============================================================================
# Master render function
# =============================================================================

def render(bundle: dict, features: list, map_name: str = "UNKNOWN",
           save_path: Optional[str] = None, dpi: int = 180):
    """
    Render the full HMA visualization.

    Args:
        bundle:     Pipeline bundle dict (heightmap, slope, aspect, curvature, ...)
        features:   Layer 3 feature list
        map_name:   Label for the map (filename stem)
        save_path:  If given, save to this path instead of showing
        dpi:        Output resolution
    """

    heightmap   = bundle['heightmap']
    slope       = bundle['slope']
    curvature   = bundle['curvature']
    elev        = heightmap.data
    cell_size   = heightmap.config.horizontal_scale
    H, W        = elev.shape

    # --- Figure layout ---
    # Left column: 4 mini plots stacked
    # Right panel: topology map (wider, taller)
    mini_w = 1.0          # relative width units
    topo_w = 3.2
    fig_w  = 14.0
    fig_h  = 10.0

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=NATO['paper'])
    fig.patch.set_facecolor(NATO['paper'])

    gs = GridSpec(
        4, 2,
        figure=fig,
        width_ratios=[mini_w, topo_w],
        height_ratios=[1, 1, 1, 1],
        hspace=0.38,
        wspace=0.08,
        left=0.04, right=0.97,
        top=0.93, bottom=0.05
    )

    ax_l0   = fig.add_subplot(gs[0, 0])
    ax_l1   = fig.add_subplot(gs[1, 0])
    ax_l2   = fig.add_subplot(gs[2, 0])
    ax_l3   = fig.add_subplot(gs[3, 0])
    ax_topo = fig.add_subplot(gs[:, 1])   # spans all 4 rows

    # --- Mini plots ---
    plot_layer0_elevation(ax_l0, elev)
    plot_layer1_slope(ax_l1, slope)
    plot_layer2_curvature(ax_l2, curvature)
    plot_layer3_summary(ax_l3, features)

    # --- Main topology map ---
    plot_topology_map(ax_topo, elev, features, cell_size=cell_size)
    ax_topo.set_title(
        f'TOPOLOGICAL SURVEY  ·  {map_name.upper()}',
        fontsize=9, fontweight='bold', fontfamily='monospace',
        color=NATO['text'], pad=6
    )

    # --- Stamp ---
    _draw_stamp(fig, ax_topo, map_name, cell_size, (H, W), len(features))

    # --- Figure title ---
    fig.suptitle(
        'HEIGHTMAP ANALYSIS  ·  LAYER 0–3  ·  STRUCTURAL DECOMPOSITION',
        fontsize=8, fontfamily='monospace', color=NATO['text_muted'],
        y=0.975
    )

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor=NATO['paper'])
        print(f"[Visualizer] Saved → {save_path}")
    else:
        plt.show()

    plt.close(fig)