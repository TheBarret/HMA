"""
Visualization — Layers 0 through 4
One plot per layer. Fast. Clean. No heavy processing.

Layout:
  Row 0: Heightmap | Slope | Curvature Type
  Row 1: Features  | Relational Summary (text)
  Row 2: Stats Bar
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict

# Import pipeline layers
from core import (
    PipelineConfig, GameType, NormalizationConfig,
    PeakFeature, RidgeFeature, ValleyFeature, SaddleFeature, FlatZoneFeature
)
from calibration import Layer0_Calibration
from lgeometry import Layer1_LocalGeometry
from rgeometry import Layer2_RegionalGeometry
from topological import Layer3_TopologicalFeatures
from relational import Layer4_Relational


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CURVATURE_CMAP = ListedColormap([
    '#808080', '#F4A582', '#D73027', '#A6611A', '#2C7BB6', '#92C5DE'
])
CURVATURE_LABELS = ['FLAT', 'RIDGE', 'PEAK', 'SADDLE', 'DEPRESSION', 'VALLEY']
SLOPE_CMAP = 'YlOrRd'


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Execution
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(png_path: str, config: PipelineConfig) -> Dict:
    """Run full pipeline, return outputs per layer."""
    from PIL import Image

    raw = np.array(Image.open(png_path).convert('L'))

    layer0 = Layer0_Calibration(config)
    heightmap = layer0.execute(raw, NormalizationConfig(
        horizontal_scale=config.horizontal_scale,
        vertical_scale=config.vertical_scale,
        sea_level_offset=config.sea_level_offset,
    ))

    layer1 = Layer1_LocalGeometry(config)
    slope_aspect = layer1.execute(heightmap)

    layer2 = Layer2_RegionalGeometry(config)
    curvature = layer2.execute(heightmap)

    layer3 = Layer3_TopologicalFeatures(config)
    bundle = {
        'heightmap': heightmap,
        'slope': slope_aspect['slope'],
        'aspect': slope_aspect['aspect'],
        'curvature': curvature['curvature'],
        'gaussian_curvature': curvature['gaussian_curvature'],
        'curvature_type': curvature['curvature_type'],
    }
    features = layer3.execute(bundle)

    bundle['features'] = features
    layer4 = Layer4_Relational(config)
    relational = layer4.execute(bundle)

    return {
        'heightmap': heightmap.data,
        'slope': slope_aspect['slope'],
        'aspect': slope_aspect['aspect'],
        'curvature_type': curvature['curvature_type'],
        'features': features,
        'relational': relational,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Layer Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_layer0(ax: plt.Axes, data: Dict):
    """Layer 0: Heightmap"""
    im = ax.imshow(data['heightmap'], cmap='terrain')
    ax.set_title('Layer 0: Heightmap', fontsize=10)
    ax.set_xlabel('X (px)')
    ax.set_ylabel('Y (px)')
    plt.colorbar(im, ax=ax, shrink=0.7, label='Elevation (m)')


def plot_layer1(ax: plt.Axes, data: Dict):
    """Layer 1: Slope"""
    im = ax.imshow(data['slope'], cmap=SLOPE_CMAP)
    ax.set_title('Layer 1: Slope', fontsize=10)
    ax.set_xlabel('X (px)')
    plt.colorbar(im, ax=ax, shrink=0.7, label='Slope (°)')


def plot_layer2(ax: plt.Axes, data: Dict):
    """Layer 2: Curvature Type"""
    type_to_int = {
        'FLAT': 0, 'CYLINDRICAL_CONVEX': 1, 'CONVEX': 2,
        'SADDLE': 3, 'CONCAVE': 4, 'CYLINDRICAL_CONCAVE': 5,
    }
    numeric = np.vectorize(lambda t: type_to_int.get(t, 0))(data['curvature_type'])

    ax.imshow(numeric, cmap=CURVATURE_CMAP, interpolation='nearest', vmin=0, vmax=5)
    ax.set_title('Layer 2: Curvature Type', fontsize=10)
    ax.set_xlabel('X (px)')
    ax.set_ylabel('Y (px)')


def plot_layer3(ax: plt.Axes, data: Dict):
    """Layer 3: Features (peaks, ridges, valleys, saddles, flats)"""
    hm = data['heightmap']
    features = data['features']

    ax.imshow(hm, cmap='terrain', alpha=0.7)

    counts = {'Peak': 0, 'Ridge': 0, 'Valley': 0, 'Saddle': 0, 'Flat': 0}

    for f in features:
        cx, cy = f.centroid

        if isinstance(f, PeakFeature):
            ax.scatter(cx, cy, c='red', marker='^', s=40,
                       edgecolors='white', linewidths=0.8, zorder=10)
            counts['Peak'] += 1

        elif isinstance(f, RidgeFeature):
            if hasattr(f, 'spine_points') and f.spine_points:
                xs = [p[0] for p in f.spine_points]
                ys = [p[1] for p in f.spine_points]
                ax.plot(xs, ys, color='orange', linewidth=1, alpha=0.8, zorder=7)
                counts['Ridge'] += 1

        elif isinstance(f, ValleyFeature):
            if hasattr(f, 'spine_points') and f.spine_points:
                xs = [p[0] for p in f.spine_points]
                ys = [p[1] for p in f.spine_points]
                ax.plot(xs, ys, color='blue', linewidth=1, alpha=0.8, zorder=6)
                counts['Valley'] += 1

        elif isinstance(f, SaddleFeature):
            ax.scatter(cx, cy, c='cyan', marker='o', s=30,
                       edgecolors='white', linewidths=0.5, zorder=9)
            counts['Saddle'] += 1

        elif isinstance(f, FlatZoneFeature):
            if 'bounds' in f.metadata:
                xmin, xmax, ymin, ymax = f.metadata['bounds']
                rect = plt.Rectangle(
                    (xmin, ymin), xmax - xmin, ymax - ymin,
                    facecolor='green', alpha=0.3, edgecolor='white', linewidth=0.5
                )
                ax.add_patch(rect)
                counts['Flat'] += 1

    ax.set_title(f'Layer 3: Features (P:{counts["Peak"]} R:{counts["Ridge"]} '
                 f'V:{counts["Valley"]} S:{counts["Saddle"]} F:{counts["Flat"]})', fontsize=9)
    ax.set_xlabel('X (px)')
    ax.set_ylabel('Y (px)')


def plot_layer4(ax: plt.Axes, data: Dict):
    """Layer 4: Relational Summary (text only - fast, reliable)"""
    relational = data.get('relational', {})
    features = data.get('features', [])

    # Get counts
    peaks = len([f for f in features if isinstance(f, PeakFeature)])
    ridges = len([f for f in features if isinstance(f, RidgeFeature)])
    valleys = len([f for f in features if isinstance(f, ValleyFeature)])
    saddles = len([f for f in features if isinstance(f, SaddleFeature)])
    flats = len([f for f in features if isinstance(f, FlatZoneFeature)])

    # Get relational stats
    vis = relational.get('visibility_graph', {})
    conn = relational.get('connectivity_graph', {})
    flow = relational.get('flow_network', {})
    watersheds = relational.get('watersheds', {})

    vis_edges = sum(len(v) for v in vis.values()) // 2
    vis_nodes = len([v for v in vis.values() if v])
    conn_edges = sum(len(v) for v in conn.values()) // 2
    conn_nodes = len([v for v in conn.values() if v])
    flow_edges = sum(len(v) for v in flow.values())

    summary = [
        "══════════════════════════════════════════",
        "           LAYER 4: RELATIONAL",
        "══════════════════════════════════════════",
        "",
        "FEATURE COUNTS",
        "─────────────",
        f"  Peaks:      {peaks}",
        f"  Ridges:     {ridges}",
        f"  Valleys:    {valleys}",
        f"  Saddles:    {saddles}",
        f"  Flat Zones: {flats}",
        "",
        "GRAPH STATISTICS",
        "────────────────",
        f"  Visibility:    {vis_nodes} nodes, {vis_edges} edges",
        f"  Connectivity:  {conn_nodes} nodes, {conn_edges} edges",
        f"  Flow Network:  {flow_edges} connections",
        f"  Watersheds:    {len(watersheds)} basins",
    ]

    ax.axis('off')
    ax.text(0.1, 0.5, '\n'.join(summary), transform=ax.transAxes, fontsize=9,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))


def plot_stats_bar(ax: plt.Axes, data: Dict):
    """Full-width stats bar with key counts"""
    features = data['features']
    relational = data.get('relational', {})

    peaks = len([f for f in features if isinstance(f, PeakFeature)])
    ridges = len([f for f in features if isinstance(f, RidgeFeature)])
    valleys = len([f for f in features if isinstance(f, ValleyFeature)])
    saddles = len([f for f in features if isinstance(f, SaddleFeature)])
    flats = len([f for f in features if isinstance(f, FlatZoneFeature)])
    watersheds = len(relational.get('watersheds', {}))

    items = [
        ('PEAKS', peaks), ('RIDGES', ridges), ('VALLEYS', valleys),
        ('SADDLES', saddles), ('FLAT ZONES', flats), ('WATERSHEDS', watersheds)
    ]

    ax.axis('off')
    n = len(items)
    step = 1.0 / n

    for i, (label, val) in enumerate(items):
        x = i * step + step * 0.5
        ax.text(x, 0.65, label, transform=ax.transAxes,
                fontsize=8, color='gray', ha='center', va='center')
        ax.text(x, 0.35, str(val), transform=ax.transAxes,
                fontsize=14, fontweight='bold', ha='center', va='center')

        if i < n - 1:
            xd = (i + 1) * step
            ax.plot([xd, xd], [0.1, 0.9], color='lightgray', linewidth=0.6,
                    transform=ax.transAxes, clip_on=False)


# ─────────────────────────────────────────────────────────────────────────────
# Main Visualizer
# ─────────────────────────────────────────────────────────────────────────────

def visualize_pipeline(png_path: str, output_name: str = None):
    """
    Render full pipeline analysis.
    
    Layout:
      Row 0: Heightmap | Slope | Curvature Type
      Row 1: Features  | Relational Summary
      Row 2: Stats Bar (full width)
    """
    print(f"Loading: {png_path}")
    config = PipelineConfig(game_type=GameType.CUSTOM)
    config.verbose = True

    data = run_pipeline(png_path, config)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.0, 0.2], hspace=0.3, wspace=0.25)

    # Row 0: Heightmap, Slope, Curvature (using nested GridSpec for 3 cols)
    gs_top = GridSpec(1, 3, figure=fig, top=0.95, bottom=0.68, left=0.08, right=0.92, wspace=0.25)
    ax0 = fig.add_subplot(gs_top[0, 0])
    ax1 = fig.add_subplot(gs_top[0, 1])
    ax2 = fig.add_subplot(gs_top[0, 2])

    # Row 1: Features (left), Relational Summary (right)
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # Row 2: Stats bar (full width)
    ax5 = fig.add_subplot(gs[2, :])

    plot_layer0(ax0, data)
    plot_layer1(ax1, data)
    plot_layer2(ax2, data)
    plot_layer3(ax3, data)
    plot_layer4(ax4, data)
    plot_stats_bar(ax5, data)

    plt.suptitle(f'Pipeline Analysis: {Path(png_path).stem}', fontsize=14, fontweight='bold', y=0.98)

    if output_name:
        plt.savefig(output_name, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_name}")
    else:
        plt.show()

    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        png_path = sys.argv[1]
        output = sys.argv[2] if len(sys.argv) > 2 else f'{Path(png_path).stem}_analysis.png'
        visualize_pipeline(png_path, output)
    else:
        print('Usage: python visualize.py <heightmap.png> [output.png]')