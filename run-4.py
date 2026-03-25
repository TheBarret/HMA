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
    PeakFeature, RidgeFeature, ValleyFeature, SaddleFeature, FlatZoneFeature,
    AnalyzedTerrain
    )
from calibration import Layer0_Calibration
from lgeometry import Layer1_LocalGeometry
from rgeometry import Layer2_RegionalGeometry
from topological import Layer3_TopologicalFeatures
from relational import Layer4_Relational
from semantics import Layer5_Semantics


# Semantic marker styles: small, high-contrast, grayscale-friendly
SEMANTIC_MARKERS = {
    # Peaks
    "defensive_position": {"marker": "^", "color": "crimson", "size": 40, "label": "Defensive"},
    "observation_post":   {"marker": "*", "color": "cyan",    "size": 16, "label": "Observation"},
    "major_peak":         {"marker": "P", "color": "red",     "size": 35, "label": "Major Peak"},
    "minor_peak":         {"marker": "^", "color": "salmon",  "size": 25, "label": "Minor Peak"},
    
    # Ridges
    "defensive_cover":    {"marker": "-", "color": "orange",  "size": 2,  "label": "Cover Ridge", "linewidth": 1.5},
    "exposed_crest":      {"marker": "-", "color": "coral",   "size": 1,  "label": "Exposed Ridge", "linewidth": 0.8},
    
    # Valleys
    "ambush_potential":   {"marker": "v", "color": "steelblue", "size": 35, "label": "Ambush Zone"},
    "major_drainage":     {"marker": "v", "color": "dodgerblue", "size": 30, "label": "Major Drainage"},
    "minor_drainage":     {"marker": "v", "color": "skyblue", "size": 20, "label": "Minor Drainage"},
    
    # Saddles
    "chokepoint":         {"marker": "X", "color": "cyan",    "size": 45, "label": "Chokepoint"},
    "high_pass":          {"marker": "d", "color": "lightcyan", "size": 25, "label": "High Pass"},
    "low_pass":           {"marker": "d", "color": "azure",   "size": 20, "label": "Low Pass"},
    
    # Flat Zones
    "major_assembly_area":{"marker": "s", "color": "limegreen", "size": 60, "label": "Major Assembly", "alpha": 0.4},
    "assembly_area":      {"marker": "s", "color": "yellowgreen", "size": 40, "label": "Assembly", "alpha": 0.5},
    "small_clearing":     {"marker": ".", "color": "green",   "size": 15, "label": "Clearing"},
}

# Grayscale colormap for semantic base map
SEMANTIC_BASE_CMAP = "gray_r"  # Reversed gray: dark=low, light=high

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
    """Run full pipeline including Layer 5 semantics, return outputs per layer."""
    from PIL import Image

    raw = np.array(Image.open(png_path).convert('L'))

    # Layer 0: Calibration
    layer0 = Layer0_Calibration(config)
    heightmap = layer0.execute(raw, NormalizationConfig(
        horizontal_scale=config.horizontal_scale,
        vertical_scale=config.vertical_scale,
        sea_level_offset=config.sea_level_offset,
    ))

    # Layer 1: Local Geometry
    layer1 = Layer1_LocalGeometry(config)
    slope_aspect = layer1.execute(heightmap)

    # Layer 2: Regional Geometry
    layer2 = Layer2_RegionalGeometry(config)
    curvature = layer2.execute(heightmap)

    # Layer 3: Topological Features
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

    # Layer 4: Relational
    layer4 = Layer4_Relational(config)
    relational = layer4.execute(bundle)
    bundle.update(relational)  # Merge graphs into bundle

    # NEW: Layer 5: Semantics
    layer5 = Layer5_Semantics(config)
    analyzed = layer5.execute(bundle)

    return {
        'heightmap': heightmap.data,
        'slope': slope_aspect['slope'],
        'aspect': slope_aspect['aspect'],
        'curvature_type': curvature['curvature_type'],
        'features': features,
        'relational': relational,
        'analyzed': analyzed,  # NEW: Full semantic output
        'semantic_index': analyzed.semantic_index,  # NEW: Queryable index
    }


# ─────────────────────────────────────────────────────────────────────────────
# Layer Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_layer0(ax: plt.Axes, data: Dict, marker_scale: float = 1.0):
    """Layer 0: Heightmap"""
    im = ax.imshow(data['heightmap'], cmap='terrain')
    ax.set_title('Heightmap', fontsize=9)
    ax.set_xlabel('X', fontsize=7)
    ax.set_ylabel('Y', fontsize=7)
    
    # Smaller colorbar for compact plots
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, label='Elevation (m)')
    cbar.ax.tick_params(labelsize=6)
    
    # Reduce tick label size
    ax.tick_params(labelsize=6)


def plot_layer1(ax: plt.Axes, data: Dict):
    """Layer 1: Slope"""
    im = ax.imshow(data['slope'], cmap=SLOPE_CMAP)
    ax.set_title('Slopes', fontsize=10)
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
    ax.set_title('Curvature Type', fontsize=10)
    ax.set_xlabel('X (px)')
    ax.set_ylabel('Y (px)')


def plot_layer3(ax: plt.Axes, data: Dict, marker_scale: float = 1.0):
    """Layer 3: Features (peaks, ridges, valleys, saddles, flats)"""
    hm = data['heightmap']
    features = data['features']
    
    # Base heightmap with reduced alpha for small plots
    ax.imshow(hm, cmap='terrain', alpha=0.6, interpolation='bilinear')
    
    # Scale markers down for small plots
    peak_size = 25 * marker_scale
    saddle_size = 18 * marker_scale
    ridge_width = 0.8 * marker_scale
    valley_width = 0.6 * marker_scale
    
    for f in features:
        cx, cy = f.centroid
        
        if isinstance(f, PeakFeature):
            ax.scatter(cx, cy, c='crimson', marker='^', s=peak_size,
                       edgecolors='white', linewidths=0.5, zorder=10, alpha=0.9)
        
        elif isinstance(f, RidgeFeature):
            if hasattr(f, 'spine_points') and f.spine_points:
                xs = [p[0] for p in f.spine_points]
                ys = [p[1] for p in f.spine_points]
                ax.plot(xs, ys, color='orange', linewidth=ridge_width, 
                        alpha=0.8, zorder=7)
        
        elif isinstance(f, ValleyFeature):
            if hasattr(f, 'spine_points') and f.spine_points:
                xs = [p[0] for p in f.spine_points]
                ys = [p[1] for p in f.spine_points]
                ax.plot(xs, ys, color='steelblue', linewidth=valley_width, 
                        alpha=0.7, zorder=6)
        
        elif isinstance(f, SaddleFeature):
            ax.scatter(cx, cy, c='cyan', marker='X', s=saddle_size,
                       edgecolors='white', linewidths=0.4, zorder=9, alpha=0.9)
        
        elif isinstance(f, FlatZoneFeature):
            if 'bounds' in f.metadata:
                xmin, xmax, ymin, ymax = f.metadata['bounds']
                rect = plt.Rectangle(
                    (xmin, ymin), xmax - xmin, ymax - ymin,
                    facecolor='limegreen', alpha=0.2, edgecolor='white', 
                    linewidth=0.3, zorder=5
                )
                ax.add_patch(rect)
    
    ax.set_title('Features', fontsize=9)
    ax.set_xlabel('X', fontsize=7)
    ax.set_ylabel('Y', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(False)


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
        "Summary",
        f"  Peaks        : {peaks}",
        f"  Ridges       : {ridges}",
        f"  Valleys      : {valleys}",
        f"  Saddles      : {saddles}",
        f"  Flat Zones   : {flats}",
        f"  Visibility   : {vis_nodes} nodes, {vis_edges} edges",
        f"  Connectivity : {conn_nodes} nodes, {conn_edges} edges",
        f"  Flow Network : {flow_edges} connections",
        f"  Watersheds   : {len(watersheds)} basins",
    ]

    ax.axis('off')
    ax.text(0.1, 0.5, '\n'.join(summary), transform=ax.transAxes, fontsize=9,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))


# -----------------------------------------------------------------------------------------

def plot_flow_direction(ax: plt.Axes, data: Dict):
    """
    Show flow direction as a quiver plot - FAST.
    Arrows show the direction water would flow.
    """
    
    
    #im = ax.imshow(data['heightmap'], cmap='terrain')
    #ax.set_title('Layer 0: Heightmap', fontsize=10)
    #ax.set_xlabel('X (px)')
    #ax.set_ylabel('Y (px)')
    #plt.colorbar(im, ax=ax, shrink=0.7, label='Elevation (m)')
    
    hm = data['heightmap']
    relational = data.get('relational', {})
    flow_network = relational.get('flow_network', {})
    features = data['features']
    
    # Base map
    #ax.imshow(hm.data, cmap='terrain', interpolation='bilinear', alpha=0.7)
    im = ax.imshow(hm.data, cmap='terrain')
    
    # Get flow directions at feature points
    positions = {f.feature_id: f.centroid for f in features}
    
    # Collect flow vectors
    flow_vectors = []
    for src_id, targets in flow_network.items():
        if src_id not in positions or not targets:
            continue
        x1, y1 = positions[src_id]
        # For each flow connection, draw an arrow
        for tgt_id in targets[:3]:  # Limit to 3 per source
            if tgt_id not in positions:
                continue
            x2, y2 = positions[tgt_id]
            flow_vectors.append((x1, y1, x2 - x1, y2 - y1))
    
    # Draw arrows (limited count for performance)
    max_arrows = 50
    if len(flow_vectors) > max_arrows:
        import random
        flow_vectors = random.sample(flow_vectors, max_arrows)
    
    for x, y, dx, dy in flow_vectors:
        ax.arrow(x, y, dx*0.8, dy*0.8, head_width=5, head_length=5, 
                 fc='red', ec='red', alpha=0.7, width=0.5, zorder=8)
    
    # Highlight valley features (where water collects)
    valleys = [f for f in features if isinstance(f, ValleyFeature)]
    for v in valleys[:20]:
        cx, cy = v.centroid
        ax.scatter(cx, cy, c='blue', marker='v', s=60, edgecolors='white', 
                  linewidths=1, zorder=10, alpha=0.8)
    
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title(f'Flow Directions', fontsize=9)
    ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    
def plot_visibility_density(ax: plt.Axes, data: Dict):
    """Show visibility connectivity as a density heatmap instead of lines."""
    hm = data['heightmap']
    relational = data.get('relational', {})
    visibility = relational.get('visibility_graph', {})
    features = data['features']
    
    # Create density map
    density = np.zeros(hm.data.shape)
    
    positions = {f.feature_id: f.centroid for f in features}
    
    # For each edge, add density along the line
    for src_id, targets in visibility.items():
        if src_id not in positions:
            continue
        x1, y1 = positions[src_id]
        for tgt_id in targets:
            if tgt_id not in positions:
                continue
            x2, y2 = positions[tgt_id]
            
            # Bresenham line to add density
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx - dy
            x, y = x1, y1
            
            while (x, y) != (x2, y2):
                if 0 <= y < density.shape[0] and 0 <= x < density.shape[1]:
                    density[y, x] += 1
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x += sx
                if e2 < dx:
                    err += dx
                    y += sy
    
    # Plot density heatmap
    if len(density) > 0:
        ax.imshow(hm.data, cmap='terrain', interpolation='bilinear', alpha=0.5)
        im = ax.imshow(density, cmap='hot', alpha=0.6, interpolation='bilinear', vmax=np.percentile(density[density > 0], 90))
    
    # Add colorbar
    plt.colorbar(im, ax=ax, shrink=0.8, label='Connection Density')
    
    # Draw only major features
    peaks = [f for f in features if isinstance(f, PeakFeature)]
    peaks.sort(key=lambda p: p.prominence, reverse=True)
    for p in peaks[:20]:
        cx, cy = p.centroid
        ax.scatter(cx, cy, c='red', marker='^', s=80, edgecolors='white', zorder=10)
    
    ax.set_title('Visibility Density', fontsize=9)
    ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)

def plot_connectivity_density(ax: plt.Axes, data: Dict):
    hm = data['heightmap']
    relational = data.get('relational', {})
    connectivity = relational.get('connectivity_graph', {})
    features = data['features']
    
    # Create density array
    density = np.zeros(hm.data.shape)
    positions = {f.feature_id: f.centroid for f in features}
    
    # Count connectivity edges per pixel
    for src_id, targets in connectivity.items():
        if src_id not in positions:
            continue
        x1, y1 = positions[src_id]
        if 0 <= y1 < density.shape[0] and 0 <= x1 < density.shape[1]:
            density[y1, x1] += len(targets)
    
    # Plot with slope shading (more informative)
    slope = data.get('slope', None)
    if slope is not None:
        #ax.imshow(slope, cmap='YlOrRd', alpha=0.5, interpolation='bilinear')
        ax.imshow(slope, cmap='terrain', alpha=0.5, interpolation='bilinear')
    #else:
    #    ax.imshow(hm.data, cmap='terrain', interpolation='bilinear', alpha=0.5)
    
    #im = ax.imshow(density, cmap='Greens', alpha=0.7, interpolation='bilinear',
    #               vmin=0, vmax=np.percentile(density[density > 0], 90) if np.any(density > 0) else 1)
    
    #cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    #cbar.set_label('Connectivity', fontsize=7)
    
    # Add flat zones for context
    flats = [f for f in features if isinstance(f, FlatZoneFeature)]
    flats.sort(key=lambda fz: fz.area_pixels, reverse=True)
    for fz in flats[:10]:
        if 'bounds' in fz.metadata:
            x_min, x_max, y_min, y_max = fz.metadata['bounds']
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                  facecolor='none', edgecolor='green', alpha=0.5, linewidth=1)
            ax.add_patch(rect)
    
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('Traversability Density', fontsize=9)
    ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
# -----------------------------------------------------------------------------------------

def plot_semantic_layer(ax: plt.Axes, data: Dict, config: PipelineConfig):
    """
    Layer 5: Semantic interpretation on grayscale heightmap.
    - Base: grayscale elevation (no color ramp clutter)
    - Markers: small, semantic-type-coded symbols
    - Legend: compact, non-blocking, toggleable
    """
    heightmap = data['heightmap']
    analyzed = data.get('analyzed')
    semantic_index = data.get('semantic_index', {})
    
    if analyzed is None:
        ax.text(0.5, 0.5, "No semantic data", transform=ax.transAxes, 
                ha='center', va='center', fontsize=10, color='gray')
        return
    
    # Grayscale heightmap with subtle contours ---
    ax.imshow(heightmap, cmap='gray_r', interpolation='bilinear', alpha=0.9)
    
    # elevation context
    #levels = np.linspace(np.min(heightmap), np.max(heightmap), 8)
    #ax.contour(heightmap, levels=levels, colors='white', linewidths=0.3, alpha=0.3)
    
    # --- Plot semantic markers ---
    plotted_labels = set()  # Track for clean legend
    
    def plot_marker(x, y, style_dict, label):
        # Translate matplotlib-incompatible keys
        kwargs = {}
        for k, v in style_dict.items():
            if k == 'size':
                kwargs['s'] = v  # scatter uses 's' not 'size'
            elif k not in ['label']:
                kwargs[k] = v
        
        if label not in plotted_labels:
            kwargs['label'] = label
            plotted_labels.add(label)
        
        ax.scatter(x, y, **kwargs)
    
    # Defensive positions (peaks)
    for item in semantic_index.get('defensive_positions', []):
        cx, cy = item['centroid']
        style = SEMANTIC_MARKERS.get('defensive_position', {})
        plot_marker(cx, cy, style, style.get('label', 'Defensive'))
    
    # Observation posts
    for item in semantic_index.get('observation_posts', []):
        cx, cy = item['centroid']
        style = SEMANTIC_MARKERS.get('observation_post', {})
        plot_marker(cx, cy, style, style.get('label', 'Observation'))
    
    # Chokepoints (saddles)
    for item in semantic_index.get('chokepoints', []):
        cx, cy = item['centroid']
        style = SEMANTIC_MARKERS.get('chokepoint', {})
        plot_marker(cx, cy, style, style.get('label', 'Chokepoint'))
    
    # Assembly areas (flat zones)
    for item in semantic_index.get('assembly_areas', []):
        cx, cy = item['centroid']
        area = item.get('area_m2', 0)
        size = np.clip(area / 50, 20, 80)
        style = SEMANTIC_MARKERS.get('assembly_area', {}).copy()
        style['size'] = size  # Will be translated to 's' by plot_marker
        plot_marker(cx, cy, style, style.get('label', 'Assembly'))
    
    # Cover positions (ridges) - plot spine as LINE, not scatter
    for item in semantic_index.get('cover_positions', []):
        spine = item.get('spine', [])
        if spine:
            xs = [p[0] for p in spine]
            ys = [p[1] for p in spine]
            style = SEMANTIC_MARKERS.get('defensive_cover', {})
            label = style.get('label', 'Cover')
            
            # Extract line-specific properties
            line_kwargs = {}
            if 'color' in style:
                line_kwargs['color'] = style['color']
            if 'linewidth' in style:
                line_kwargs['linewidth'] = style['linewidth']
            if 'alpha' in style:
                line_kwargs['alpha'] = style['alpha']
            
            if label not in plotted_labels:
                ax.plot(xs, ys, label=label, **line_kwargs)
                plotted_labels.add(label)
            else:
                ax.plot(xs, ys, **line_kwargs)
    
    # Ambush positions (valleys) - plot spine as LINE
    for item in semantic_index.get('ambush_positions', []):
        spine = item.get('spine', [])
        if spine:
            cx, cy = item['centroid']
            style = SEMANTIC_MARKERS.get('ambush_potential', {})
            plot_marker(cx, cy, style, style.get('label', 'Ambush'))
    
    # --- Compact legend (bottom-right, semi-transparent) ---
    if plotted_labels:
        legend = ax.legend(
            loc='lower right', 
            fontsize=7, 
            frameon=True,
            framealpha=0.85,
            labelspacing=0.3,
            handlelength=1.2,
            borderpad=0.4,
            ncol=2  # Two columns to save vertical space
        )
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('gray')
    
    # --- Title and axis labels ---
    ax.set_title('Tactical Analysis', fontsize=10, fontweight='bold')
    ax.set_xlabel('X (px)', fontsize=8)
    ax.set_ylabel('Y (px)', fontsize=8)
    
    # --- Optional: semantic summary text box ---
    summary_lines = []
    for key in ['defensive_positions', 'observation_posts', 'chokepoints', 'assembly_areas']:
        count = len(semantic_index.get(key, []))
        if count > 0:
            label = key.replace('_', ' ').title()
            summary_lines.append(f"{label}: {count}")
    
    if summary_lines:
        summary_text = '\n'.join(summary_lines[:4])  # Limit to 4 lines
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
                fontsize=7, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7, edgecolor='gray'))
    
    ax.grid(False)  # No grid on semantic view
    
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
    Render full pipeline analysis with improved layout.
    
    Layout:
      Left column (30%):  L0, L1, L2, L3 (stacked vertically)
      Right column (70%): L5 Semantic Layer (full height)
    """
    print(f"Loading: {png_path}")
    config = PipelineConfig(game_type=GameType.ARMA_3)
    config.verbose = True

    data = run_pipeline(png_path, config)  # Includes Layer 5 now

    # Create figure with asymmetric columns: 1:3 width ratio
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(
        nrows=4, ncols=2,
        width_ratios=[1, 3],      # Left small, right large
        height_ratios=[1, 1, 1, 1],  # Equal height for left plots
        hspace=0.15, wspace=0.1   # Tight spacing
    )

    # --- Left column: Small layer plots (stacked) ---
    ax_l0 = fig.add_subplot(gs[0, 0])
    ax_l1 = fig.add_subplot(gs[1, 0])
    ax_l2 = fig.add_subplot(gs[2, 0])
    ax_l3 = fig.add_subplot(gs[3, 0])

    # --- Right column: Large Semantic Layer (spans all rows) ---
    ax_semantic = fig.add_subplot(gs[:, 1])

    # --- Plot each layer ---
    plot_layer0(ax_l0, data, marker_scale=0.5)
    #plot_layer1(ax_l1, data)
    plot_visibility_density(ax_l1, data)
    #plot_layer2(ax_l2, data)
    plot_flow_direction(ax_l2, data)
    plot_layer3(ax_l3, data, marker_scale=0.5)  # Smaller markers
    
    # Large semantic plot (full size markers)
    plot_semantic_layer(ax_semantic, data, config)

    # --- Global title ---
    plt.suptitle(f'Terrain Analysis: {Path(png_path).stem}', 
                 fontsize=14, fontweight='bold', y=0.995)

    # --- Tight layout to prevent clipping ---
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave room for suptitle

    # --- Save or show ---
    if output_name:
        plt.savefig(output_name, dpi=150, bbox_inches='tight', facecolor='white')
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