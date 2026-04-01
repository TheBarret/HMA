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
from typing import List, Dict, Optional, Set
from pathlib import Path

from core import (
    PipelineConfig,
    PeakFeature, RidgeFeature, ValleyFeature, SaddleFeature, FlatZoneFeature,
    TerrainFeature, PixelCoord
)

# =============================================================================
# NATO Palette
# =============================================================================

NATO = {
    'paper':        '#e8dfc8',   # map paper base
    'paper_dark':   '#d4c9a8',   # slightly darker for contrast panels
    'contour':      '#8b7355',   # brown contour lines
    'water':        '#6396c9',   # water bodies
    'water_deep':   '#3c6d9e',   # deeper water
    'lowland':      '#c8d8a8',   # low elevation fill
    'highland':     '#b8a878',   # high elevation fill
    'peak_marker':  '#1a1a1a',   # black peak triangle
    'ridge_line':   '#2c1810',   # dark brown ridge spine
    'valley_line':  '#4a6b8a',   # muted blue valley spine
    'saddle_mark':  '#5c3d1e',   # brown saddle marker
    'flat_fill':    '#c8d4a0',   # olive flat zone
    'flat_edge':    '#b8a878',   # flat zone border
    'grid':         '#b8a878',   # grid lines
    'text':         '#1a1a1a',   # primary text
    'text_muted':   '#5c4a2a',   # secondary text
    'border':       '#2c1810',   # frame border
    'danger':       '#df0ceb',   # for cliffs / impassable
    'highlight':    '#c8781e',   # accent / callout
    'visibility':   '#e03d60',   # visibility lines
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
# Placeholder
# =============================================================================

def plot_empty_map(ax, heightmap_data: np.ndarray, features: List[TerrainFeature],
                      cell_size: float = 2.0):
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

# =============================================================================
# LAYER 3: TOPOLOGY VISUALIZATION
# =============================================================================

def _plot_topology_features(ax, features: List[TerrainFeature], 
                            cell_size: float, heightmap_data: np.ndarray):
    """
    Plot terrain features (peaks, ridges, valleys, saddles, flat zones) 
    as NATO-style overlays on the main topology map.
    """
    
    # =========================================================================
    # Peaks (triangles with prominence-based sizing)
    # =========================================================================
    peak_x = []
    peak_y = []
    peak_sizes = []
    peak_prominence = []
    
    for f in features:
        if isinstance(f, PeakFeature):
            x, y = f.centroid
            peak_x.append(x * cell_size)
            peak_y.append(y * cell_size)
            
            # Size by prominence (bigger = more prominent)
            prominence = getattr(f, 'prominence', 10.0)
            peak_prominence.append(prominence)
            
            # Scale: 12-32 points based on prominence (clamped)
            size = np.clip(12 + prominence / 20.0, 12, 32)
            peak_sizes.append(size)
    
    if peak_x:
        # Plot all peaks as black triangles
        ax.scatter(peak_x, peak_y, 
                  marker='^', 
                  s=peak_sizes,
                  c=NATO['peak_marker'],
                  edgecolor=NATO['paper'],
                  linewidth=0.8,
                  zorder=10,
                  alpha=0.95,
                  label='_nolegend_')  # Handled by legend separately
        
        # Add prominence labels for major peaks (prominence > 30m)
        for x, y, prom in zip(peak_x, peak_y, peak_prominence):
            if prom > 30.0:
                ax.annotate(f'{prom:.0f}m', 
                          (x, y),
                          textcoords="offset points",
                          xytext=(8, 8),
                          fontsize=5,
                          fontfamily='monospace',
                          color=NATO['text_muted'],
                          bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor=NATO['paper'],
                                  edgecolor=NATO['border'],
                                  alpha=0.7,
                                  linewidth=0.5),
                          zorder=11)
    
    # =========================================================================
    # Ridges (spine lines)
    # =========================================================================
    for f in features:
        if isinstance(f, RidgeFeature) and hasattr(f, 'spine_points') and f.spine_points:
            spine_x = [p[0] * cell_size for p in f.spine_points]
            spine_y = [p[1] * cell_size for p in f.spine_points]
            
            # Vary line width by ridge length (longer = thicker)
            line_width = np.clip(1.0 + len(f.spine_points) / 200.0, 0.8, 2.5)
            
            ax.plot(spine_x, spine_y,
                   color=NATO['ridge_line'],
                   linewidth=line_width,
                   linestyle='-',
                   alpha=0.85,
                   solid_capstyle='round',
                   zorder=8,
                   label='_nolegend_')
    
    # =========================================================================
    # Valleys (spine lines, different style)
    # =========================================================================
    for f in features:
        if isinstance(f, ValleyFeature) and hasattr(f, 'spine_points') and f.spine_points:
            spine_x = [p[0] * cell_size for p in f.spine_points]
            spine_y = [p[1] * cell_size for p in f.spine_points]
            
            ax.plot(spine_x, spine_y,
                   color=NATO['valley_line'],
                   linewidth=1.0,
                   linestyle='-',
                   alpha=0.8,
                   solid_capstyle='round',
                   zorder=7,
                   label='_nolegend_')
    
    # =========================================================================
    # Saddles (small squares with elevation)
    # =========================================================================
    saddle_x = []
    saddle_y = []
    saddle_elev = []
    
    for f in features:
        if isinstance(f, SaddleFeature):
            x, y = f.centroid
            saddle_x.append(x * cell_size)
            saddle_y.append(y * cell_size)
            saddle_elev.append(getattr(f, 'saddle_elevation_m', None))
    
    if saddle_x:
        ax.scatter(saddle_x, saddle_y,
                  marker='s',
                  s=8,
                  c=NATO['saddle_mark'],
                  edgecolor=NATO['paper'],
                  linewidth=0.5,
                  zorder=9,
                  alpha=0.9,
                  label='_nolegend_')
        
        # Add elevation labels for saddles (if available)
        for x, y, elev in zip(saddle_x, saddle_y, saddle_elev):
            if elev is not None:
                ax.annotate(f'{elev:.0f}m',
                          (x, y),
                          textcoords="offset points",
                          xytext=(6, -6),
                          fontsize=4.5,
                          fontfamily='monospace',
                          color=NATO['text_muted'],
                          bbox=dict(boxstyle='round,pad=0.15',
                                  facecolor=NATO['paper'],
                                  alpha=0.6,
                                  linewidth=0),
                          zorder=11)
    
    # =========================================================================
    # Flat zones (polygon outlines with hatched fill for large areas)
    # =========================================================================
    for f in features:
        if isinstance(f, FlatZoneFeature):
            # Flat zones are represented by their centroid + area indicator
            # Use a circular marker scaled by area
            x, y = f.centroid
            area_px = getattr(f, 'area_pixels', 0)
            
            if area_px > 0:
                # Scale marker size by sqrt(area) in world units
                area_m2 = area_px * (cell_size ** 2)
                radius_m = np.sqrt(area_m2 / np.pi)
                radius_px_marker = radius_m / cell_size
                marker_size = np.clip(radius_px_marker * 2, 15, 80)
                
                # Draw as translucent circle with border
                circle = plt.Circle((x * cell_size, y * cell_size),
                                   radius_m,
                                   facecolor=NATO['flat_fill'],
                                   edgecolor=NATO['flat_edge'],
                                   alpha=0.4,
                                   linewidth=0.8,
                                   zorder=5)
                ax.add_patch(circle)
                
                # if area_m2 > self.config.assembly_major_area_threshold_m2:
                    # hatch_circle = plt.Circle((x * cell_size, y * cell_size),
                                              # radius_m,
                                              # facecolor='none',
                                              # edgecolor=NATO['flat_edge'],
                                              # hatch='///',
                                              # alpha=0.3,
                                              # linewidth=0,
                                              # zorder=6)
                    # ax.add_patch(hatch_circle)
    
    # =========================================================================
    # Prominence/dominance annotations for major peaks
    # =========================================================================
    # Find top 3 peaks by prominence
    peaks = [f for f in features if isinstance(f, PeakFeature) and hasattr(f, 'prominence')]
    if peaks:
        top_peaks = sorted(peaks, key=lambda p: p.prominence, reverse=True)[:3]
        
        for i, peak in enumerate(top_peaks):
            x, y = peak.centroid
            prominence = peak.prominence
            
            # Add small dominance marker
            ax.annotate(f'★ {prominence:.0f}m',
                       (x * cell_size, y * cell_size),
                       textcoords="offset points",
                       xytext=(0, -12),
                       fontsize=5,
                       fontweight='bold',
                       fontfamily='monospace',
                       color=NATO['highlight'],
                       ha='center',
                       bbox=dict(boxstyle='round,pad=0.2',
                               facecolor=NATO['paper'],
                               edgecolor=NATO['highlight'],
                               alpha=0.85,
                               linewidth=0.8),
                       zorder=12)

# =============================================================================
# LAYER 4: RELATIONAL VISUALIZATION
# =============================================================================

def plot_layer4_watersheds(ax, watershed_labels: np.ndarray, 
                           outlets: List[PixelCoord],
                           cell_size: float,
                           alpha: float = 0.05):
    """
    Render watershed basins as colored regions.
    
    OPTIMIZATION: Use unique labels + single polygon per basin
    instead of per-pixel rendering.
    """
    from scipy.ndimage import find_objects
    
    unique_labels = np.unique(watershed_labels[watershed_labels >= 0])
    
    # Limit to top-N largest basins (avoid clutter)
    if len(unique_labels) > 12:
        basin_sizes = [(lbl, np.sum(watershed_labels == lbl)) 
                       for lbl in unique_labels]
        basin_sizes.sort(key=lambda x: x[1], reverse=True)
        unique_labels = [lbl for lbl, _ in basin_sizes[:12]]
    
    # Generate distinct colors for basins
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, lbl in enumerate(unique_labels):
        mask = (watershed_labels == lbl)
        
        # Skip tiny basins
        if np.sum(mask) < 100:
            continue
        
        # Find bounding box for efficiency
        slices = find_objects(mask.astype(int))
        if slices:
            y_slice, x_slice = slices[0]
            y_min, y_max = y_slice.start, y_slice.stop
            x_min, x_max = x_slice.start, x_slice.stop
            
            # Render as image patch (faster than contour)
            patch = np.zeros_like(mask, dtype=float)
            patch[y_min:y_max, x_min:x_max] = mask[y_min:y_max, x_min:x_max]
            
            ax.imshow(patch, cmap=mcolors.ListedColormap([colors[i]]),
                     alpha=alpha, origin='upper',
                     extent=[0, watershed_labels.shape[1] * cell_size,
                            watershed_labels.shape[0] * cell_size, 0],
                     vmin=0, vmax=1)
    
    # Draw outlet markers
    outlet_x = [o[0] * cell_size for o in outlets[:20]]  # Limit to 20
    outlet_y = [o[1] * cell_size for o in outlets[:20]]
    if outlet_x:
        ax.scatter(outlet_x, outlet_y, c=NATO['water_deep'], s=8, 
                  marker='v', edgecolor=NATO['border'], linewidth=0.5,
                  label='Outlets', zorder=5)


def plot_layer4_streams(ax, stream_mask: np.ndarray, 
                        cell_size: float,
                        color: str = None,
                        linewidth: float = 0.8):
    """
    Render stream network as vector lines.
    
    OPTIMIZATION: Convert pixel mask to line segments
    instead of drawing individual pixels.
    """
    from skimage.morphology import skeletonize
    
    # Ensure 1-pixel wide
    if stream_mask.dtype == bool:
        stream_mask = skeletonize(stream_mask)
    
    # Find stream pixels
    ys, xs = np.where(stream_mask > 0)
    
    if len(xs) == 0:
        return
    
    # Convert to world coordinates
    stream_x = xs * cell_size
    stream_y = ys * cell_size
    
    # Draw as scatter (faster than line collection for dense networks)
    ax.scatter(stream_x, stream_y, c=color or NATO['water'],
              s=0.5, alpha=0.7, marker='.', linewidths=0,
              label='Streams', zorder=4)


def plot_layer4_flow_network(ax, features: List[TerrainFeature],
                             flow_graph: Dict[str, List[str]],
                             heightmap, cell_size: float,
                             max_edges: int = 50):
    """
    Render hydrological flow connections between features.
    
    OPTIMIZATION: Limit edges, use curved arrows for clarity.
    """
    from matplotlib.patches import FancyArrowPatch
    
    # Build feature lookup
    feat_lookup = {f.feature_id: f for f in features}
    
    edges_drawn = 0
    for src_id, dst_ids in flow_graph.items():
        if src_id not in feat_lookup:
            continue
        src = feat_lookup[src_id]
        
        for dst_id in dst_ids:
            if dst_id not in feat_lookup:
                continue
            if edges_drawn >= max_edges:
                break
            
            dst = feat_lookup[dst_id]
            
            # World coordinates
            x1, y1 = src.centroid[0] * cell_size, src.centroid[1] * cell_size
            x2, y2 = dst.centroid[0] * cell_size, dst.centroid[1] * cell_size
            
            # Elevation difference for arrow styling
            z1 = heightmap.data[src.centroid[1], src.centroid[0]]
            z2 = heightmap.data[dst.centroid[1], dst.centroid[0]]
            elevation_drop = z1 - z2
            
            # Color by elevation drop (blue = more drop)
            color_intensity = min(1.0, elevation_drop / 50.0)
            edge_color = plt.cm.Blues(0.3 + 0.7 * color_intensity)
            
            # Curved arrow (avoids overlapping with straight lines)
            arrow = FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle='->,head_width=8,head_length=10',
                mutation_scale=1.5,
                linewidth=1.2,
                color=edge_color,
                alpha=0.8,
                connectionstyle='arc3,rad=0.15',
                zorder=6
            )
            ax.add_patch(arrow)
            edges_drawn += 1
        
        if edges_drawn >= max_edges:
            break
    
    if edges_drawn > 0:
        ax.plot([], [], color=NATO['water'], linewidth=1.5, 
               label=f'Flow ({edges_drawn} edges)')


def plot_layer4_connectivity(ax, features: List[TerrainFeature],
                             connectivity_graph: Dict[str, Set[str]],
                             cell_size: float,
                             max_edges: int = 100,
                             show_labels: bool = False):
    """
    Render traversability connectivity graph.
    
    OPTIMIZATION: Density-based rendering for large graphs.
    """
    from matplotlib.patches import FancyArrowPatch
    
    feat_lookup = {f.feature_id: f for f in features}
    edges_drawn = 0
    processed_pairs = set()
    
    # Sort features by importance (peaks first, then by prominence)
    def feature_priority(f):
        if isinstance(f, PeakFeature):
            return (0, -getattr(f, 'prominence', 0))
        return (1, 0)
    
    sorted_features = sorted(features, key=feature_priority)
    
    for src in sorted_features:
        if src.feature_id not in connectivity_graph:
            continue
        
        for dst_id in connectivity_graph[src.feature_id]:
            if dst_id not in feat_lookup:
                continue
            
            # Avoid duplicate edges (undirected graph)
            pair_key = tuple(sorted([src.feature_id, dst_id]))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)
            
            if edges_drawn >= max_edges:
                break
            
            dst = feat_lookup[dst_id]
            
            x1, y1 = src.centroid[0] * cell_size, src.centroid[1] * cell_size
            x2, y2 = dst.centroid[0] * cell_size, dst.centroid[1] * cell_size
            
            # Line style by feature type
            if isinstance(src, PeakFeature) and isinstance(dst, PeakFeature):
                line_style = '-'
                line_width = 1.5
                line_color = NATO['highlight']
            elif isinstance(src, (RidgeFeature, ValleyFeature)):
                line_style = '--'
                line_width = 1.0
                line_color = NATO['ridge_line']
            else:
                line_style = ':'
                line_width = 0.8
                line_color = NATO['text_muted']
            
            ax.plot([x1, x2], [y1, y2], 
                   linestyle=line_style, linewidth=line_width,
                   color=line_color, alpha=0.5, zorder=3)
            edges_drawn += 1
        
        if edges_drawn >= max_edges:
            break
    
    if edges_drawn > 0:
        ax.plot([], [], color=NATO['highlight'], linewidth=1.5,
               label=f'Connectivity ({edges_drawn} edges)')


def plot_layer4_visibility(ax, features: List[TerrainFeature],
                           visibility_graph: Dict[str, Set[str]],
                           heightmap, cell_size: float,
                           max_edges: int = 30):
    """
    Render line-of-sight visibility connections.
    
    OPTIMIZATION: Only show high-value visibility (peaks to peaks).
    """
    feat_lookup = {f.feature_id: f for f in features}
    edges_drawn = 0
    
    # Filter to peak-to-peak visibility only (most tactically relevant)
    peaks = [f for f in features if isinstance(f, PeakFeature)]
    peak_ids = {p.feature_id for p in peaks}
    
    for src in peaks:
        if src.feature_id not in visibility_graph:
            continue
        
        visible_ids = visibility_graph[src.feature_id]
        
        for dst_id in visible_ids:
            if dst_id not in peak_ids:
                continue
            if dst_id not in feat_lookup:
                continue
            
            dst = feat_lookup[dst_id]
            
            x1, y1 = src.centroid[0] * cell_size, src.centroid[1] * cell_size
            x2, y2 = dst.centroid[0] * cell_size, dst.centroid[1] * cell_size
            
            # Check if line crosses high terrain (optional LOS validation)
            ax.plot([x1, x2], [y1, y2],
                   linestyle='-', linewidth=0.8,
                   color=NATO['visibility'], alpha=0.4,
                   zorder=2)
            edges_drawn += 1
        
        if edges_drawn >= max_edges:
            break
    
    if edges_drawn > 0:
        ax.plot([], [], color=NATO['visibility'], linewidth=1.0,
               label=f'Visibility ({edges_drawn} pairs)')


def plot_layer4_cost_surface(ax, cost_surface: np.ndarray,
                             cell_size: float,
                             alpha: float = 0.05):
    """
    Render traversability cost as heatmap overlay.
    Downsample for rendering, use perceptual colormap.
    """
    # Downsample if too large
    if cost_surface.shape[0] > 512:
        from scipy.ndimage import zoom
        scale = 512 / cost_surface.shape[0]
        cost_surface = zoom(cost_surface, scale, order=1)
        cell_size = cell_size / scale
    
    # Normalize to 0-1 (clip extremes)
    cost_clipped = np.clip(cost_surface, 
                          np.percentile(cost_surface, 5),
                          np.percentile(cost_surface, 95))
    cost_norm = (cost_clipped - cost_clipped.min()) / (np.ptp(cost_clipped) + 1e-8)
    
    # Use yellow-red for cost (intuitive: yellow=easy, red=hard)
    cmap = plt.cm.YlOrRd
    
    ax.imshow(cost_norm, cmap=cmap, alpha=alpha, origin='upper',
             extent=[0, cost_surface.shape[1] * cell_size,
                    cost_surface.shape[0] * cell_size, 0],
             vmin=0, vmax=1, zorder=1)

# =============================================================================
# Stamp block (bottom-right corner of topology map)
# =============================================================================

def _draw_stamp(fig, ax_topo, map_name: str, cell_size: float,
                shape: tuple, feature_count: int):
    """Draw a NATO-style map stamp in the corner of the topology axes."""
    H, W = shape
    stamp_text = (
        f"GENERAL INFORMATION\n"
        f"SCALE 1:{int(cell_size * 1000)}  CELL {cell_size}m/px\n"
        f"GRID {W*cell_size:.0f}m × {H*cell_size:.0f}m\n"
        f"FEATURES · {feature_count}"
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

def _draw_legend(ax, relational: dict, features: List[TerrainFeature], 
                 heightmap_shape: tuple = None, cell_size: float = None):
    """
    Draw a NATO-style composite legend for the main topology map.
    Automatically detects which relational layers are present and positions
    legend in least-crowded corner based on feature distribution.
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = []
    
    # =========================================================================
    # Terrain features (always present)
    # =========================================================================
    feature_counts = {
        'Peaks': sum(1 for f in features if isinstance(f, PeakFeature)),
        'Ridges': sum(1 for f in features if isinstance(f, RidgeFeature)),
        'Valleys': sum(1 for f in features if isinstance(f, ValleyFeature)),
        'Saddles': sum(1 for f in features if isinstance(f, SaddleFeature)),
        'Flat zones': sum(1 for f in features if isinstance(f, FlatZoneFeature)),
    }
    
    # Only show feature types that exist
    for name, count in feature_counts.items():
        if count > 0:
            if name == 'Peaks':
                legend_elements.append(Line2D([0], [0], marker='^', color='w', 
                    markerfacecolor=NATO['peak_marker'], markersize=6, 
                    linestyle='None', label=f'{name} ({count})'))
            elif name == 'Ridges':
                legend_elements.append(Line2D([0], [0], color=NATO['ridge_line'], 
                    linewidth=1.5, linestyle='-', label=f'{name} ({count})'))
            elif name == 'Valleys':
                legend_elements.append(Line2D([0], [0], color=NATO['valley_line'], 
                    linewidth=1.2, linestyle='-', label=f'{name} ({count})'))
            elif name == 'Saddles':
                legend_elements.append(Line2D([0], [0], marker='s', color='w', 
                    markerfacecolor=NATO['saddle_mark'], markersize=5, 
                    linestyle='None', label=f'{name} ({count})'))
            elif name == 'Flat zones':
                legend_elements.append(Patch(facecolor=NATO['flat_fill'], 
                    edgecolor=NATO['flat_edge'], linewidth=0.8, 
                    label=f'{name} ({count})'))
    
    # =========================================================================
    # Relational layers (optional)
    # =========================================================================
    if relational:
        if 'stream_network_pixels' in relational:
            legend_elements.append(Line2D([0], [0], color=NATO['water'], 
                linewidth=1.0, linestyle='-', alpha=0.7, label='Streams'))
        
        if 'outlet_pixels' in relational and relational['outlet_pixels']:
            legend_elements.append(Line2D([0], [0], marker='v', color='w', 
                markerfacecolor=NATO['water_deep'], markersize=6, 
                linestyle='None', label='Basin outlets'))
        
        if 'flow_network' in relational and relational['flow_network']:
            legend_elements.append(Line2D([0], [0], color=NATO['water'], 
                linewidth=1.2, linestyle='-', alpha=0.8, label='Flow direction'))
        
        if 'connectivity_graph' in relational and relational['connectivity_graph']:
            legend_elements.append(Line2D([0], [0], color=NATO['highlight'], 
                linewidth=1.2, linestyle='-', label='Traversable routes'))
        
        if 'visibility_graph' in relational and relational['visibility_graph']:
            legend_elements.append(Line2D([0], [0], color=NATO['visibility'], 
                linewidth=0.8, linestyle='-', alpha=0.5, label='Line of sight'))
        
        if 'cost_surface' in relational:
            legend_elements.append(Patch(facecolor='#ff7f0e', alpha=0.3, 
                edgecolor='none', label='High cost areas'))
    
    # =========================================================================
    # Smart positioning: find least-crowded corner
    # =========================================================================
    if heightmap_shape and cell_size and features:
        H, W = heightmap_shape
        # Count features in each quadrant
        quadrants = {'lower_left': 0, 'lower_right': 0, 'upper_left': 0, 'upper_right': 0}
        
        for f in features:
            x, y = f.centroid
            # Convert to normalized coordinates (0-1)
            nx = x * cell_size / (W * cell_size)
            ny = y * cell_size / (H * cell_size)
            
            if nx < 0.5 and ny < 0.5:
                quadrants['lower_left'] += 1
            elif nx >= 0.5 and ny < 0.5:
                quadrants['lower_right'] += 1
            elif nx < 0.5 and ny >= 0.5:
                quadrants['upper_left'] += 1
            else:
                quadrants['upper_right'] += 1
        
        # Find quadrant with fewest features
        best_quadrant = min(quadrants, key=quadrants.get)
        
        # Map quadrant to bbox_to_anchor
        position_map = {
            'lower_left': ('lower left', (0.02, 0.02)),
            'lower_right': ('lower right', (0.98, 0.02)),
            'upper_left': ('upper left', (0.02, 0.98)),
            'upper_right': ('upper right', (0.98, 0.98))
        }
        loc, anchor = position_map[best_quadrant]
    else:
        # Default: lower right
        loc = 'lower right'
        anchor = (0.98, 0.02)
    
    # =========================================================================
    # Draw legend with NATO styling
    # =========================================================================
    if legend_elements:
        legend = ax.legend(
            handles=legend_elements,
            loc=loc,
            bbox_to_anchor=anchor,
            fontsize=5.5,
            frameon=True,
            fancybox=False,
            edgecolor=NATO['border'],
            facecolor=NATO['paper'],
            framealpha=0.92,
            title='LEGEND',
            title_fontsize=6,
            handlelength=2.0,
            handletextpad=0.8,
            borderpad=0.6,
            labelspacing=0.4
        )
        legend.get_title().set_color(NATO['text'])
        legend.get_title().set_fontfamily('monospace')
        
        for text in legend.get_texts():
            text.set_color(NATO['text'])
            text.set_fontfamily('monospace')
        
        legend.set_zorder(20)
        
        return legend
    
    return None

# =============================================================================
# Master render function
# =============================================================================

def render(bundle: dict, features: list, relational: list, map_name: str = "UNDEFINED",
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

    # --- main portrait plot ---
    plot_empty_map(ax_topo, elev, features, cell_size=cell_size)
    
    ax_topo.set_title(
        f'HMA ANALYSIS · {map_name.upper()}',
        fontsize=9, fontweight='bold', fontfamily='monospace',
        color=NATO['text'], pad=6
    )
    
    _plot_topology_features(ax_topo, features, cell_size, elev)
    
    if relational:
        # order matters: cost → watersheds → streams → graphs
        if 'cost_surface' in relational:
            plot_layer4_cost_surface(ax_topo, relational['cost_surface'], cell_size, alpha=0.2)
        if 'watershed_labels' in relational and 'outlet_pixels' in relational:
            plot_layer4_watersheds(ax_topo, relational['watershed_labels'], relational['outlet_pixels'], cell_size)
        if 'stream_network_pixels' in relational:
            plot_layer4_streams(ax_topo, relational['stream_network_pixels'], cell_size)
        if 'flow_network' in relational:
            plot_layer4_flow_network(ax_topo, features, relational['flow_network'], heightmap, cell_size)
        if 'connectivity_graph' in relational:
            plot_layer4_connectivity(ax_topo, features, relational['connectivity_graph'], cell_size)
        if 'visibility_graph' in relational:
            plot_layer4_visibility(ax_topo, features, relational['visibility_graph'], heightmap, cell_size)
    

    # --- Legend ---
    _draw_legend(ax_topo, relational, features, (H, W), cell_size)

    legend = _draw_legend(ax_topo, relational, features, (H, W), cell_size)
    if legend and legend._loc == 'lower right':
        stamp_x = 0.01  # Bottom-left
        stamp_va = 'bottom'
    else:
        stamp_x = 0.01
        stamp_va = 'bottom'
        
    # --- Stamp ---
    _draw_stamp(fig, ax_topo, map_name, cell_size, (H, W), len(features))
    

    # --- Figure title ---
    fig.suptitle(
        'HEIGHTMAP ANALYSIS  ·  STRUCTURAL DECOMPOSITION',
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