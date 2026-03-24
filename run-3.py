"""
Pipeline Proof-of-Concept Visualization — Layers 0 through 3
Shows the complete data flow: Raw → Calibrated → Slope → Curvature → Features
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple

# Import pipeline layers
from core import PipelineConfig, GameType, Heightmap, NormalizationConfig
from calibration import Layer0_Calibration
from lgeometry import Layer1_LocalGeometry
from rgeometry import Layer2_RegionalGeometry
from topological import Layer3_TopologicalFeatures


# ─────────────────────────────────────────────────────────────────────────────
# Color Maps
# ─────────────────────────────────────────────────────────────────────────────

# Curvature colormap for 6 types
CURVATURE_CMAP = ListedColormap([
    '#808080',      # 0: FLAT - gray
    '#F4A582',      # 1: CYLINDRICAL_CONVEX - light orange (ridges)
    '#D73027',      # 2: CONVEX - red (peaks/domes)
    '#A6611A',      # 3: SADDLE - brown (passes)
    '#2C7BB6',      # 4: CONCAVE - blue (bowls/depressions)
    '#92C5DE',      # 5: CYLINDRICAL_CONCAVE - light blue (valleys)
])

CURVATURE_LABELS = [
    'FLAT',
    'RIDGE (cylindrical)',
    'PEAK (dome)',
    'SADDLE',
    'DEPRESSION (bowl)',
    'VALLEY (cylindrical)'
]

SLOPE_CMAP = 'YlOrRd'
ASPECT_CMAP = 'hsv'


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Execution
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline_on_png(png_path: str, config: PipelineConfig) -> Dict:
    """
    Run Layers 0-3 on a PNG heightmap and capture all intermediate outputs.
    
    Returns dict with all layer outputs for visualization.
    """
    # Load PNG as raw uint8
    raw = np.array(Image.open(png_path).convert('L'))
    
    # Layer 0: Calibration
    layer0 = Layer0_Calibration(config)
    heightmap = layer0.execute(
        raw,
        NormalizationConfig(
            horizontal_scale=config.horizontal_scale,
            vertical_scale=config.vertical_scale,
            sea_level_offset=config.sea_level_offset
        )
    )
    
    # Layer 1: Local Geometry (Slope/Aspect)
    layer1 = Layer1_LocalGeometry(config)
    slope_aspect = layer1.execute(heightmap)
    
    # Layer 2: Regional Geometry (Curvature)
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
    
    return {
        'heightmap': heightmap,
        'slope': slope_aspect['slope'],
        'aspect': slope_aspect['aspect'],
        'mean_curvature': curvature['curvature'],
        'gaussian_curvature': curvature['gaussian_curvature'],
        'curvature_type': curvature['curvature_type'],
        'features': features,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature Filtering Helpers
# ─────────────────────────────────────────────────────────────────────────────

def filter_features_by_type(features: List, feature_type: str) -> List:
    """Return features of a specific class."""
    return [f for f in features if f.__class__.__name__ == feature_type]


def get_peak_prominence(peak) -> float:
    """Extract peak prominence for filtering."""
    return peak.prominence if hasattr(peak, 'prominence') else 0


def get_ridge_length(ridge) -> int:
    """Extract ridge length for filtering."""
    return len(ridge.spine_points) if hasattr(ridge, 'spine_points') else 0


def get_valley_length(valley) -> int:
    """Extract valley length for filtering."""
    return len(valley.spine_points) if hasattr(valley, 'spine_points') else 0


def get_flat_zone_area(flat_zone) -> int:
    """Extract flat zone area for filtering."""
    return flat_zone.area_pixels if hasattr(flat_zone, 'area_pixels') else 0


# ─────────────────────────────────────────────────────────────────────────────
# Visualization Functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_heightmap(ax: plt.Axes, data: Dict):
    """Layer 0: Calibrated Heightmap."""
    hm = data['heightmap']
    im = ax.imshow(hm.data, cmap='terrain', interpolation='bilinear')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.colorbar(im, ax=ax, shrink=0.8, label='Elevation (m)')


def plot_slope(ax: plt.Axes, data: Dict):
    """Layer 1: Slope Map."""
    slope = data['slope']
    im = ax.imshow(slope, cmap=SLOPE_CMAP, interpolation='bilinear')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.colorbar(im, ax=ax, shrink=0.8, label='Slope (°)')


def plot_aspect(ax: plt.Axes, data: Dict):
    """Layer 1: Aspect Map."""
    aspect = data['aspect']
    im = ax.imshow(aspect, cmap=ASPECT_CMAP, interpolation='bilinear')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label='Aspect (rad)')
    cbar.set_ticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    cbar.set_ticklabels(['0\n(N)', 'π/2\n(E)', 'π\n(S)', '3π/2\n(W)', '2π'])


def plot_mean_curvature(ax: plt.Axes, data: Dict):
    """Layer 2: Mean Curvature (H) Heatmap."""
    H = data['mean_curvature']
    vmax = np.percentile(np.abs(H), 95)
    im = ax.imshow(H, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='bilinear')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.colorbar(im, ax=ax, shrink=0.8, label='H (1/m)')


def plot_curvature_classification(ax: plt.Axes, data: Dict):
    """Layer 2: Curvature Classification with all 6 types."""
    ctype = data['curvature_type']
    
    # Map each curvature type to an integer
    type_to_int = {
        'FLAT': 0,
        'CYLINDRICAL_CONVEX': 1,
        'CONVEX': 2,
        'SADDLE': 3,
        'CONCAVE': 4,
        'CYLINDRICAL_CONCAVE': 5,
    }
    
    def safe_convert(t):
        return type_to_int.get(t, 0)
    
    numeric = np.vectorize(safe_convert)(ctype)
    
    im = ax.imshow(numeric, cmap=CURVATURE_CMAP, interpolation='nearest', vmin=0, vmax=5)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Custom legend
    legend_handles = [Patch(color=CURVATURE_CMAP(i), label=CURVATURE_LABELS[i]) for i in range(6)]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=7, framealpha=0.9, ncol=2)

# ─────────────────────────────────────────────────────────────────────────────
# Feature Visualization (Split into Logical Groups)
# ─────────────────────────────────────────────────────────────────────────────

def plot_peaks_and_saddles(ax: plt.Axes, data: Dict, min_prominence_m: float = 2.0):
    """
    Show peaks (triangles) and saddles (dots) on heightmap.
    """
    hm = data['heightmap']
    features = data['features']
    
    # Base heightmap
    ax.imshow(hm.data, cmap='terrain', interpolation='bilinear', alpha=0.7)
    
    # Filter peaks by prominence
    peaks = filter_features_by_type(features, 'PeakFeature')
    significant_peaks = [p for p in peaks if get_peak_prominence(p) >= min_prominence_m]
    minor_peaks = [p for p in peaks if get_peak_prominence(p) < min_prominence_m]
    
    # Plot significant peaks (large, filled)
    for p in significant_peaks:
        cx, cy = p.centroid
        ax.scatter(cx, cy, c='red', marker='^', s=150, 
                   edgecolors='white', linewidths=1.5, zorder=10, label='Major Peak')
    
    # Plot minor peaks (smaller, semi-transparent)
    for p in minor_peaks:
        cx, cy = p.centroid
        ax.scatter(cx, cy, c='#FF9999', marker='^', s=80, 
                   alpha=0.6, edgecolors='white', linewidths=0.5, zorder=9)
    
    # Plot saddles
    saddles = filter_features_by_type(features, 'SaddleFeature')
    for s in saddles:
        cx, cy = s.centroid
        ax.scatter(cx, cy, c='cyan', marker='o', s=60, 
                   edgecolors='white', linewidths=0.5, zorder=8, label='Saddle')
    
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Legend (deduplicate labels)
    from collections import OrderedDict
    handles = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
               markersize=10, label=f'Major Peaks (≥{min_prominence_m}m)'),
        Patch(facecolor='#FF9999', edgecolor='white', alpha=0.6, label='Minor Peaks'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', 
               markersize=6, label='Saddles'),
    ]
    l = ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.9)
    l.set_zorder(100)


def plot_ridges_and_valleys(ax: plt.Axes, data: Dict, min_length_px: int = 10):
    """
    Show ridges (orange lines) and valleys (blue lines).
    """
    hm = data['heightmap']
    features = data['features']
    
    # Base heightmap
    ax.imshow(hm.data, cmap='terrain', interpolation='bilinear', alpha=0.7)
    
    # Filter ridges by length
    ridges = filter_features_by_type(features, 'RidgeFeature')
    significant_ridges = [r for r in ridges if get_ridge_length(r) >= min_length_px]
    minor_ridges = [r for r in ridges if get_ridge_length(r) < min_length_px]
    
    # Plot significant ridges (thick, opaque)
    for r in significant_ridges:
        if hasattr(r, 'spine_points') and r.spine_points:
            xs = [p[0] for p in r.spine_points]
            ys = [p[1] for p in r.spine_points]
            ax.plot(xs, ys, color='#D95F02', linewidth=2.5, alpha=0.9, zorder=7, label='Major Ridge')
    
    # Plot minor ridges (thin, semi-transparent)
    for r in minor_ridges:
        if hasattr(r, 'spine_points') and r.spine_points:
            xs = [p[0] for p in r.spine_points]
            ys = [p[1] for p in r.spine_points]
            ax.plot(xs, ys, color='#FDBF6F', linewidth=1, alpha=0.5, zorder=6)
    
    # Filter valleys by length
    valleys = filter_features_by_type(features, 'ValleyFeature')
    significant_valleys = [v for v in valleys if get_valley_length(v) >= min_length_px]
    minor_valleys = [v for v in valleys if get_valley_length(v) < min_length_px]
    
    # Plot significant valleys
    for v in significant_valleys:
        if hasattr(v, 'spine_points') and v.spine_points:
            xs = [p[0] for p in v.spine_points]
            ys = [p[1] for p in v.spine_points]
            ax.plot(xs, ys, color='#1F78B4', linewidth=2.5, alpha=0.9, zorder=7, label='Major Valley')
    
    # Plot minor valleys
    for v in minor_valleys:
        if hasattr(v, 'spine_points') and v.spine_points:
            xs = [p[0] for p in v.spine_points]
            ys = [p[1] for p in v.spine_points]
            ax.plot(xs, ys, color='#A6CEE3', linewidth=1, alpha=0.5, zorder=6)
    
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Legend
    handles = [
        Line2D([0], [0], color='#D95F02', linewidth=2.5, label=f'Major Ridges (≥{min_length_px}px)'),
        Line2D([0], [0], color='#FDBF6F', linewidth=1, label='Minor Ridges'),
        Line2D([0], [0], color='#1F78B4', linewidth=2.5, label=f'Major Valleys (≥{min_length_px}px)'),
        Line2D([0], [0], color='#A6CEE3', linewidth=1, label='Minor Valleys'),
    ]
    l = ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.9)
    l.set_zorder(100)

def plot_flat_zones(ax: plt.Axes, data: Dict, min_area_px: int = 100):
    """
    Show flat zones as semi-transparent polygons.
    """
    hm = data['heightmap']
    features = data['features']
    
    # Base heightmap
    ax.imshow(hm.data, cmap='terrain', interpolation='bilinear', alpha=0.7)
    
    # Filter flat zones by area
    flat_zones = filter_features_by_type(features, 'FlatZoneFeature')
    significant_zones = [fz for fz in flat_zones if get_flat_zone_area(fz) >= min_area_px]
    minor_zones = [fz for fz in flat_zones if get_flat_zone_area(fz) < min_area_px]
    
    # Plot significant flat zones (filled, visible)
    for fz in significant_zones:
        if 'bounds' in fz.metadata:
            x_min, x_max, y_min, y_max = fz.metadata['bounds']
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                  facecolor='#33A02C', edgecolor='white', 
                                  alpha=0.4, linewidth=1.5, zorder=5, label='Major Flat Zone')
            ax.add_patch(rect)
    
    # Plot minor flat zones (light fill, no outline)
    for fz in minor_zones:
        if 'bounds' in fz.metadata:
            x_min, x_max, y_min, y_max = fz.metadata['bounds']
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                  facecolor='#B2DF8A', edgecolor='none', 
                                  alpha=0.3, zorder=4)
            ax.add_patch(rect)
    
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Legend
    handles = [
        Patch(facecolor='#33A02C', edgecolor='white', alpha=0.5, label=f'Major Flat Zones (≥{min_area_px}px)'),
        Patch(facecolor='#B2DF8A', edgecolor='none', alpha=0.3, label='Minor Flat Zones'),
    ]
    l = ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.9)
    l.set_zorder(100)

# ─────────────────────────────────────────────────────────────────────────────
# Main Visualization
# ─────────────────────────────────────────────────────────────────────────────

def visualize_pipeline_poc(test_dir: str = ".", output_name: str = "debug-preview.png"):
    """
    Create comprehensive pipeline visualization for all 4 test cases.
    
    Each test gets a 3x3 grid:
    Row 1: Heightmap | Slope | Aspect
    Row 2: Mean Curvature | Curvature Type | Peaks & Saddles
    Row 3: Ridges & Valleys | Flat Zones | Feature Summary
    """
    
    test_files = {
        't_single_peak.png': 'Single Peak',
        't_ridge.png': 'Ridge',
        't_saddle.png': 'Saddle Peaks',
        't_flatzone.png': 'Flat Zone',
    }
    
    config = PipelineConfig(game_type=GameType.CUSTOM)
    config.verbose = False  # Suppress debug prints
    
    # 3x3 grid for each test
    fig, axes = plt.subplots(len(test_files), 9, figsize=(24, 6 * len(test_files)))
    
    # Handle single-row case
    if len(test_files) == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, (png_file, test_title) in enumerate(test_files.items()):
        png_path = Path(test_dir) / png_file
        
        if not png_path.exists():
            print(f"Warning: {png_file} not found, skipping...")
            continue
        
        # Run pipeline
        data = run_pipeline_on_png(str(png_path), config)
        
        # Row 1: Primary Geometry
        plot_heightmap(axes[row_idx, 0], data)
        plot_slope(axes[row_idx, 1], data)
        plot_aspect(axes[row_idx, 2], data)
        
        # Row 2: Secondary Geometry + Features Group 1
        plot_mean_curvature(axes[row_idx, 3], data)
        plot_curvature_classification(axes[row_idx, 4], data)
        plot_peaks_and_saddles(axes[row_idx, 5], data)
        
        # Row 3: Features Group 2 + Group 3 + Summary
        plot_ridges_and_valleys(axes[row_idx, 6], data)
        plot_flat_zones(axes[row_idx, 7], data)
        
        # Feature Summary (text summary of feature counts)
        ax = axes[row_idx, 8]
        ax.axis('off')
        features = data['features']
        peaks = len([f for f in features if f.__class__.__name__ == 'PeakFeature'])
        ridges = len([f for f in features if f.__class__.__name__ == 'RidgeFeature'])
        valleys = len([f for f in features if f.__class__.__name__ == 'ValleyFeature'])
        saddles = len([f for f in features if f.__class__.__name__ == 'SaddleFeature'])
        flat_zones = len([f for f in features if f.__class__.__name__ == 'FlatZoneFeature'])
        
        summary_text = f"Feature Summary\n{'-' * 20}\n\n"
        summary_text += f"Peaks:     {peaks}\n"
        summary_text += f"Ridges:    {ridges}\n"
        summary_text += f"Valleys:   {valleys}\n"
        summary_text += f"Saddles:   {saddles}\n"
        summary_text += f"Flat Zones: {flat_zones}"
        
        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add row label
        axes[row_idx, 0].text(-0.1, 0.5, test_title, transform=axes[row_idx, 0].transAxes,
                             fontsize=12, fontweight='bold', va='center', ha='right',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Column labels
    col_labels = [
        'Heightmap\n(Elevation)', 'Slope\n(degrees)', 'Aspect\n(radians)',
        'Mean Curvature\n(H, 1/m)', 'Curvature Type\n(6 classes)', 'Peaks & Saddles\n(▲ = peak, ● = saddle)',
        'Ridges & Valleys\n(orange = ridge, blue = valley)', 'Flat Zones\n(green polygons)', 'Feature Summary\n(counts)'
    ]
    
    for col_idx, label in enumerate(col_labels):
        axes[0, col_idx].set_title(label, fontsize=10, fontweight='bold', pad=10)
    
    plt.suptitle('Heightmap Analysis Pipeline: Calibration → Slope/Aspect → Curvature → Feature Detection',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save
    output_path = Path(test_dir) / output_name
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved pipeline visualization to: {output_path}")


def visualize_single_test_detailed(png_path: str, output_name: str = None):
    """
    Create detailed 3x3 grid for a single test.
    
    Layout:
    Row 1: Heightmap | Slope | Aspect
    Row 2: Mean Curvature | Curvature Type | Peaks & Saddles
    Row 3: Ridges & Valleys | Flat Zones | Feature Summary
    """
    
    config = PipelineConfig(game_type=GameType.CUSTOM)
    config.verbose = True
    
    data = run_pipeline_on_png(png_path, config)
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # Row 1: Primary Geometry
    plot_heightmap(axes[0, 0], data)
    plot_slope(axes[0, 1], data)
    plot_aspect(axes[0, 2], data)
    
    # Row 2: Secondary Geometry + Features Group 1
    plot_mean_curvature(axes[1, 0], data)
    plot_curvature_classification(axes[1, 1], data)
    plot_peaks_and_saddles(axes[1, 2], data)
    
    # Row 3: Features Group 2 + Group 3 + Summary
    plot_ridges_and_valleys(axes[2, 0], data)
    plot_flat_zones(axes[2, 1], data)
    
    # Feature Summary
    ax = axes[2, 2]
    ax.axis('off')
    features = data['features']
    peaks = len([f for f in features if f.__class__.__name__ == 'PeakFeature'])
    ridges = len([f for f in features if f.__class__.__name__ == 'RidgeFeature'])
    valleys = len([f for f in features if f.__class__.__name__ == 'ValleyFeature'])
    saddles = len([f for f in features if f.__class__.__name__ == 'SaddleFeature'])
    flat_zones = len([f for f in features if f.__class__.__name__ == 'FlatZoneFeature'])
    
    summary_text = f"Feature Summary\n{'-' * 20}\n\n"
    summary_text += f"Peaks:     {peaks}\n"
    summary_text += f"Ridges:    {ridges}\n"
    summary_text += f"Valleys:   {valleys}\n"
    summary_text += f"Saddles:   {saddles}\n"
    summary_text += f"Flat Zones: {flat_zones}"
    
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Column labels - fixed indexing
    col_labels = [
        'Heightmap\n(Elevation)', 'Slope\n(degrees)', 'Aspect\n(radians)',
        'Mean Curvature\n(H, 1/m)', 'Curvature Type\n(6 classes)', 'Peaks & Saddles\n(▲ = peak, ● = saddle)',
        'Ridges & Valleys\n(orange = ridge, blue = valley)', 'Flat Zones\n(green polygons)', 'Feature Summary\n(counts)'
    ]

    # Row 1 labels (columns 0-2)
    #for col_idx in range(3):
    #    axes[0, col_idx].set_title(col_labels[col_idx], fontsize=10, fontweight='bold', pad=10)

    # Row 2 labels (columns 3-5)
    #for col_idx in range(3):
    #    axes[0, col_idx + 3].set_title(col_labels[col_idx + 3], fontsize=10, fontweight='bold', pad=10)

    # Row 3 labels (columns 6-8)
    #for col_idx in range(3):
    #    axes[0, col_idx + 6].set_title(col_labels[col_idx + 6], fontsize=10, fontweight='bold', pad=10)
    
    plt.suptitle(f'Detailed Analysis: {Path(png_path).stem}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_name:
        plt.savefig(output_name, dpi=150, bbox_inches='tight')
        print(f"Saved detailed view to: {output_name}")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Detailed single test
        png_path = sys.argv[1]
        output = sys.argv[2] if len(sys.argv) > 2 else f"{Path(png_path).stem}_detail.png"
        visualize_single_test_detailed(png_path, output)
    else:
        # Full pipeline PoC
        visualize_pipeline_poc()