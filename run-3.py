"""
Pipeline Proof-of-Concept Visualization — Layers 0 through 3
Shows the complete data flow: Raw → Calibrated → Slope → Curvature → Features
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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

CURVATURE_CMAP = ListedColormap([
    '#444444',  # FLAT (dark gray)
    '#e41a1c',  # CONVEX (red)
    '#377eb8',  # CONCAVE (blue)
    '#984ea3'   # SADDLE (purple)
])
CURVATURE_LABELS = ['FLAT', 'CONVEX', 'CONCAVE', 'SADDLE']

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
# Visualization Functions
# ─────────────────────────────────────────────────────────────────────────────

def annotate_features(ax: plt.Axes, features: List, feature_type: str):
    """Overlay detected features on an axis."""
    colors = {
        'PeakFeature': 'red',
        'RidgeFeature': 'orange',
        'ValleyFeature': 'blue',
        'SaddleFeature': 'red',
        'FlatZoneFeature': 'green',
    }
    markers = {
        'PeakFeature': '^',
        'RidgeFeature': '-',
        'ValleyFeature': 'v',
        'SaddleFeature': '.',
        'FlatZoneFeature': 's',
    }
    
    color = colors.get(feature_type, 'white')
    marker = markers.get(feature_type, 'o')
    
    for feat in features:
        if feat.__class__.__name__ == feature_type:
            cx, cy = feat.centroid
            if marker == '-':
                # Ridge/Valley: draw spine
                if hasattr(feat, 'spine_points') and feat.spine_points:
                    xs = [p[0] for p in feat.spine_points]
                    ys = [p[1] for p in feat.spine_points]
                    ax.plot(xs, ys, color=color, linewidth=2, alpha=0.8, zorder=10)
            else:
                # Point feature
                if marker in ['^', 'v', 's', 'o']:  # Filled markers
                    ax.scatter([cx], [cy], c=color, marker=marker, s=120, 
                              edgecolors='white', linewidths=1.5, zorder=10)
                else:  # Unfilled markers like 'x'
                    ax.scatter([cx], [cy], c=color, marker=marker, s=120, zorder=10)


def plot_layer0(ax: plt.Axes, data: Dict, test_name: str):
    """Layer 0: Calibrated Heightmap."""
    hm = data['heightmap']
    im = ax.imshow(hm.data, cmap='terrain', interpolation='bilinear')
    #ax.set_title('Layer 0: Calibrated Heightmap\n(elevation in meters)', fontsize=10, fontweight='bold')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.colorbar(im, ax=ax, shrink=0.8, label='Elevation (m)')


def plot_layer1_slope(ax: plt.Axes, data: Dict, test_name: str):
    """Layer 1: Slope Map."""
    slope = data['slope']
    im = ax.imshow(slope, cmap=SLOPE_CMAP, interpolation='bilinear')
    #ax.set_title('Layer 1: Slope Magnitude\n(degrees from horizontal)', fontsize=10, fontweight='bold')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.colorbar(im, ax=ax, shrink=0.8, label='Slope (°)')


def plot_layer1_aspect(ax: plt.Axes, data: Dict, test_name: str):
    """Layer 1: Aspect Map."""
    aspect = data['aspect']
    im = ax.imshow(aspect, cmap=ASPECT_CMAP, interpolation='bilinear')
    #ax.set_title('Layer 1: Aspect Direction\n(radians, 0=North, clockwise)', fontsize=10, fontweight='bold')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label='Aspect (rad)')
    cbar.set_ticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    cbar.set_ticklabels(['0\n(N)', 'π/2\n(E)', 'π\n(S)', '3π/2\n(W)', '2π'])


def plot_layer2_curvature(ax: plt.Axes, data: Dict, test_name: str):
    """Layer 2: Curvature Classification."""
    ctype = data['curvature_type']
    
    # Convert string labels to integers for plotting
    type_to_int = {'FLAT': 0, 'CONVEX': 1, 'CONCAVE': 2, 'SADDLE': 3}
    numeric = np.vectorize(type_to_int.get)(ctype)
    
    im = ax.imshow(numeric, cmap=CURVATURE_CMAP, interpolation='nearest')
    #ax.set_title('Layer 2: Curvature Classification\n(H + K thresholds)', fontsize=10, fontweight='bold')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=CURVATURE_CMAP(i), label=CURVATURE_LABELS[i]) 
                      for i in range(4)]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.9)


def plot_layer2_mean_curvature(ax: plt.Axes, data: Dict, test_name: str):
    """Layer 2: Mean Curvature (H) Heatmap."""
    H = data['mean_curvature']
    vmax = np.percentile(np.abs(H), 95)
    im = ax.imshow(H, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='bilinear')
    #ax.set_title('Layer 2: Mean Curvature (H)\n(1/meters)', fontsize=10, fontweight='bold')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.colorbar(im, ax=ax, shrink=0.8, label='H (1/m)')


def plot_layer3_features(ax: plt.Axes, data: Dict, test_name: str):
    """Layer 3: Detected Features Overlaid on Heightmap."""
    hm = data['heightmap']
    features = data['features']
    
    # Base heightmap - lower z-order
    im = ax.imshow(hm.data, cmap='terrain', interpolation='bilinear', alpha=0.7, zorder=1)
    
    # Overlay features by type - medium z-order
    for feat_type in ['PeakFeature', 'RidgeFeature', 'ValleyFeature', 
                      'SaddleFeature', 'FlatZoneFeature']:
        annotate_features(ax, features, feat_type)  # Make sure annotate_features sets zorder
    
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Legend - highest z-order
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, label='Peaks'),
        Line2D([0], [0], marker='.', color='w', markerfacecolor='red', markersize=10, label='Saddles'),
        Line2D([0], [0], color='brown', linewidth=2, label='Ridges'),
        Line2D([0], [0], color='blue', linewidth=2, label='Valleys'),
        Patch(facecolor='green', edgecolor='white', label='Flat Zones'),
    ]
    legend = ax.legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.9)
    legend.set_zorder(100)  # Set legend to highest z-order


# ─────────────────────────────────────────────────────────────────────────────
# Main Visualization
# ─────────────────────────────────────────────────────────────────────────────

def visualize_pipeline_poc(test_dir: str = ".", output_name: str = "debug-preview.png"):
    """
    Create comprehensive pipeline visualization for all 4 test cases.
    
    Each test gets a row showing:
    - Layer 0: Heightmap
    - Layer 1: Slope
    - Layer 2: Curvature Classification
    - Layer 3: Features
    """
    
    test_files = {
        't_single_peak.png': 'Single Peak',
        't_ridge.png': 'Ridge',
        't_saddle.png': 'Saddle Between Peaks',
        't_flatzone.png': 'Flat Zone',
    }
    
    config = PipelineConfig(game_type=GameType.CUSTOM)
    config.verbose = False  # Suppress debug prints
    
    fig, axes = plt.subplots(len(test_files), 4, figsize=(20, 5 * len(test_files)))
    
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
        
        # Plot each layer
        plot_layer0(axes[row_idx, 0], data, test_title)
        plot_layer1_slope(axes[row_idx, 1], data, test_title)
        plot_layer2_curvature(axes[row_idx, 2], data, test_title)
        plot_layer3_features(axes[row_idx, 3], data, test_title)
        
        # Add row label
        axes[row_idx, 0].text(-0.15, 0.5, test_title, transform=axes[row_idx, 0].transAxes,
                             fontsize=12, fontweight='bold', va='center', ha='right',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Column labels
    layer_labels = ['Layer 0\nCalibration', 'Layer 1\nLocal Geometry', 
                    'Layer 2\nRegional Geometry', 'Layer 3\nTopology']
    for col_idx, label in enumerate(layer_labels):
        axes[0, col_idx].text(0.5, 1.08, label, transform=axes[0, col_idx].transAxes,
                             fontsize=11, fontweight='bold', ha='center', va='bottom',
                             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.suptitle('Calibration → Slope/Aspect → Curvature → Feature Detection',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save
    output_path = Path(test_dir) / output_name
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved pipeline visualization to: {output_path}")


def visualize_single_test_detailed(png_path: str, output_name: str = None):
    """
    Create detailed 2x3 grid for a single test showing ALL layer outputs.
    
    Layout:
    [Layer 0: Heightmap]     [Layer 1: Slope]        [Layer 1: Aspect]
    [Layer 2: Mean H]        [Layer 2: Curvature]    [Layer 3: Features]
    """
    
    config = PipelineConfig(game_type=GameType.CUSTOM)
    config.verbose = True
    
    data = run_pipeline_on_png(png_path, config)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    plot_layer0(axes[0, 0], data, "")
    plot_layer1_slope(axes[0, 1], data, "")
    plot_layer1_aspect(axes[0, 2], data, "")
    plot_layer2_mean_curvature(axes[1, 0], data, "")
    plot_layer2_curvature(axes[1, 1], data, "")
    plot_layer3_features(axes[1, 2], data, "")
    
    plt.tight_layout()
    
    if output_name:
        plt.savefig(output_name, dpi=150, bbox_inches='tight')
        print(f"Saved detailed view to: {output_name}")
    


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