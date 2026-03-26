import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import argparse

from core import PipelineConfig, Template
from calibration import Layer0_Calibration
from lgeometry import Layer1_LocalGeometry
from rgeometry import Layer2_RegionalGeometry
from topological import Layer3_TopologicalFeatures
from relational import Layer4_Relational
from semantics import Layer5_Semantics
from styles import get_visualizer


def run_pipeline(png_path: str, config: PipelineConfig) -> Dict:
    """Execute full pipeline, return all outputs."""
    from PIL import Image
    
    raw = np.array(Image.open(png_path).convert('L'))
    
    # Layer 0
    layer0 = Layer0_Calibration(config)
    heightmap = layer0.execute(raw)
    
    # Layer 1
    layer1 = Layer1_LocalGeometry(config)
    slope_aspect = layer1.execute(heightmap)
    
    # Layer 2
    layer2 = Layer2_RegionalGeometry(config)
    curvature = layer2.execute(heightmap)
    
    # Layer 3
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
    
    # Layer 4
    layer4 = Layer4_Relational(config)
    relational = layer4.execute(bundle)
    bundle.update(relational)
    
    # Layer 5
    layer5 = Layer5_Semantics(config)
    analyzed = layer5.execute(bundle)
    
    return {
        'heightmap': heightmap.data,
        'features': features,
        'relational': relational,
        'analyzed': analyzed,
        'semantic_index': analyzed.semantic_index,
    }


def visualize(png_path: str, 
              output_name: Optional[str] = None, 
              style: str = 'gis',
              overlays: Optional[List[str]] = None,
              show_grid: bool = True,
              show_legend: bool = True,
              base_alpha: float = 0.7,
              dpi: int = 150) -> None:
    """
    Orchestrate pipeline run and visualization.
    
    Args:
        png_path: Path to input heightmap image
        output_name: Output filename (None = show, not save)
        style: Visualization style ('gis', 'game', 'military', 'orienteering')
        overlays: List of overlays to show ('strategic', 'tactical', 'logistical', 'hydro', 'quality', 'all', 'none')
        show_grid: Show coordinate grid
        show_legend: Show legend
        base_alpha: Base terrain transparency (0-1)
        dpi: Output resolution
    """
    print(f"Loading: {png_path}")
    
    # Setup config
    config = PipelineConfig(baseline=Template.ARMA_3)
    config.verbose = True
    
    # Apply visualization settings
    config.visualization_style = style
    config.visualization_overlays = overlays if overlays else ['strategic']
    config.visualization_show_grid = show_grid
    config.visualization_show_legend = show_legend
    config.visualization_base_alpha = base_alpha
    config.visualization_dpi = dpi
    
    # Run pipeline
    data = run_pipeline(png_path, config)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Get visualizer and plot
    visualizer = get_visualizer(config)
    visualizer.plot(ax, data)
    
    # Add legend if enabled
    if show_legend:
        legend_elements = visualizer.legend_elements()
        if legend_elements:
            ax.legend(handles=legend_elements, loc='lower right', 
                      fontsize=8, framealpha=0.9)
    
    # Title with style and overlays info
    title = f'Terrain Analysis: {Path(png_path).stem}'
    if overlays and overlays != ['all']:
        title += f' | Overlays: {", ".join(overlays)}'
    plt.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save or show
    if output_name:
        plt.savefig(output_name, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_name}")
    else:
        plt.show()
    
    plt.close(fig)


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    """Command line entry point with overlay support."""
    parser = argparse.ArgumentParser(
        description='Terrain Analysis Pipeline - Generate tactical terrain maps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyse.py map.png
  python analyse.py map.png --style game
  python analyse.py map.png --style gis --overlays strategic tactical
  python analyse.py map.png --style military --overlays all --output military_map.png
  python analyse.py map.png --style orienteering --overlays none


  REMARK: ONLY GIS HAS OVERLAYS ATM!

Overlay Types:
  strategic   - Defensive positions, observation posts
  tactical    - Chokepoints, ambush zones, cover ridges
  logistical  - Assembly areas, vehicle routes
  hydro       - Drainage networks, valley types
  quality     - Quality metrics (defensive scores, cover quality)
  all         - All overlays
  none        - Terrain only
        """
    )
    
    parser.add_argument('input', help='Input heightmap image (PNG)')
    parser.add_argument('--style', '-s', default='gis',
                        choices=['gis', 'game', 'military', 'orienteering'],
                        help='Visualization style (default: gis)')
    parser.add_argument('--overlays', '-o', nargs='+',
                        default=['strategic'],
                        choices=['strategic', 'tactical', 'logistical', 'hydro', 'quality', 'all', 'none'],
                        help='Overlay layers to show (default: strategic)')
    parser.add_argument('--output', '-out', default=None,
                        help='Output filename (default: inputname_style.png)')
    parser.add_argument('--no-grid', action='store_true',
                        help='Hide coordinate grid')
    parser.add_argument('--no-legend', action='store_true',
                        help='Hide legend')
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='Base terrain transparency (0-1, default: 0.7)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Output resolution (default: 150)')
    
    args = parser.parse_args()
    
    # Handle 'all' and 'none' special cases
    if 'all' in args.overlays:
        overlays = ['strategic', 'tactical', 'logistical', 'hydro', 'quality']
    elif 'none' in args.overlays:
        overlays = []
    else:
        overlays = args.overlays
    
    # Generate default output name if not provided
    output_name = args.output
    if output_name is None:
        output_name = f'{Path(args.input).stem}_{args.style}.png'
    
    # Run visualization
    visualize(
        png_path=args.input,
        output_name=output_name,
        style=args.style,
        overlays=overlays,
        show_grid=not args.no_grid,
        show_legend=not args.no_legend,
        base_alpha=args.alpha,
        dpi=args.dpi
    )


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    main()