"""
Visualization styles for terrain analysis.

Each style renders the same pipeline data according to domain conventions:
- Military: NATO APP-6 symbology, contour lines, MGRS grid
- Orienteering: ISOM standard, vegetation colors, minimal overlays
- Game: Intuitive icons, saturated colors, no legend
- GIS: Scientific polygons, transparency, color gradients
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Ellipse, Circle
from matplotlib.colors import LinearSegmentedColormap, LightSource
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any

from core import PipelineConfig


# =============================================================================
# Base Visualizer (Abstract)
# =============================================================================

class BaseVisualizer(ABC):
    """Abstract base for all terrain visualization styles."""
    
    def __init__(self, config):
        self.config = config
        self.overlays = getattr(config, 'visualization_overlays', ['strategic'])
        self.show_grid = getattr(config, 'visualization_show_grid', True)
        self.show_legend = getattr(config, 'visualization_show_legend', True)
        self.base_alpha = getattr(config, 'visualization_base_alpha', 0.7)
        self.dpi = getattr(config, 'visualization_dpi', 150)
    
    @abstractmethod
    def plot(self, ax: plt.Axes, data: Dict) -> None:
        """Render the terrain analysis. Must be implemented by all visualizers."""
        pass
    
    @abstractmethod
    def legend_elements(self) -> List:
        """Return legend handles. Must be implemented by all visualizers."""
        pass
    
    # =========================================================================
    # Overlay System
    # =========================================================================
    
    def _dispatch_overlays(self, ax: plt.Axes, data: Dict) -> None:
        """
        Call overlay methods if they exist.
        This is safe to call from plot() without breaking existing visualizers.
        """
        if hasattr(self, '_plot_strategic'):
            #self._plot_strategic(ax, data)
            pass
        
        if hasattr(self, '_plot_tactical'):
            self._plot_tactical(ax, data)
        
        if hasattr(self, '_plot_logistical'):
            #self._plot_logistical(ax, data)
            pass
        
        if hasattr(self, '_plot_hydro'):
            #self._plot_hydro(ax, data)
            pass
        
        if hasattr(self, '_plot_quality'):
            #self._plot_quality(ax, data)
            pass
    
    # =========================================================================
    # Helper methods (available to all visualizers)
    # =========================================================================
    
    def _hillshade(self, heightmap: np.ndarray) -> np.ndarray:
        """Compute hillshade for base terrain."""
        from matplotlib.colors import LightSource
        ls = LightSource(azdeg=315, altdeg=45)
        return ls.hillshade(heightmap, vert_exag=2)
    
    def _contours(self, ax: plt.Axes, heightmap: np.ndarray, 
                  levels: int = 10, **kwargs) -> None:
        """Add contour lines to axes."""
        default_kwargs = {'colors': 'white', 'linewidths': 0.3, 'alpha': 0.4}
        default_kwargs.update(kwargs)
        levels_vals = np.linspace(np.min(heightmap), np.max(heightmap), levels)
        ax.contour(heightmap, levels=levels_vals, **default_kwargs)


# =============================================================================
# GIS Thematic Style (Scientific)
# =============================================================================

class GISVisualizer(BaseVisualizer):
    """
    GIS-style thematic map.
    
    Features: Polygons with transparency, thickness-based lines,
              radial gradients for defensive positions.
    """
    
    def plot(self, ax: plt.Axes, data: Dict) -> None:
        """Render base terrain, then dispatch overlays based on config."""
        heightmap = data['heightmap']
        
        # Base: hillshade (always shown)
        ax.imshow(self._hillshade(heightmap), cmap='gray', 
                  alpha=self.base_alpha, interpolation='bilinear')
        
        # Optional: subtle elevation tint
        if getattr(self.config, 'gis_show_elevation_tint', True):
            ax.imshow(heightmap, cmap='terrain', alpha=0.3, 
                      interpolation='bilinear')
        
        # Grid (optional)
        if self.show_grid:
            ax.grid(True, color='gray', linestyle='--', linewidth=0.3, alpha=0.5)
        
        # Dispatch overlays based on config
        self._dispatch_overlays(ax, data)
        
        ax.set_title('Terrain Analysis (GIS)', fontsize=10, fontweight='bold')
    
    def _dispatch_overlays(self, ax: plt.Axes, data: Dict) -> None:
        """Call overlay methods based on self.overlays config."""
        semantic_index = data.get('semantic_index', {})
        
        if 'strategic' in self.overlays or 'all' in self.overlays:
            self._draw_strategic(ax, semantic_index)
        
        if 'tactical' in self.overlays or 'all' in self.overlays:
            self._draw_tactical(ax, semantic_index)
        
        if 'logistical' in self.overlays or 'all' in self.overlays:
            self._draw_logistical(ax, semantic_index)
        
        if 'hydro' in self.overlays or 'all' in self.overlays:
            self._draw_hydro(ax, semantic_index)
        
        if 'quality' in self.overlays or 'all' in self.overlays:
            self._draw_quality(ax, semantic_index)
    
    # =========================================================================
    # Overlay Drawing Methods
    # =========================================================================
    
    def _draw_strategic(self, ax: plt.Axes, semantic_index: Dict) -> None:
        """Defensive positions and observation posts."""
        for item in semantic_index.get('defensive_positions', []):
            self._draw_defensive_radial(ax, item)
        
        for item in semantic_index.get('observation_posts', []):
            self._draw_observation_marker(ax, item)
    
    def _draw_tactical(self, ax: plt.Axes, semantic_index: Dict) -> None:
        """Chokepoints, ambush zones, cover ridges."""
        for item in semantic_index.get('chokepoints', []):
            self._draw_chokepoint_ellipse(ax, item)
        
        for item in semantic_index.get('ambush_positions', []):
            self._draw_ambush_corridor(ax, item)
        
        for item in semantic_index.get('cover_positions', []):
            self._draw_ridge_line(ax, item)
    
    def _draw_logistical(self, ax: plt.Axes, semantic_index: Dict) -> None:
        """Assembly areas and vehicle routes."""
        for item in semantic_index.get('assembly_areas', []):
            self._draw_assembly_polygon(ax, item)
    
    def _draw_hydro(self, ax: plt.Axes, semantic_index: Dict) -> None:
        """Drainage networks (valleys with flow magnitude)."""
        for item in semantic_index.get('hydro_features', []):
            drainage_mag = item.get('drainage_magnitude', 0)
            if drainage_mag > 0:
                self._draw_drainage_line(ax, item, drainage_mag)
    
    def _draw_quality(self, ax: plt.Axes, semantic_index: Dict) -> None:
        """Enhance existing features with quality metrics."""
        # Re-draw defensive positions with score-based color intensity
        for item in semantic_index.get('defensive_positions', []):
            score = item.get('defensive_score', 0.5)
            self._enhance_defensive_quality(ax, item, score)
  
    # =========================================================================
    # Individual Drawing Helpers 
    # =========================================================================
    
    def _draw_assembly_polygon(self, ax, item: Dict) -> None:
        """Draw assembly area as semi-transparent green rectangle."""
        x, y = item.get('centroid', (0, 0))
        area_m2 = item.get('area_m2', 0)
        size = np.sqrt(area_m2) / 5
        size = np.clip(size, 15, 60)
        
        rect = Rectangle((x - size/2, y - size/2), size, size,
                         facecolor='green', alpha=0.4, 
                         edgecolor='darkgreen', linewidth=1)
        ax.add_patch(rect)
    
    def _draw_defensive_radial(self, ax, item: Dict) -> None:
        """Draw defensive position as radial gradient."""
        x, y = item.get('centroid', (0, 0))
        score = item.get('defensive_score', 0.5)
        radius = 15 + score * 20
        
        outer = Circle((x, y), radius, facecolor='red', alpha=0.15, 
                       edgecolor='darkred', linewidth=0.8)
        ax.add_patch(outer)
        
        inner = Circle((x, y), radius * 0.4, facecolor='red', alpha=0.5,
                       edgecolor='darkred', linewidth=0.5)
        ax.add_patch(inner)
    
    def _draw_observation_marker(self, ax, item: Dict) -> None:
        """Draw observation post as star marker."""
        x, y = item.get('centroid', (0, 0))
        ax.scatter(x, y, s=25, c='steelblue', marker='*', 
                   edgecolor='white', linewidth=0.5, zorder=11)
    
    def _draw_chokepoint_ellipse(self, ax, item: Dict) -> None:
        """Draw chokepoint as ellipse sized by connectivity degree."""
        x, y = item.get('centroid', (0, 0))
        degree = item.get('connectivity_degree', 2)
        
        width = 20 + degree * 8
        height = 15 + degree * 6
        
        ellipse = Ellipse((x, y), width, height, facecolor='orange', 
                          alpha=0.35, edgecolor='darkorange', linewidth=1)
        ax.add_patch(ellipse)
    
    def _draw_ridge_line(self, ax, item: Dict) -> None:
        """Draw ridge with thickness proportional to cover quality."""
        spine = item.get('spine', [])
        quality = item.get('quality', 0.5)
        
        if len(spine) >= 2:
            xs = [p[0] for p in spine]
            ys = [p[1] for p in spine]
            linewidth = 1 + quality * 3
            ax.plot(xs, ys, color='saddlebrown', linewidth=linewidth, 
                    alpha=0.8, solid_capstyle='round')
    
    def _draw_ambush_corridor(self, ax, item: Dict) -> None:
        """Draw ambush zone as red-tinted corridor."""
        spine = item.get('spine', [])
        rating = item.get('rating', 0)
        
        if len(spine) >= 2:
            xs = [p[0] for p in spine]
            ys = [p[1] for p in spine]
            intensity = 0.3 + rating * 0.5
            color = (intensity, 0.2, 0.2)
            linestyle = '--' if rating > 0.5 else '-'
            
            ax.plot(xs, ys, color=color, linewidth=2, 
                    linestyle=linestyle, alpha=0.7)
    
    def _draw_drainage_line(self, ax, item: Dict, magnitude: int) -> None:
        """Draw drainage line with thickness by magnitude."""
        spine = item.get('spine', [])
        if len(spine) >= 2:
            xs = [p[0] for p in spine]
            ys = [p[1] for p in spine]
            linewidth = 0.5 + min(magnitude / 20, 3)
            ax.plot(xs, ys, color='blue', linewidth=linewidth, alpha=0.6)
    
    def _enhance_defensive_quality(self, ax, item: Dict, score: float) -> None:
        """Add quality indicator to defensive position."""
        x, y = item.get('centroid', (0, 0))
        # Small white glow for high-value positions
        if score > 0.7:
            glow = Circle((x, y), 8, facecolor='white', alpha=0.3, 
                          edgecolor='none', zorder=9)
            ax.add_patch(glow)
    
    def legend_elements(self) -> List:
        """Return legend handles for GIS style."""
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        
        return [
            Patch(facecolor='green', alpha=0.4, edgecolor='darkgreen', 
                  label='Assembly Area'),
            Patch(facecolor='red', alpha=0.3, edgecolor='darkred', 
                  label='Defensive Position'),
            Patch(facecolor='orange', alpha=0.35, edgecolor='darkorange', 
                  label='Chokepoint'),
            Line2D([0], [0], color='saddlebrown', linewidth=2, label='Ridge (Cover)'),
            Line2D([0], [0], color='darkred', linewidth=2, linestyle='--', 
                   label='Ambush Zone'),
            Line2D([0], [0], color='blue', linewidth=1.5, label='Drainage'),
        ]


# =============================================================================
# Game Tactical Style (Military Simulators)
# =============================================================================

# TODO: implement overlays

class GameVisualizer(BaseVisualizer):
    """
    Game-style tactical map.
    
    Features: Saturated icons, shield/eye symbols, minimal text.
    """
    
    def plot(self, ax: plt.Axes, data: Dict) -> None:
        heightmap = data['heightmap']
        semantic_index = data.get('semantic_index', {})
        
        # Base: hillshade (lighter for icon visibility)
        ax.imshow(self._hillshade(heightmap), cmap='gray', 
                  alpha=0.5, interpolation='bilinear')
        
        # Assembly areas (green squares)
        for item in semantic_index.get('assembly_areas', []):
            self._draw_assembly_icon(ax, item)
        
        # Defensive positions (shield icons via marker)
        for item in semantic_index.get('defensive_positions', []):
            self._draw_defensive_icon(ax, item)
        
        # Observation posts (eye icons)
        for item in semantic_index.get('observation_posts', []):
            self._draw_observation_icon(ax, item)
        
        # Chokepoints (diamonds)
        for item in semantic_index.get('chokepoints', []):
            self._draw_chokepoint_icon(ax, item)
        
        # Cover positions (thick lines)
        for item in semantic_index.get('cover_positions', []):
            self._draw_cover_line(ax, item)
        
        # Ambush positions (skull-like marker)
        for item in semantic_index.get('ambush_positions', []):
            self._draw_ambush_icon(ax, item)
        
        ax.set_title('TACTICAL MAP', fontsize=12, fontweight='bold')
        ax.grid(False)
    
    def _draw_assembly_icon(self, ax, item: Dict) -> None:
        x, y = item.get('centroid', (0, 0))
        ax.scatter(x, y, s=50, c='limegreen', marker='s', 
                   edgecolor='white', linewidth=1, zorder=10)
    
    def _draw_defensive_icon(self, ax, item: Dict) -> None:
        x, y = item.get('centroid', (0, 0))
        score = item.get('defensive_score', 0.5)
        size = 40 + score * 30
        ax.scatter(x, y, s=size, c='crimson', marker='^', 
                   edgecolor='white', linewidth=1.5, zorder=10)
    
    def _draw_observation_icon(self, ax, item: Dict) -> None:
        x, y = item.get('centroid', (0, 0))
        ax.scatter(x, y, s=30, c='steelblue', marker='*', 
                   edgecolor='white', linewidth=0.5, zorder=10)
    
    def _draw_chokepoint_icon(self, ax, item: Dict) -> None:
        x, y = item.get('centroid', (0, 0))
        ax.scatter(x, y, s=40, c='orange', marker='D', 
                   edgecolor='white', linewidth=1, zorder=9)
    
    def _draw_cover_line(self, ax, item: Dict) -> None:
        spine = item.get('spine', [])
        if len(spine) >= 2:
            xs = [p[0] for p in spine]
            ys = [p[1] for p in spine]
            ax.plot(xs, ys, color='saddlebrown', linewidth=2.5, 
                    alpha=0.9, solid_capstyle='round')
    
    def _draw_ambush_icon(self, ax, item: Dict) -> None:
        x, y = item.get('centroid', (0, 0))
        rating = item.get('rating', 0)
        color = 'darkred' if rating > 0.5 else 'red'
        ax.scatter(x, y, s=35, c=color, marker='v', 
                   edgecolor='white', linewidth=1, zorder=8)
    
    def legend_elements(self) -> List:
        from matplotlib.lines import Line2D
        return [
            plt.scatter([], [], s=50, c='limegreen', marker='s', label='Assembly'),
            plt.scatter([], [], s=40, c='crimson', marker='^', label='Defensive'),
            plt.scatter([], [], s=30, c='steelblue', marker='*', label='Observation'),
            plt.scatter([], [], s=40, c='orange', marker='D', label='Chokepoint'),
            plt.scatter([], [], s=35, c='red', marker='v', label='Ambush'),
            Line2D([0], [0], color='saddlebrown', linewidth=2.5, label='Cover'),
        ]


# =============================================================================
# Military Style (NATO APP-6)
# =============================================================================

# TODO: implement overlays

class MilitaryVisualizer(BaseVisualizer):
    """
    Military-style tactical map with NATO symbology.
    
    Features: Contour lines, MGRS-style grid, formal icons.
    """
    
    def plot(self, ax: plt.Axes, data: Dict) -> None:
        heightmap = data['heightmap']
        semantic_index = data.get('semantic_index', {})
        
        # Base: hillshade (subtle)
        ax.imshow(self._hillshade(heightmap), cmap='gray', 
                  alpha=0.6, interpolation='bilinear')
        
        # Contour lines (military standard)
        self._contours(ax, heightmap, levels=12, colors='brown', linewidths=0.5)
        
        # Defensive positions (shield frame)
        for item in semantic_index.get('defensive_positions', []):
            self._draw_military_defensive(ax, item)
        
        # Observation posts (eye symbol in square)
        for item in semantic_index.get('observation_posts', []):
            self._draw_military_observation(ax, item)
        
        # Chokepoints (diamond with cross)
        for item in semantic_index.get('chokepoints', []):
            self._draw_military_chokepoint(ax, item)
        
        # Assembly areas (rectangle with flag)
        for item in semantic_index.get('assembly_areas', []):
            self._draw_military_assembly(ax, item)
        
        # Grid (MGRS-style)
        if self.show_grid:
            self._draw_mgrs_grid(ax, heightmap.shape)
        
        ax.set_title('TERRAIN ANALYSIS (NATO)', fontsize=10, fontweight='bold')
    
    def _draw_military_defensive(self, ax, item: Dict) -> None:
        x, y = item.get('centroid', (0, 0))
        # Shield-like frame (square with rounded corners approximation)
        size = 12
        rect = Rectangle((x - size/2, y - size/2), size, size,
                         facecolor='none', edgecolor='crimson', linewidth=1.5)
        ax.add_patch(rect)
        # Inner dot
        ax.scatter(x, y, s=10, c='crimson', marker='o', zorder=11)
    
    def _draw_military_observation(self, ax, item: Dict) -> None:
        x, y = item.get('centroid', (0, 0))
        # Square frame with dot
        size = 10
        rect = Rectangle((x - size/2, y - size/2), size, size,
                         facecolor='none', edgecolor='steelblue', linewidth=1.2)
        ax.add_patch(rect)
        ax.scatter(x, y, s=8, c='steelblue', marker='o', zorder=11)
    
    def _draw_military_chokepoint(self, ax, item: Dict) -> None:
        x, y = item.get('centroid', (0, 0))
        # Diamond with cross
        diamond = Ellipse((x, y), 12, 12, facecolor='none', 
                          edgecolor='orange', linewidth=1.5)
        ax.add_patch(diamond)
        # Simple cross
        ax.plot([x-6, x+6], [y, y], color='orange', linewidth=1)
        ax.plot([x, x], [y-6, y+6], color='orange', linewidth=1)
    
    def _draw_military_assembly(self, ax, item: Dict) -> None:
        x, y = item.get('centroid', (0, 0))
        # Rectangle with flag-like marker
        size = 16
        rect = Rectangle((x - size/2, y - size/2), size, size,
                         facecolor='lightgreen', alpha=0.5, edgecolor='green', linewidth=1)
        ax.add_patch(rect)
        # Flag pole
        ax.plot([x + size/2, x + size/2 + 5], [y, y + 5], color='black', linewidth=1)
    
    def _draw_mgrs_grid(self, ax, shape: Tuple[int, int]) -> None:
        """Draw MGRS-style grid (every 100 pixels approximating 1km)."""
        h, w = shape
        grid_spacing = 100  # pixels
        for x in range(0, w, grid_spacing):
            ax.axvline(x, color='gray', linestyle='-', linewidth=0.3, alpha=0.6)
        for y in range(0, h, grid_spacing):
            ax.axhline(y, color='gray', linestyle='-', linewidth=0.3, alpha=0.6)
    
    def legend_elements(self) -> List:
        from matplotlib.patches import Rectangle
        return [
            Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='crimson', 
                      label='Defensive Position'),
            Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='steelblue', 
                      label='Observation Post'),
            Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='orange', 
                      label='Chokepoint'),
            Rectangle((0, 0), 1, 1, facecolor='lightgreen', alpha=0.5, 
                      edgecolor='green', label='Assembly Area'),
        ]


# =============================================================================
# Orienteering Style (ISOM)
# =============================================================================

# TODO: implement overlays

class OrienteeringVisualizer(BaseVisualizer):
    """
    Orienteering-style map (ISOM standard).
    
    Features: Clean contour lines, minimal overlays, vegetation coloring.
    """
    
    def plot(self, ax: plt.Axes, data: Dict) -> None:
        heightmap = data['heightmap']
        semantic_index = data.get('semantic_index', {})
        
        # Base: elevation tints (white = high, yellow = low, green = mid)
        self._draw_elevation_tint(ax, heightmap)
        
        # Contour lines (brown, standard)
        self._contours(ax, heightmap, levels=15, colors='brown', linewidths=0.4)
        
        # Water features (valleys as blue lines)
        for item in semantic_index.get('ambush_positions', []):
            self._draw_valley_line(ax, item)
        
        # Rock features (ridges as black lines)
        for item in semantic_index.get('cover_positions', []):
            self._draw_ridge_line(ax, item)
        
        # Man-made features (assembly areas as green patches)
        for item in semantic_index.get('assembly_areas', []):
            self._draw_clearing_patch(ax, item)
        
        ax.set_title('Orienteering Map', fontsize=10, fontweight='bold')
        ax.grid(False)
    
    def _draw_elevation_tint(self, ax, heightmap: np.ndarray) -> None:
        """ISOM-style elevation coloring."""
        # White = high, yellow = open, green = forest (simplified)
        norm = plt.Normalize(vmin=np.min(heightmap), vmax=np.max(heightmap))
        cmap = LinearSegmentedColormap.from_list('isom', 
                                                   ['lightgreen', 'yellow', 'white'])
        ax.imshow(heightmap, cmap=cmap, alpha=0.8, interpolation='bilinear')
    
    def _draw_valley_line(self, ax, item: Dict) -> None:
        spine = item.get('spine', [])
        if len(spine) >= 2:
            xs = [p[0] for p in spine]
            ys = [p[1] for p in spine]
            ax.plot(xs, ys, color='blue', linewidth=1, alpha=0.7)
    
    def _draw_ridge_line(self, ax, item: Dict) -> None:
        spine = item.get('spine', [])
        if len(spine) >= 2:
            xs = [p[0] for p in spine]
            ys = [p[1] for p in spine]
            ax.plot(xs, ys, color='black', linewidth=0.8, alpha=0.9)
    
    def _draw_clearing_patch(self, ax, item: Dict) -> None:
        x, y = item.get('centroid', (0, 0))
        area_m2 = item.get('area_m2', 0)
        size = np.clip(np.sqrt(area_m2) / 4, 10, 40)
        rect = Rectangle((x - size/2, y - size/2), size, size,
                         facecolor='yellow', alpha=0.5, edgecolor='none')
        ax.add_patch(rect)
    
    def legend_elements(self) -> List:
        from matplotlib.lines import Line2D
        return [
            Line2D([0], [0], color='brown', linewidth=0.5, label='Contour'),
            Line2D([0], [0], color='blue', linewidth=1, label='Stream/Valley'),
            Line2D([0], [0], color='black', linewidth=0.8, label='Rock/Ridge'),
            Rectangle((0, 0), 1, 1, facecolor='yellow', alpha=0.5, label='Clearing'),
        ]


# =============================================================================
# Visualizer Factory
# =============================================================================

VISUALIZER_MAP = {
    'gis': GISVisualizer,
    'game': GameVisualizer,
    'military': MilitaryVisualizer,
    'orienteering': OrienteeringVisualizer,
}

def get_visualizer(config: PipelineConfig) -> BaseVisualizer:
    """Factory function returning the appropriate visualizer."""
    style = getattr(config, 'visualization_style', 'gis')
    visualizer_class = VISUALIZER_MAP.get(style, GISVisualizer)
    return visualizer_class(config)