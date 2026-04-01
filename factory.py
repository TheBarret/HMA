"""
STG Factory (Synthetic Terrain Generator)
Ground Truth Heightmaps generator with `predictable` features

Config-aware: uses PipelineConfig to determine scale,
synthetic terrain is generated at the same scale the pipeline will process.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum

from core import Heightmap, NormalizationConfig, PipelineConfig


class FeatureType(Enum):
    PEAK = "peak"
    RIDGE = "ridge"
    VALLEY = "valley"
    SADDLE = "saddle"
    FLAT_ZONE = "flat_zone"


@dataclass
class GroundTruthFeature:
    """Ground truth for a single terrain feature."""
    type: FeatureType
    centroid_px: Tuple[float, float]  # pixel coordinates
    centroid_world: Tuple[float, float]  # world coordinates (meters)
    properties: Dict[str, Any] = field(default_factory=dict)
    geometry_px: Optional[Any] = None
    geometry_world: Optional[Any] = None


@dataclass
class SyntheticTerrainResult:
    """Output from synthetic terrain generator."""
    heightmap: Heightmap
    ground_truth: List[GroundTruthFeature]
    config: PipelineConfig
    metadata: Dict[str, Any] = field(default_factory=dict)


class SyntheticTerrain:
    """
    Generate heightmaps with known features, scaled to PipelineConfig.
    
    All feature sizes (radius, width, etc.) are specified in WORLD METERS,
    then converted to pixels using the config's horizontal_scale.
    This ensures features are physically meaningful at any scale.
    """
    
    def __init__(self, config: PipelineConfig, size_px: int = 128):
        """
        Args:
            config: PipelineConfig that will be used for processing.
                    Determines horizontal_scale, vertical_scale, etc.
            size_px: Size of square heightmap in pixels.
        """
        self.config = config
        self.size_px = size_px
        self.h_scale = config.horizontal_scale
        self.v_scale = config.vertical_scale
        self.sea_level = config.sea_level_offset
        
        # Initialize empty heightmap (sea level)
        self.z = np.full((size_px, size_px), self.sea_level, dtype=np.float32)
        self.ground_truth: List[GroundTruthFeature] = []
        
        # gentle tilt to avoid perfectly flat terrain (prevents gradient issues)
        # Tilt: 0.5m rise across map width = ~0.22° slope
        x = np.linspace(-1, 1, size_px)
        y = np.linspace(-1, 1, size_px)
        xx, yy = np.meshgrid(x, y)
        tilt = xx * 0.5  # 0.5m from left to right edge
        self.z += tilt
    
    def _meters_to_pixels(self, meters: float) -> float:
        """Convert world meters to pixels at current horizontal scale."""
        return meters / self.h_scale
    
    def _pixels_to_meters(self, pixels: float) -> float:
        """Convert pixels to world meters."""
        return pixels * self.h_scale
    
    def add_peak(self, x_world: float, y_world: float, height_m: float, 
                 radius_m: float = 10.0, shape: str = "paraboloid") -> None:
        """
        Add a peak at world coordinates (x_world, y_world) in meters.
        
        Args:
            x_world, y_world: Peak center in world meters
            height_m: Peak height above sea level in meters
            radius_m: Peak radius in meters (where height drops to 0 for paraboloid)
            shape: "paraboloid" or "gaussian"
        """
        # Convert to pixel coordinates
        x_px = x_world / self.h_scale + self.size_px / 2
        y_px = y_world / self.h_scale + self.size_px / 2
        radius_px = radius_m / self.h_scale
        
        # Create meshgrid in pixel space
        xx, yy = np.meshgrid(np.arange(self.size_px), np.arange(self.size_px))
        dx = xx - x_px
        dy = yy - y_px
        r2 = dx**2 + dy**2
        r2_max = radius_px**2
        
        if shape == "paraboloid":
            peak_z = height_m * (1 - r2 / r2_max)
            peak_z[r2 > r2_max] = 0
        else:  # gaussian
            sigma_px = radius_px / 2
            peak_z = height_m * np.exp(-r2 / (2 * sigma_px**2))
        
        self.z += peak_z
        
        self.ground_truth.append(GroundTruthFeature(
            type=FeatureType.PEAK,
            centroid_px=(x_px, y_px),
            centroid_world=(x_world, y_world),
            properties={
                'height_m': height_m, 
                'radius_m': radius_m, 
                'shape': shape,
                'prominence_m': height_m - self.sea_level
            }
        ))
    
    def add_ridge(self, x1_world: float, y1_world: float, 
                  x2_world: float, y2_world: float,
                  height_m: float, width_m: float = 5.0) -> None:
        """
        Add a ridge line from (x1,y1) to (x2,y2) in world meters.
        
        Args:
            x1_world, y1_world: Start point in meters
            x2_world, y2_world: End point in meters
            height_m: Ridge height above sea level in meters
            width_m: Ridge width (standard deviation of Gaussian profile) in meters
        """
        # Convert to pixel coordinates
        cx = self.size_px / 2
        x1_px = x1_world / self.h_scale + cx
        y1_px = y1_world / self.h_scale + cx
        x2_px = x2_world / self.h_scale + cx
        y2_px = y2_world / self.h_scale + cx
        
        width_px = width_m / self.h_scale
        
        xx, yy = np.meshgrid(np.arange(self.size_px), np.arange(self.size_px))
        
        # Distance to line segment
        px, py = x2_px - x1_px, y2_px - y1_px
        seg_len_sq = px*px + py*py
        if seg_len_sq == 0:
            return
        
        t = ((xx - x1_px) * px + (yy - y1_px) * py) / seg_len_sq
        t = np.clip(t, 0, 1)
        proj_x = x1_px + t * px
        proj_y = y1_px + t * py
        dist_to_line = np.sqrt((xx - proj_x)**2 + (yy - proj_y)**2)
        
        # Gaussian ridge profile
        ridge_z = height_m * np.exp(-dist_to_line**2 / (2 * width_px**2))
        self.z += ridge_z
        
        self.ground_truth.append(GroundTruthFeature(
            type=FeatureType.RIDGE,
            centroid_px=((x1_px + x2_px)/2, (y1_px + y2_px)/2),
            centroid_world=((x1_world + x2_world)/2, (y1_world + y2_world)/2),
            properties={
                'height_m': height_m, 
                'width_m': width_m, 
                'start_world': (x1_world, y1_world), 
                'end_world': (x2_world, y2_world)
            },
            geometry_px=[(x1_px, y1_px), (x2_px, y2_px)],
            geometry_world=[(x1_world, y1_world), (x2_world, y2_world)]
        ))
    
    def add_valley(self, x1_world: float, y1_world: float,
                   x2_world: float, y2_world: float,
                   depth_m: float, width_m: float = 5.0) -> None:
        """
        Add a valley line (negative ridge) in world coordinates.
        """
        # Same as ridge but subtract
        cx = self.size_px / 2
        x1_px = x1_world / self.h_scale + cx
        y1_px = y1_world / self.h_scale + cx
        x2_px = x2_world / self.h_scale + cx
        y2_px = y2_world / self.h_scale + cx
        
        width_px = width_m / self.h_scale
        
        xx, yy = np.meshgrid(np.arange(self.size_px), np.arange(self.size_px))
        
        px, py = x2_px - x1_px, y2_px - y1_px
        seg_len_sq = px*px + py*py
        if seg_len_sq == 0:
            return
        
        t = ((xx - x1_px) * px + (yy - y1_px) * py) / seg_len_sq
        t = np.clip(t, 0, 1)
        proj_x = x1_px + t * px
        proj_y = y1_px + t * py
        dist_to_line = np.sqrt((xx - proj_x)**2 + (yy - proj_y)**2)
        
        valley_z = -depth_m * np.exp(-dist_to_line**2 / (2 * width_px**2))
        self.z += valley_z
        
        self.ground_truth.append(GroundTruthFeature(
            type=FeatureType.VALLEY,
            centroid_px=((x1_px + x2_px)/2, (y1_px + y2_px)/2),
            centroid_world=((x1_world + x2_world)/2, (y1_world + y2_world)/2),
            properties={
                'depth_m': depth_m, 
                'width_m': width_m, 
                'start_world': (x1_world, y1_world), 
                'end_world': (x2_world, y2_world)
            },
            geometry_px=[(x1_px, y1_px), (x2_px, y2_px)],
            geometry_world=[(x1_world, y1_world), (x2_world, y2_world)]
        ))
    
    def add_saddle(self, x_world: float, y_world: float, height_m: float,
                   a: float = 0.001, b: float = 0.001,
                   influence_radius_m: float = 20.0) -> None:
        """
        Add a localized saddle point.
        
        Args:
            x_world, y_world: Saddle center in world meters
            height_m: Saddle elevation at center
            a, b: Curvature parameters (z = a*dx² - b*dy² + height)
            influence_radius_m: Radius in meters over which saddle tapers to zero
        """
        cx = self.size_px / 2
        x_px = x_world / self.h_scale + cx
        y_px = y_world / self.h_scale + cx
        influence_radius_px = influence_radius_m / self.h_scale
        
        xx, yy = np.meshgrid(np.arange(self.size_px), np.arange(self.size_px))
        dx_px = xx - x_px
        dy_px = yy - y_px
        
        # Convert curvature parameters to pixel space
        # a and b are in 1/m², need to scale to 1/pixel²
        a_px = a * (self.h_scale ** 2)
        b_px = b * (self.h_scale ** 2)
        
        saddle_z = a_px * dx_px**2 - b_px * dy_px**2 + height_m
        
        # Taper to zero at influence radius
        r_px = np.sqrt(dx_px**2 + dy_px**2)
        window = np.clip((influence_radius_px - r_px) / influence_radius_px, 0.0, 1.0)
        self.z += saddle_z * window
        
        self.ground_truth.append(GroundTruthFeature(
            type=FeatureType.SADDLE,
            centroid_px=(x_px, y_px),
            centroid_world=(x_world, y_world),
            properties={'height_m': height_m, 'a': a, 'b': b, 'influence_radius_m': influence_radius_m}
        ))
    
    def add_flat_zone(self, x_min_world: float, x_max_world: float,
                      y_min_world: float, y_max_world: float,
                      elevation_m: Optional[float] = None) -> None:
        """
        Add a flat zone (constant elevation) in world coordinates.
        """
        if elevation_m is None:
            elevation_m = self.sea_level
        
        cx = self.size_px / 2
        x_min_px = int(x_min_world / self.h_scale + cx)
        x_max_px = int(x_max_world / self.h_scale + cx)
        y_min_px = int(y_min_world / self.h_scale + cx)
        y_max_px = int(y_max_world / self.h_scale + cx)
        
        self.z[y_min_px:y_max_px, x_min_px:x_max_px] = elevation_m
        
        self.ground_truth.append(GroundTruthFeature(
            type=FeatureType.FLAT_ZONE,
            centroid_px=((x_min_px + x_max_px)/2, (y_min_px + y_max_px)/2),
            centroid_world=((x_min_world + x_max_world)/2, (y_min_world + y_max_world)/2),
            properties={
                'elevation_m': elevation_m,
                'bounds_world': (x_min_world, x_max_world, y_min_world, y_max_world)
            }
        ))
    
    def save_png(self, filepath: str) -> None:
        """Save current heightmap as grayscale PNG."""
        from PIL import Image
        
        grayscale = (self.z - self.sea_level) / self.v_scale
        grayscale = np.clip(grayscale, 0, 255).astype(np.uint8)
        img = Image.fromarray(grayscale, mode='L')
        img.save(filepath)
        print(f"Saved heightmap to: {filepath}")
    
    def build(self) -> SyntheticTerrainResult:
        """Build final heightmap with config and return result."""
        cfg = NormalizationConfig(
            horizontal_scale=self.h_scale,
            vertical_scale=self.v_scale,
            sea_level_offset=self.sea_level
        )
        
        # Clip to valid elevation range
        max_elev = 255 * self.v_scale + self.sea_level
        self.z = np.clip(self.z, self.sea_level, max_elev - 0.1)
        
        heightmap = Heightmap(
            data=self.z.astype(np.float32),
            config=cfg,
            pixel_to_world=lambda p: (p[0] * self.h_scale - self.size_px * self.h_scale / 2,
                                       p[1] * self.h_scale - self.size_px * self.h_scale / 2),
            origin=(-self.size_px * self.h_scale / 2, -self.size_px * self.h_scale / 2)
        )
        
        return SyntheticTerrainResult(
            heightmap=heightmap,
            ground_truth=self.ground_truth,
            config=self.config,
            metadata={'size_px': self.size_px, 'h_scale': self.h_scale, 'v_scale': self.v_scale}
        )