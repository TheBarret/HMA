"""
Synthetic Terrain Generator — Ground Truth Heightmaps with Known Features

Generates 128×128 heightmaps with explicitly placed features.
Each feature includes its ground truth location, shape, and properties.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum

from core import Heightmap, NormalizationConfig


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
    centroid: Tuple[float, float]  # pixel coordinates
    properties: Dict[str, Any] = field(default_factory=dict)
    geometry: Optional[Any] = None  # polyline, bounding box, etc.


@dataclass
class SyntheticTerrainResult:
    """Output from synthetic terrain generator."""
    heightmap: Heightmap
    ground_truth: List[GroundTruthFeature]
    metadata: Dict[str, Any] = field(default_factory=dict)


class SyntheticTerrain:
    """
    Generate 128×128 heightmaps with known features for validation.
    
    Features are added as mathematical surfaces and summed.
    """
    
    def __init__(self, size: int = 128, h_scale: float = 2.0, 
                 v_scale: float = 0.2, sea_level: float = 10.0):
        self.size = size
        self.h_scale = h_scale
        self.v_scale = v_scale
        self.sea_level = sea_level
        self.z = np.zeros((size, size), dtype=np.float32) + sea_level
        self.ground_truth: List[GroundTruthFeature] = []
    
    def add_peak(self, x: int, y: int, height: float, radius: float = 10.0,
                 shape: str = "paraboloid") -> None:
        """
        Add a peak at (x,y) with given height and radius.
        
        shape: "paraboloid" (pure convex everywhere) or "gaussian"
        """
        xx, yy = np.meshgrid(np.arange(self.size), np.arange(self.size))
        dx = xx - x
        dy = yy - y
        r2 = dx**2 + dy**2
        r2_max = radius**2
        
        if shape == "paraboloid":
            # z = H * (1 - r²/R²) for r <= R, 0 elsewhere
            peak_z = height * (1 - r2 / r2_max)
            peak_z[r2 > r2_max] = 0
        else:  # gaussian
            sigma = radius / 2
            peak_z = height * np.exp(-r2 / (2 * sigma**2))
        
        self.z += peak_z
        
        self.ground_truth.append(GroundTruthFeature(
            type=FeatureType.PEAK,
            centroid=(float(x), float(y)),
            properties={'height': height, 'radius': radius, 'shape': shape}
        ))
    
    def add_ridge(self, x1: int, y1: int, x2: int, y2: int, 
                  height: float, width: float = 5.0) -> None:
        """
        Add a ridge line from (x1,y1) to (x2,y2).
        Ridge is a Gaussian ridge with given height and width.
        """
        xx, yy = np.meshgrid(np.arange(self.size), np.arange(self.size))
        
        # Distance to line segment
        px, py = x2 - x1, y2 - y1
        seg_len_sq = px*px + py*py
        if seg_len_sq == 0:
            return
        
        t = ((xx - x1) * px + (yy - y1) * py) / seg_len_sq
        t = np.clip(t, 0, 1)
        proj_x = x1 + t * px
        proj_y = y1 + t * py
        dist_to_line = np.sqrt((xx - proj_x)**2 + (yy - proj_y)**2)
        
        # Gaussian ridge profile
        ridge_z = height * np.exp(-dist_to_line**2 / (2 * width**2))
        self.z += ridge_z
        
        self.ground_truth.append(GroundTruthFeature(
            type=FeatureType.RIDGE,
            centroid=((x1+x2)/2, (y1+y2)/2),
            properties={'height': height, 'width': width, 'start': (x1,y1), 'end': (x2,y2)},
            geometry=[(x1,y1), (x2,y2)]
        ))
    
    def add_valley(self, x1: int, y1: int, x2: int, y2: int,
                   depth: float, width: float = 5.0) -> None:
        """
        Add a valley line (negative ridge).
        """
        # Same as ridge but subtract
        xx, yy = np.meshgrid(np.arange(self.size), np.arange(self.size))
        
        px, py = x2 - x1, y2 - y1
        seg_len_sq = px*px + py*py
        if seg_len_sq == 0:
            return
        
        t = ((xx - x1) * px + (yy - y1) * py) / seg_len_sq
        t = np.clip(t, 0, 1)
        proj_x = x1 + t * px
        proj_y = y1 + t * py
        dist_to_line = np.sqrt((xx - proj_x)**2 + (yy - proj_y)**2)
        
        valley_z = -depth * np.exp(-dist_to_line**2 / (2 * width**2))
        self.z += valley_z
        
        self.ground_truth.append(GroundTruthFeature(
            type=FeatureType.VALLEY,
            centroid=((x1+x2)/2, (y1+y2)/2),
            properties={'depth': depth, 'width': width, 'start': (x1,y1), 'end': (x2,y2)},
            geometry=[(x1,y1), (x2,y2)]
        ))
    
    def add_saddle(self, x: int, y: int, height: float,
                   a: float = 0.001, b: float = 0.001,
                   influence_radius: float = 20.0) -> None:
        """
        Add a localized saddle point using hyperboloid: z += a*dx² - b*dy² + height
        tapered to zero outside influence_radius so it doesn't globally offset the map.

        The old implementation added the constant `height` to every pixel, inflating
        the elevation floor and making the saddle extractor's percentile gate useless.
        """
        xx, yy = np.meshgrid(np.arange(self.size), np.arange(self.size))
        dx = xx - x
        dy = yy - y
        saddle_z = a * dx**2 - b * dy**2 + height
        # Cosine taper: full weight at centre, zero at influence_radius
        r = np.sqrt(dx**2 + dy**2)
        window = np.clip((influence_radius - r) / influence_radius, 0.0, 1.0)
        self.z += saddle_z * window
        
        self.ground_truth.append(GroundTruthFeature(
            type=FeatureType.SADDLE,
            centroid=(float(x), float(y)),
            properties={'height': height, 'a': a, 'b': b}
        ))
    
    def add_flat_zone(self, x_min: int, x_max: int, y_min: int, y_max: int,
                      elevation: Optional[float] = None) -> None:
        """
        Add a flat zone (constant elevation) in bounding box.
        """
        if elevation is None:
            elevation = self.sea_level
        
        self.z[y_min:y_max, x_min:x_max] = elevation
        
        self.ground_truth.append(GroundTruthFeature(
            type=FeatureType.FLAT_ZONE,
            centroid=((x_min+x_max)/2, (y_min+y_max)/2),
            properties={'elevation': elevation, 'bounds': (x_min, x_max, y_min, y_max)},
            geometry=[(x_min, y_min), (x_max, y_max)]
        ))
    
    def save_png(self, filepath: str) -> None:
        """
        Save the current heightmap as a grayscale PNG for visual inspection.
        
        Converts elevation data to 0-255 grayscale using the configured vertical scale.
        """
        from PIL import Image
        
        # Convert elevation to grayscale (0-255)
        # elevation = (grayscale * v_scale) + offset
        # so grayscale = (elevation - offset) / v_scale
        grayscale = (self.z - self.sea_level) / self.v_scale
        grayscale = np.clip(grayscale, 0, 255).astype(np.uint8)
        
        img = Image.fromarray(grayscale, mode='L')
        img.save(filepath)
        
        print(f"Saved heightmap to: {filepath}")
    
    def build(self) -> SyntheticTerrainResult:
        """
        Build final heightmap with config and return result.
        """
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
            pixel_to_world=lambda p: (p[0] * self.h_scale, p[1] * self.h_scale),
            origin=(0.0, 0.0)
        )
        
        return SyntheticTerrainResult(
            heightmap=heightmap,
            ground_truth=self.ground_truth,
            metadata={'size': self.size, 'h_scale': self.h_scale, 'v_scale': self.v_scale}
        )