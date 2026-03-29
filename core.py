import numpy as np
import os.path
from pathlib import Path
from enum import auto, Enum
from typing import TypeAlias
from typing import Dict, Any
from typing import Optional
from typing import List, Set
from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar
from dataclasses import dataclass, field, KW_ONLY
from uuid import uuid4
import hashlib
import json

#
# TYPES - Primitive Contracts
#

# Coordinate systems
PixelCoord: TypeAlias = tuple[int, int]                 # (x, y) in image space
WorldCoord: TypeAlias = tuple[float, float]             # (easting, northing) in meters
GridCoord: TypeAlias = tuple[float, float, float]       # (x, y, elevation) in meters

# Mathematical primitives
ScalarField: TypeAlias = np.ndarray                     # 2D array of float32
VectorField: TypeAlias = tuple[np.ndarray, np.ndarray]  # (dx, dy) components

# Typed bundle passed between layers, explicit over opaque dicts
# Keys defined per-layer in output_schema
LayerBundle: TypeAlias = Dict[str, Any]  

# Feature classification

class CurvatureType(Enum):
    CONVEX = auto()                 # H > 0, K > 0 (dome/peak)
    CONCAVE = auto()                # H < 0, K > 0 (bowl/depression)
    CYLINDRICAL_CONVEX = auto()     # H > 0, K ≈ 0 (ridge)
    CYLINDRICAL_CONCAVE = auto()    # H < 0, K ≈ 0 (valley)
    SADDLE = auto()                 # K < 0
    FLAT = auto()                   # H ≈ 0, K ≈ 0
    
    def __repr__(self):
        return self.name
    
    def __str__(self):
        return self.name

class Traversability(Enum):
    FREE = auto()
    DIFFICULT = auto()
    BLOCKED = auto()
    
    def __repr__(self):
        return self.name
    
    def __str__(self):
        return self.name

#
#  Vehicle profile
#

@dataclass
class Vehicle:
    """Vehicle mobility characteristics"""
    name: str
    max_slope_deg: float
    max_water_depth_m: float = 0.0
    width_m: float = 2.5
    length_m: float = 5.0
    
    def can_traverse_slope(self, slope_deg: float) -> bool:
        return slope_deg <= self.max_slope_deg
    
    def __repr__(self):
        return f"Vehicle({self.name}, max_slope={self.max_slope_deg}°)"

# =========================================================================
#  MASTER PIPELINE CONFIG
# =========================================================================

@dataclass
class PipelineConfig:
    """Global configuration for the analysis pipeline."""
    
    # =========================================================================
    # LAYER 0: CALIBRATION - Input Processing
    # =========================================================================
    
    horizontal_scale: float = 2.0           # [0.5-10.0] meters per pixel
    vertical_scale: float = 0.2             # [0.05-1.0] meters per grayscale unit
    sea_level_offset: float = 0.0          # [0-100] meters, base elevation
    noise_reduction_sigma: float = 2.0      # [0.5-5.0] Gaussian blur strength
    max_elevation_range: int = 10000        # max elevation bounds [10km range]
    histogram_bins: int = 50                # histogram quantizing pool
    std_gaussian: float = 0.6745            # stddev conversion factor for a Gaussian [0.6745]
    
    
    # =========================================================================
    # LAYER 1: LOCAL GEOMETRY - Slope & Aspect
    # =========================================================================
    
    gradient_method: str = "central_difference"  # ["central_difference", "sobel"]
    flat_threshold_deg: float = 0.5              # [0-2] degrees, aspect undefined below this
    
    # =========================================================================
    # LAYER 2: REGIONAL GEOMETRY - Curvature
    # =========================================================================
    
    adaptive_epsilon: bool = True               # [True/False] auto-tune thresholds
    curvature_epsilon: float = 0.00001          # [1e-6-1e-3] static fallback (1/m)
    curvature_epsilon_h_factor: float = 0.25    # [0.3-0.9] multiplier for mean curvature
    curvature_epsilon_k_factor: float = 0.35     # [0.3-0.9] multiplier for Gaussian curvature
    
    #    Terrain Type    |  adaptive percentile | Rationale
    # -------------------------------------------------------------------------------------------------------
    #    Mountainous     |  75-85               | High curvature variance, can afford higher threshold
    #    Rolling Hills	 |  60-70               | Moderate features, need sensitivity
    #    Sparse Features |  50-60               | Isolated ridges/valleys on flat terrain (default)
    #    Urban/Man-made  |  80-90               | Sharp edges dominate, filter out noise
    adaptive_percentile: float = 45.0
    
    # prevent over-deflation
    curvature_epsilon_h_min: float = 0.00005     # [1e-6-1e-2] minimum mean curvature (1/m)
    curvature_epsilon_k_min: float = 0.000005    # [1e-7-1e-3] minimum Gaussian curvature (1/m²)
    
    # prevent over-inflation
    curvature_epsilon_h_max: float = 0.05       # [0.01-0.2] maximum mean curvature threshold (1/m)
    curvature_epsilon_k_max: float = 0.001      # [0.0001-0.01] maximum Gaussian curvature threshold (1/m²)
    
    # =========================================================================
    # LAYER 3: TOPOLOGY - Feature Detection
    # =========================================================================
    
    # --- Peaks (local maxima with convex surroundings) ---
    peak_confidence : float = 20.0              # [0-50] Peak confidence
    min_peak_size_px: int = 1                   # [1-10] pixels, minimum peak footprint
    peak_min_prominence_m: float = 5.0          # [1-50] meters, minimum height above saddle
    peak_nms_radius_px: int = 25                # [5-30] pixels, non-maximum suppression radius
    peak_shoulder_convex_ratio: float = 0.18    # [0.01-0.3] convex pixels in annular ring
    peak_annular_inner_m: float = 5.0           # [2-15] meters, inner shoulder radius
    peak_annular_outer_m: float = 20.0          # [5-25] meters, outer shoulder radius
    peak_smooth_sigma: float = 1.5              # ndimage.gaussian_filter(z, sigma)
    
    # --- Ridges & Valleys (linear features) ---
    min_ridge_length_px: int = 10                # [3-20] pixels, minimum ridge length
    min_valley_length_px: int = 10               # [3-20] pixels, minimum valley length
    
    # --- Flat Zones (traversable areas) ---
    min_flat_zone_size_px: int = 450            # [50-500] pixels, minimum flat area
    flat_zone_slope_threshold_deg: float = 4.0  # [1-10] degrees, max slope for "flat"
    
    # --- Saddles (passes between peaks) ---
    saddle_k_min_threshold: float = 0.00050     # [1e-6-1e-2] minimum |K| for saddle (1/m²)
    saddle_confidence_threshold: float = 1.0    # [0.0-1.0] normalized confidence (1.0 = all)

    # --- Sea Level ---
    exclude_below_reference: bool = True         # [True/False] exclude sea domain features
    elevation_reference_m: float = 8.0           # meters
        
    # --- General Topology ---
    border_margin_px: int = 10                   # [5-30] pixels, ignore edges
    prominence_search_radius_m: float = 100.0    # [50-500] meters, search radius for prominence
    feature_connection_tolerance_px: float = 10.0 # [1-10] connection tolerance (T pixels at (Template)m/px = N meters)
    
    
    # =========================================================================
    # LAYER 4: RELATIONAL - Connectivity & Flow
    # =========================================================================
    
    
    # --- Visibility ---
    visibility_max_range_m: float = 1000.0      # [200-3000] meters, max line-of-sight distance
    viewshed_sample_step_px: int = 6            # [1-50] pixels, step size for ray casting (performance)
    visibility_sample_radius: int = 5           # [3-15] pixels, sampling radius for viewshed
    visibility_epsilon_m: float = 0.1           # [0.05-0.5] meters, line-of-sight precision
    
    # --- Flow Network ---
    flow_step_px: int = 10                      # [5-20] pixels, step size for flow accumulation
    flow_neighbor_distance_px: int = 50         # [20-100] pixels, max distance to downstream feature
    
    # --- Connectivity ---
    connectivity_max_neighbors: int = 20          # [10-50] max neighbors to consider per feature
    distance_connectivity_fallback_px: int = 50 # [50~] fallcback value
    connection_radius_m: float = 75.0          # [50-500] meters, feature connection radius
    vehicle_climb_angle: float = 25.0           # [20-45] degrees, max slope for vehicles
    cliff_threshold_degrees: float = 45.0       # [30-60] degrees, impassable terrain
    
    # --- Watersheds ---
    watershed_min_area_m2: float = 150.0       # [500-10000] m², minimum watershed area
    watershed_sample_step_px: int = 10          # [5-20] pixels, step size for watershed delineation
    max_watershed_outlets: int = 100             # [50-500] maximum number of outlets to identify
    watershed_area_estimate_factor: float = 100.0  # [50-200] multiplier for feature count → area estimate
    
    
    # =========================================================================
    # LAYER 5: SEMANTICS
    # =========================================================================
    max_feature_coverage: float = 0.5           # Max 50% of map for any single feature
    
    # --- Defensive Positions ---
    threshold_major_peak: float = 15.0          # major peaks [m]
    threshold_minor_peak: float = 5.0           # minor peaks [m]
    saddle_elevation_high: float = 30.0         # saddle elevation high [m]
    saddle_elevation_low: float = 10.0          # saddle elevation low [m]
    valley_avg_slope: float = 15.0              # valley avg slope [m]
    defensive_min_prominence_m: float = 8.0     # [5-30] meters, minimum height advantage
    defensive_min_elevation_m: float = 5.0      # [2-20] meters, minimum absolute height
    defensive_max_slope_deg: float = 25.0       # [15-35] degrees, max slope for defense
    defensive_min_visibility: int = 5           # [3-15] number of visible features
    defensive_prominence_divisor: float = 20.0     # [10-50] meters
    defensive_visibility_divisor: float = 10.0     # [5-20] features
    
    # --- Observation Points ---
    observation_visibility_divisor: float = 15.0   # [10-30] features
    observation_min_prominence_m: float = 10.0  # [5-50] meters, minimum prominence
    observation_min_visibility: int = 10        # [5-30] number of visible features
    
    # --- Assembly Areas ---
    assembly_min_area_m2: float = 2000.0        # [500-10000] square meters, staging area
    assembly_max_slope_deg: float = 5.0         # [2-10] degrees, max slope for assembly
    assembly_major_area_threshold_m2: float = 10000.0  # [5000-25000] m²
    assembly_capacity_divisor: float = 100.0           # [50-200] m² per unit
    
    # --- Chokepoints ---
    chokepoint_min_connectivity: int = 2        # [2-5] minimum connections to be chokepoint
    
    # --- Cover positions ---
    cover_quality_width_divisor: float = 10.0      # [5-20] meters
    cover_min_width_m: float = 5.0              # [2-15] meters, minimum cover width
    
    # --- Drainage classification
    drainage_major_threshold: int = 3             # [1-15~] upstream features
    drainage_minor_threshold: int = 1              # [1-15~] upstream features
    
    # --- Ambush rating
    ambush_slope_divisor: float = 30.0             # [20-45] degrees
    ambush_ridge_divisor: float = 3.0              # [2-5] adjacent ridges

    # --- Trafficability classification
    trafficability_ideal_threshold_deg: float = 5.0    # [3-10] degrees
    trafficability_good_threshold_deg: float = 15.0    # [10-25] degrees
    
    
    # Vehicle profiles
    VEHICLE_PROFILES = {
        'infantry'      : Vehicle('human', max_slope_deg=45.0, max_water_depth_m=0.5),
        'light_wheeled' : Vehicle('car', max_slope_deg=25.0, max_water_depth_m=0.3),
        'heavy_wheeled' : Vehicle('offroad', max_slope_deg=20.0, max_water_depth_m=0.5),
        'tracked'       : Vehicle('army light', max_slope_deg=35.0, max_water_depth_m=1.0),
        'tank'          : Vehicle('army heavy', max_slope_deg=30.0, max_water_depth_m=1.5)
    }
    
    # =========================================================================
    # RUNTIME
    # =========================================================================
    
    verbose: bool = True                        # [True/False] enable debug logging
    cache_dir: str = 'cache'                    # [folder] cache data
   
    # =========================================================================
    # Extensions
    # =========================================================================
    def is_traversable(self, slope_degrees: float) -> Traversability:
        """Classify slope by vehicle constraints."""
        if slope_degrees > self.cliff_threshold_degrees:
            return Traversability.BLOCKED
        elif slope_degrees > self.vehicle_climb_angle:
            return Traversability.DIFFICULT
        return Traversability.FREE
        
#
#  Immutable Data Containers
#

@dataclass(frozen=True)
class NormalizationConfig:
    """Converts pixel/grayscale values to real-world units."""
    horizontal_scale: float  # meters per pixel
    vertical_scale: float    # meters per grayscale unit (0-255)
    sea_level_offset: float = 0.0  # optional Z offset
    
    def normalize_elevation(self, grayscale_value: int) -> float:
        return (grayscale_value * self.vertical_scale) + self.sea_level_offset

@dataclass
class RawImageInput:
    """
    Container for raw input data before calibration.
    
    Allows for flexible input sources while maintaining type safety.
    """
    data: np.ndarray  # 2D array of uint8 (0-255) grayscale values
    metadata: Optional[dict] = None
    
    # Helpers
    def validate(self) -> bool:
        """Validate input data format."""
        # explicitly convert all checks to Python bool (not numpy.bool_)
        if not isinstance(self.data, np.ndarray):
            return False
        if int(self.data.ndim) != 2:
            return False
        if self.data.dtype != np.uint8:
            return False
        if not bool(np.isfinite(self.data).all()):
            return False
        return True

    def get_hash(self) -> str:
        """Generate deterministic hash of the input data."""
        data_hash = hashlib.sha256(self.data.tobytes()).hexdigest()
        if self.metadata:
            meta_str = json.dumps(self.metadata, sort_keys=True)
            meta_hash = hashlib.sha256(meta_str.encode()).hexdigest()
            combined = data_hash + meta_hash
        else:
            combined = data_hash
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

@dataclass(frozen=True)
class Heightmap:
    """
    Immutable calibrated heightmap surface.
    Represents z = f(x, y) as a 2.5D scalar field.
    """
    data: ScalarField                                    # Normalized elevation values in meters
    config: NormalizationConfig
    pixel_to_world: Callable[[PixelCoord], WorldCoord]   # Coordinate transform: image space → real-world meters
    origin: WorldCoord = (0.0, 0.0)                      # Real-world coordinate of (0,0) pixel
    
    @property
    def shape(self) -> PixelCoord:
        return self.data.shape
    
    def elevation_at(self, pixel: PixelCoord) -> Optional[float]:
        x, y = pixel
        H, W = self.data.shape
        if 0 <= x < W and 0 <= y < H:
            return self.data[y, x]
        return None
    
    def world_at(self, pixel: PixelCoord) -> Optional[WorldCoord]:
        """Convert pixel coordinate to real-world (easting, northing) in meters."""
        if self.elevation_at(pixel) is not None:
            return self.pixel_to_world(pixel)
        return None

@dataclass
class TerrainFeature:
    """Abstract base for all detected terrain structures."""
    centroid: PixelCoord
    elevation_range: tuple[float, float]              # (min_z, max_z) in meters
    feature_id: str = field(default_factory=lambda: str(uuid4()))  # Stable unique ID for graph edges
    metadata: dict = field(default_factory=dict)
    
    # To be implemented by subclasses
    def contains_point(self, pixel: PixelCoord) -> bool:
        raise NotImplementedError
        
        
        
#
#  The Layer Protocol
#

        

# Generic type for layer outputs
T = TypeVar('T')

class PipelineLayer(ABC, Generic[T]):
    """
    Abstract base for all analysis pipeline stages.
    
    Enforces the dependency chain: each layer receives output from previous
    and produces input for next. Layers are stateless and idempotent.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    @abstractmethod
    def execute(self, input_data: Heightmap | dict) -> T:
        """
        Process input and return layer-specific output.
        
        Args:
            input_data: Output from previous layer (or Heightmap for Layer 0)
            
        Returns:
            Layer-specific result (e.g., slope map, feature list, etc.)
        """
        pass
    
    @property
    @abstractmethod
    def output_schema(self) -> dict:
        """Describes the structure of returned data for validation."""
        pass
        

"""
    Topological Features Abstractions
    
"""

class ClassifiedFeature(TerrainFeature, ABC):
    """
    A TerrainFeature with geometric classification.
    
    All concrete features (Peak, Ridge, etc.) inherit from this.
    """
    
    @property
    @abstractmethod
    def curvature_type(self) -> CurvatureType:
        pass
    
    @property
    @abstractmethod
    def avg_slope(self) -> float:
        """Average slope magnitude across feature area."""
        pass
    
    @abstractmethod
    def is_traversable(self, config: 'PipelineConfig') -> Traversability:
        """Evaluate traversability against vehicle constraints."""
        pass

#
#  Concrete Feature Types
#         

@dataclass
class PeakFeature(ClassifiedFeature):
    """Local maximum with convex surroundings."""
    _: KW_ONLY
    prominence: float
    visibility_radius: Optional[float] = None
    _avg_slope: float = field(default=15.0, init=False)  # Store internally
    
    @property
    def curvature_type(self) -> CurvatureType:
        return CurvatureType.CONVEX

    @property
    def avg_slope(self) -> float:
        return self._avg_slope

    def is_traversable(self, config: PipelineConfig) -> Traversability:
        return config.is_traversable(self.avg_slope)

    def contains_point(self, pixel: PixelCoord) -> bool:
        x, y = self.centroid
        return abs(pixel[0] - x) < 3 and abs(pixel[1] - y) < 3

@dataclass  
class RidgeFeature(ClassifiedFeature):
    """Linear convex feature connecting peaks."""
    _: KW_ONLY
    spine_points: List[PixelCoord]    # Polyline defining ridge crest
    connected_peaks: Set[str]         # Feature IDs of connected peaks
    _avg_slope: float = field(default=10.0, init=False)
    
    @property
    def curvature_type(self) -> CurvatureType:
        return CurvatureType.CONVEX
   
    def is_traversable(self, config: PipelineConfig) -> Traversability:
        return config.is_traversable(self.avg_slope)
    
    def contains_point(self, pixel: PixelCoord) -> bool:
        """Check if point is within 3 pixels of ridge spine (Manhattan distance)."""
        if not self.spine_points:
            return False
        distances = [abs(pixel[0] - x) + abs(pixel[1] - y)
                     for x, y in self.spine_points]
        return min(distances) < 3
    
    @property
    def avg_slope(self) -> float:
        return self._avg_slope
        
@dataclass
class ValleyFeature(ClassifiedFeature):
    """Local minimum with concave surroundings."""
    _: KW_ONLY
    spine_points: List[PixelCoord] = field(default_factory=list)
    drainage_area: Optional[float] = None
    
    @property
    def curvature_type(self) -> CurvatureType:
        return CurvatureType.CONCAVE
    
    @property
    def avg_slope(self) -> float:
        return float(self.metadata.get('avg_slope', 10.0))
    
    def is_traversable(self, config: 'PipelineConfig') -> Traversability:
        # Valley Traversability Reasoning:
        # - Valleys if conditioned wet OR difficult => penalty
        # - Valleys default to difficult
        base = config.is_traversable(self.avg_slope)
        if base == Traversability.FREE:
            return Traversability.DIFFICULT  
        return base
    
    def contains_point(self, pixel: PixelCoord) -> bool:
        """Check if point is within 5 pixels of valley spine."""
        if not self.spine_points:
            return False
        distances = [np.sqrt((pixel[0]-x)**2 + (pixel[1]-y)**2) 
                    for x, y in self.spine_points]
        return min(distances) < 5
        
@dataclass
class SaddleFeature(ClassifiedFeature):
    """Pass connecting ridges/valleys."""
    _: KW_ONLY
    elevation: float = 0.0
    connecting_ridges: Set[str] = field(default_factory=set)
    connecting_valleys: Set[str] = field(default_factory=set)
    k_curvature: Optional[float] = None  # Gaussian curvature magnitude
    
    @property
    def curvature_type(self) -> CurvatureType:
        return CurvatureType.SADDLE
    
    @property
    def avg_slope(self) -> float:
        return float(self.metadata.get('avg_slope', 10.0))
    
    def is_traversable(self, config: 'PipelineConfig') -> Traversability:
        # Saddles are natural passes, often traversable
        return config.is_traversable(self.avg_slope)
    
    def contains_point(self, pixel: PixelCoord) -> bool:
        x, y = self.centroid
        # Saddle points are localized
        return abs(pixel[0] - x) < 3 and abs(pixel[1] - y) < 3


@dataclass
class FlatZoneFeature(ClassifiedFeature):
    """Large flat area suitable for traversability."""
    _: KW_ONLY
    area_pixels: int = 0
    max_slope: float = 0.0
    min_slope: float = 0.0
    
    @property
    def curvature_type(self) -> CurvatureType:
        return CurvatureType.FLAT
    
    @property
    def avg_slope(self) -> float:
        return float(self.metadata.get('avg_slope', 0.0))
    
    def is_traversable(self, config: 'PipelineConfig') -> Traversability:
        # Flat zones are ideal for travel
        if self.max_slope < config.vehicle_climb_angle:
            return Traversability.FREE
        elif self.max_slope < config.cliff_threshold_degrees:
            return Traversability.DIFFICULT
        return Traversability.BLOCKED
    
    def contains_point(self, pixel: PixelCoord) -> bool:
        # Flat zones are large areas - use bounding box
        if 'bounds' not in self.metadata:
            return False
        x, y = pixel
        x_min, x_max, y_min, y_max = self.metadata['bounds']
        return x_min <= x <= x_max and y_min <= y <= y_max


#
#  The Dependency Chain Executor
#

class Pipeline:
    """
    Executes the analysis pipeline in dependency order.
    
    Enforces: Calibration → Derivatives → Curvature → Topology → Relations → Semantics
    """
    
    def __init__(self, config: PipelineConfig, layers: List[PipelineLayer]):
        self.config = config
        self.layers = layers  # Must be ordered correctly
        
    def run(self, source: Heightmap) -> "AnalyzedTerrain":
        # Initialize the bundle with the source
        bundle: LayerBundle = {"heightmap": source}
        
        for layer in self.layers:
            if isinstance(layer, Layer0_Calibration):
                # Layer 0 creates the Heightmap from raw data (if needed)
                # For this flow, we assume 'source' is already calibrated Heightmap
                continue 
            elif isinstance(layer, Layer1_LocalGeometry):
                result = layer.execute(bundle["heightmap"])
                bundle["slope"] = result["slope"]
                bundle["aspect"] = result["aspect"]
            elif isinstance(layer, Layer2_RegionalGeometry):
                result = layer.execute(bundle["heightmap"])
                bundle["curvature"] = result["curvature"]
                bundle["curvature_type"] = result["curvature_type"]
            elif isinstance(layer, Layer3_TopologicalFeatures):
                result = layer.execute(bundle)  # Pass full bundle
                bundle["features"] = result
            elif isinstance(layer, Layer4_Relational):
                result = layer.execute(bundle)  # Pass full bundle
                bundle.update(result)           # Merge graphs into bundle
            elif isinstance(layer, Layer5_Semantics):
                return layer.execute(bundle)    # Final output
                
        raise RuntimeError("Pipeline has no return object [AnalyzedTerrain]")
        
#
#  Finalizing to an analyzed terrain object
#  

@dataclass
class AnalyzedTerrain:
    """
    The aggregated output of the full analysis pipeline.
    
    A vector-layer representation of terrain where every shape knows:
    - What it is (geometric classification)
    - Where it is (spatial bounds)
    - What it connects to (topology)
    - What it means (semantic tags)
    """
    
    # Source reference
    source_heightmap: Heightmap
    
    # Vector layers (grouped by feature type)
    peaks: List[ClassifiedFeature] = field(default_factory=list)
    valleys: List[ClassifiedFeature] = field(default_factory=list)
    ridges: List[ClassifiedFeature] = field(default_factory=list)
    saddles: List[ClassifiedFeature] = field(default_factory=list)
    flat_zones: List[ClassifiedFeature] = field(default_factory=list)
    
    # Relational data
    visibility_graph: Dict[str, Set[str]] = field(default_factory=dict)   # feature_id → visible feature_ids
    flow_network: Dict[str, List[str]] = field(default_factory=dict)      # feature_id → downstream feature_ids
    connectivity_graph: Dict[str, Set[str]] = field(default_factory=dict) # feature_id → adjacent traversable feature_ids
    watersheds: Dict[str, Set[str]] = field(default_factory=dict)         # basin_id → member feature_ids
    flow_accumulation: Optional[ScalarField] = None                       # per-pixel upstream area (pixels²)
    semantic_index: Dict[str, Any] = field(default_factory=dict)           # tactical index built by Layer 5
    
    # Query interface
    def find_by_type(self, feature_type: type) -> List[ClassifiedFeature]:
        """Return all features of specified class."""
        pass
    
    def find_visible_from(self, feature_id: str) -> Set[str]:
        """Return IDs of features visible from given feature."""
        pass
    
    def find_path(self, start_id: str, end_id: str, 
                  traversability: Traversability = Traversability.FREE) -> Optional[List[str]]:
        """Find least-cost path between features respecting slope constraints."""
        pass
    
    def query(self, **filters) -> List[ClassifiedFeature]:
        """
        Flexible query interface.
        
        Example:
          plane.query(
              type=PeakFeature,
              min_prominence=50.0,
              visibility_coverage={"capture_point_A": 0.7},
              traversable_by=Vehicle(tank=True)
          )
        """
        pass
        
        
# ----------------------------------------------------
#   Auto cache mechanic
#   Supports: layer 0,1 and 3 
# ----------------------------------------------------

class Datacache:
    """
    Per-layer cache manager
    
    Each layer gets its own cache directory and manages its own entries.
    Handles the dual-file pattern (JSON metadata + NPY arrays).
    * Avoid complex (nested) types, break them down or store as (NPY) binary.
    * Make sure the 'name' tags corresponds with filename identifiers.
    """
    
    def __init__(self, layer_name: str, cache_dir: str = "cache"):
        self.layer_name = layer_name
        self.cache_dir = Path(cache_dir)
        
    def _ensure_dir(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def make_key(self, *hashes: str) -> str:
        """Combine multiple hash strings into a single cache key."""
        combined = "".join(hashes)
        combined_sum = hashlib.sha256(combined.encode()).hexdigest()[:16]
        print(f'[cache] hashing: {combined_sum }')
        return combined_sum 
    
    def exists(self, cache_id: str, name: str, extension: str) -> bool:
        """Check if a cache file exists."""
        file_path = self.cache_dir / f"{cache_id}.{self.layer_name}.{name}.{extension}"
        file_exists = file_path.exists()
        print(f'[cache] seek: {file_path}, exists={file_exists}')
        return file_exists
    
    def save_json(self, cache_id: str, name: str, data: dict) -> None:
        """Save metadata as JSON."""
        self._ensure_dir()
        path = self.cache_dir / f"{cache_id}.{self.layer_name}.{name}.json"
        print(f'[cache] writing: {path}...')
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_json(self, cache_id: str, name: str) -> dict:
        """Load metadata JSON."""
        path = self.cache_dir / f"{cache_id}.{self.layer_name}.{name}.json"
        print(f'[cache] reading: {path}...')
        with open(path, 'r') as f:
            return json.load(f)
    
    def save_array(self, cache_id: str, name: str, array: np.ndarray) -> None:
        """Save numpy array as NPY file."""
        self._ensure_dir()
        path = self.cache_dir / f"{cache_id}.{self.layer_name}.{name}.npy"
        np.save(path, array)
    
    def load_array(self, cache_id: str, name: str) -> np.ndarray:
        """Load numpy array from NPY file."""
        path = self.cache_dir / f"{cache_id}.{self.layer_name}.{name}.npy"
        return np.load(path)