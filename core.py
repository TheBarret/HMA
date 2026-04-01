import h5py
import hashlib
import json
import numpy as np
#import os.path
from pathlib import Path
from uuid import uuid4
from enum import auto, Enum
from typing import TypeAlias
from typing import Dict, Any
from typing import Optional
from typing import List, Tuple, Union, Set
from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar
from dataclasses import dataclass, field, KW_ONLY

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

class FeatureType(Enum):
    """Classification of terrain feature types."""
    PEAK = auto()
    RIDGE = auto()
    VALLEY = auto()
    SADDLE = auto()
    FLAT = auto()
    
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
    vertical_scale: float = 0.492             # [0.05-1.0] meters per grayscale unit [default: 0.392]
    sea_level_offset: float = 0.1          # [0-100] meters, base elevation
    noise_reduction_sigma: float = 1.0      # [0.5-5.0] Gaussian blur strength
    max_elevation_range: int = 10000        # max elevation bounds [10km range]
    histogram_bins: int = 50                # histogram quantizing pool
    std_gaussian: float = 0.6745            # stddev conversion factor for a Gaussian [default: 0.6745]
    
    
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
    adaptive_percentile: float = 62.0
    
    # prevent over-deflation
    curvature_epsilon_h_min: float = 0.00005     # [1e-6-1e-2] minimum mean curvature (1/m)
    curvature_epsilon_k_min: float = 0.00002    # [1e-7-1e-3] minimum Gaussian curvature (1/m²)
    
    # prevent over-inflation
    curvature_epsilon_h_max: float = 0.05       # [0.01-0.2] maximum mean curvature threshold (1/m)
    curvature_epsilon_k_max: float = 0.001      # [0.0001-0.01] maximum Gaussian curvature threshold (1/m²)
    
    # =========================================================================
    # LAYER 3: TOPOLOGY - Feature Detection
    # =========================================================================
    
    # --- Peaks (local maxima with convex surroundings) ---
    peak_confidence : float = 20.0              # [0-50] Peak confidence
    min_peak_size_px: int = 1                   # [1-10] pixels, minimum peak footprint
    peak_min_prominence_m: float = 10.0          # [1-50] meters, minimum height above saddle
    peak_nms_radius_px: int = 30               # [1-100] pixels, non-maximum suppression radius [tweak this when saddles are too surpressed]
    peak_shoulder_convex_ratio: float = 0.12    # [0.01-0.3] n% convex pixels in annular ring
    peak_annular_inner_m: float = 2.0           # [2-15] meters, inner shoulder radius
    peak_annular_outer_m: float = 40.0          # [1-50] meters, outer shoulder radius
    peak_smooth_sigma: float = 2.5              # ndimage.gaussian_filter(z, sigma)
    
    # --- Ridges & Valleys (linear features) ---
    min_ridge_length_px: int = 15                # [3-20] pixels, minimum ridge length
    min_valley_length_px: int = 10               # [3-20] pixels, minimum valley length
    
    # --- Flat Zones (traversable areas) ---
    min_flat_zone_size_px: int = 150            # [50-500] pixels, minimum flat area
    flat_zone_slope_threshold_deg: float = 4.0  # [1-10] degrees, max slope for "flat"
    
    # --- Saddles (passes between peaks) ---
    saddle_k_min_threshold: float = 0.00020     # [2.00e-4/m²] minimum |K| for saddle (1/m²)
    saddle_confidence_threshold: float = 0.3    # [0.0-1.0] normalized confidence (1.0 = all)

    # --- Sea Level ---
    exclude_below_reference: bool = True         # [True/False] exclude sea domain features
    elevation_reference_m: float = 1.5           # [0-10] in meters
        
    # --- General Topology ---
    border_margin_px: int = 10                   # [5-30] pixels, ignore edges
    prominence_search_radius_m: float = 50.0    # [50-500] meters, search radius for prominence
    feature_connection_tolerance_px: float = 10.0 # [1-10] connection tolerance (T pixels at (Template)m/px = N meters)
    
    # =========================================================================
    # LAYER 4: RELATIONAL - Connectivity & Flow (REV 2)
    # =========================================================================

    # --- Visibility ---
    visibility_max_range_m: float = 500.0          # [200-3000] meters
    visibility_epsilon_m: float = 0.1              # [0.05-0.5] meters, LOS precision
    visibility_observer_height_m: float = 1.7      # [0-10] meters, observer eye height
    visibility_target_height_m: float = 0.0        # [0-10] meters, target ground height

    # --- Flow Network ---
    flow_direction_method: str = "d8"              # ['d8'] supported: d8 only
    stream_accumulation_threshold_px: int = 65     # [10-500] pixels
    feature_snap_distance_px: int = 10             # [5-20] pixels
    snap_method: str = "nearest"                   # ["nearest", "flow_directed", "none"]
    include_edge_basins: bool = True               # include basins draining off-map

    # --- Connectivity ---
    connection_radius_m: float = 50.0              # [50-500] meters                        [performance tweaker]
    connectivity_max_neighbors: int = 16           # [4-50] max graph degree
    vehicle_climb_angle: float = 25.0              # [20-45] degrees
    cliff_threshold_degrees: float = 45.0          # [30-60] degrees

    # --- Cost Surface ---
    traversability_cost_function: str = "tobler"   # ["tobler", "vehicle_quadratic", "custom"]
    traversability_unit: str = "time"              # ["time", "energy", "risk"]
    cost_slope_weight: float = 1.0                 # [0.5-2.0]
    cost_curvature_weight: float = 0.3             # [0.1-0.5]
    cost_roughness_weight: float = 0.2             # [0.0-0.5]

    # --- Observer/target offsets ---
    ridge_visibility_scale: float = 0.8            # Scale factor for ridge observer height

    # --- Pathfinding ---
    connectivity_max_cost: float = 1000.0          # float('inf') or maximum allowed path cost [performance tweaker]
    connectivity_max_visited_ratio: float = 0.45  # Max % of search space to explore          [performance tweaker]
    connectivity_heuristic_buffer: float = 1.5   # heuristic_buffer                          [performance tweaker]
    
    # --- Watersheds ---
    watershed_min_area_m2: float = 750.0           # [500-10000] m² (note: any lower at 500 at 2m/px scale = 62.5 pixels, may include noise basins)
    
    # --- Fallbacks ---
    skimage_stream_min_size_px: int = 10           # extract_stream_network() -> skimage.morphology.remove_small_objects(<object>, <size>)
    
    
    
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
    centroid: PixelCoord                                            # Pixel coordinate container
    elevation_range: tuple[float, float]                            # (min_z, max_z) in meters
    feature_id: str = field(default_factory=lambda: str(uuid4()))   # Stable unique ID for graph edges
    metadata: dict = field(default_factory=dict)                    # metadata container
        
        
#
#  The Layer Protocol
#

T = TypeVar('T') # Generic type for layer outputs

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

@dataclass
class ClassifiedFeature(TerrainFeature):
    """Base class for classified terrain features."""
    _: KW_ONLY
    feature_type: FeatureType = FeatureType.PEAK      # default
    confidence: float = 0.0                           # default
    metadata: Dict[str, Any] = field(default_factory=dict)
    _avg_slope: float = field(default=0.0, init=False)
    _avg_curvature: float = field(default=0.0, init=False)
    
    @property
    def avg_slope(self) -> float:
        """Average slope of feature region."""
        return self._avg_slope
    
    @property
    def avg_curvature(self) -> float:
        """Average curvature of feature region."""
        return self._avg_curvature
    
    @abstractmethod
    def curvature_type(self) -> CurvatureType:
        """Return curvature classification."""
        pass
    
    @abstractmethod
    def is_traversable(self, config: PipelineConfig) -> Traversability:
        """Determine if feature is traversable."""
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
    feature_type: FeatureType = FeatureType.PEAK      # default
    confidence: float = 0.0                           # default
    watershed_id: Optional[str] = None
    flow_distance_m: Optional[float] = None         # Distance to outlet via flow
    is_evacuation_point: bool = False               # Semantic flag (Layer 5)
    
    _avg_slope: float = field(default=15.0, init=False)
    
    @property
    def curvature_type(self) -> CurvatureType:
        return CurvatureType.CONVEX
    
    @property
    def avg_slope(self) -> float:
        return self._avg_slope
    
    def is_traversable(self, config: 'PipelineConfig') -> Traversability:
        """Peak traversability (no penalty—peaks are high ground)."""
        return config.is_traversable(self.avg_slope)
    
    def contains_point(self, pixel: PixelCoord) -> bool:
        """
        Check if point is within prominence radius of peak.
        NOTE: May be deprecated in favor of watershed label lookup.
        """
        dx = pixel[0] - self.centroid[0]
        dy = pixel[1] - self.centroid[1]
        distance = np.sqrt(dx*dx + dy*dy)
        return distance < self.prominence / config.horizontal_scale

@dataclass
class RidgeFeature(ClassifiedFeature):
    """Linear convex feature (drainage divide)."""
    _: KW_ONLY
    spine_points: List[PixelCoord] = field(default_factory=list)
    adjacent_watersheds: List[str] = field(default_factory=list)
    is_primary_divide: bool = False
    connected_peaks: Set[str] = field(default_factory=set) # SET BY TOPOLOGY._find_connected_peaks() 
    
    @property
    def curvature_type(self) -> CurvatureType:
        return CurvatureType.CYLINDRICAL_CONVEX
    
    @property
    def avg_slope(self) -> float:
        return float(self.metadata.get('avg_slope', 15.0))
    
    def is_traversable(self, config: 'PipelineConfig') -> Traversability:
        return config.is_traversable(self.avg_slope)
    
    def contains_point(self, pixel: PixelCoord, config: PipelineConfig) -> bool:
        if not self.spine_points:
            return False
        distances = [np.sqrt((pixel[0]-x)**2 + (pixel[1]-y)**2) 
                    for x, y in self.spine_points]
        tolerance_px = config.feature_connection_tolerance_px
        return min(distances) < tolerance_px
        
@dataclass
class SaddleFeature(ClassifiedFeature):
    """Col or pass between two higher areas."""
    _: KW_ONLY
    connected_peaks: List[str] = field(default_factory=list)  # Peak IDs
    connected_watersheds: List[str] = field(default_factory=list)  # basins on each side
    saddle_elevation_m: Optional[float] = None
    k_curvature: Optional[float] = None
    is_key_col: bool = False  # primary connection between major basins
    
    connecting_ridges: List[str] = field(default_factory=list) # SET BY TOPOLOGY._build_feature_hierarchy()
    connecting_valleys: List[str] = field(default_factory=list) # SET BY TOPOLOGY._build_feature_hierarchy()
    
    @property
    def curvature_type(self) -> CurvatureType:
        # Saddle is convex in one direction, concave in the other
        return CurvatureType.SADDLE  # may need to add this enum value
    
    @property
    def avg_slope(self) -> float:
        return float(self.metadata.get('avg_slope', 10.0))
    
    def is_traversable(self, config: 'PipelineConfig') -> Traversability:
        # Saddles are natural traversal points (passes)
        base = config.is_traversable(self.avg_slope)
        # No penalty—saddles are often easiest crossing points
        return base
    
    def get_ascent_angle(self, from_peak_id: str) -> float:
        """Get slope angle from a specific peak to saddle."""
        # Useful for route planning
        pass

@dataclass
class ValleyFeature(ClassifiedFeature):
    """Local minimum with concave surroundings."""
    _: KW_ONLY
    spine_points: List[PixelCoord] = field(default_factory=list)
    drainage_area: Optional[float] = None
    
    watershed_id: Optional[str] = None              # Basin membership
    stream_order: int = 1                           # Strahler order
    distance_to_outlet_m: Optional[float] = None    # Flow path length to outlet
    is_flood_prone: bool = False                    # Semantic flag (Layer 5)
    
    @property
    def curvature_type(self) -> CurvatureType:
        return CurvatureType.CONCAVE
    
    @property
    def avg_slope(self) -> float:
        return float(self.metadata.get('avg_slope', 10.0))
    
    def is_traversable(self, config: 'PipelineConfig') -> Traversability:
        """Valley traversability with wetness penalty."""
        base = config.is_traversable(self.avg_slope)
        if base == Traversability.FREE:
            return Traversability.DIFFICULT  
        return base
    
    def contains_point(self, pixel: PixelCoord) -> bool:
        """
        Check if point is within 5 pixels of valley spine.
        NOTE: May be deprecated in favor of watershed label lookup.
        """
        if not self.spine_points:
            return False
        distances = [np.sqrt((pixel[0]-x)**2 + (pixel[1]-y)**2) 
                    for x, y in self.spine_points]
        return min(distances) < 5

@dataclass
class FlatZoneFeature(ClassifiedFeature):
    """Area with near-zero slope."""
    _: KW_ONLY
    area_pixels: int = 0
    watershed_ids: List[str] = field(default_factory=list)  # can span multiple
    is_wetland: bool = False  # semantic flag (Layer 5)
    is_flood_zone: bool = False  # flood risk flag
    flow_direction_ambiguity: float = 0.0  # 0=uniform, 1=chaotic
    max_slope: Optional[float] = None
    min_slope: Optional[float] = None
    
    @property
    def curvature_type(self) -> CurvatureType:
        # Flat zones are flat (neither convex nor concave)
        return CurvatureType.FLAT 
    
    @property
    def get_max_slope(self) -> float:
        return self.metadata.get('max_slope', self.avg_slope)
        
    @property
    def get_min_slope(self) -> float:
        return self.metadata.get('min_slope', self.avg_slope)
    
    @property
    def avg_slope(self) -> float:
        return float(self.metadata.get('avg_slope', 1.0))  # Low by definition
    
    def is_traversable(self, config: 'PipelineConfig') -> Traversability:
        # Flat zones are traversable but may be wet
        base = config.is_traversable(self.avg_slope)
        if self.is_wetland and base == Traversability.FREE:
            return Traversability.DIFFICULT
        return base
    
    def get_dominant_flow(self) -> Optional[PixelCoord]:
        """Determine primary flow direction across flat zone."""
        # Useful for flood modeling
        pass


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
#  Finalizing to an analyzed terrain object (LLM CONTEXT)
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
    
    Also provides LLM-friendly query interfaces and pre-computed context.
    """
    
    # =========================================================================
    # CORE DATA (Populated by pipeline layers)
    # =========================================================================
    
    # Source reference
    source_heightmap: Heightmap
    
    # Vector layers (grouped by feature type)
    peaks: List[ClassifiedFeature] = field(default_factory=list)
    valleys: List[ClassifiedFeature] = field(default_factory=list)
    ridges: List[ClassifiedFeature] = field(default_factory=list)
    saddles: List[ClassifiedFeature] = field(default_factory=list)
    flat_zones: List[ClassifiedFeature] = field(default_factory=list)
    
    # Relational data (populated by Layer 4)
    visibility_graph: Dict[str, Set[str]] = field(default_factory=dict)
    flow_network: Dict[str, List[str]] = field(default_factory=dict)
    connectivity_graph: Dict[str, Set[str]] = field(default_factory=dict)
    watersheds: Dict[str, Set[str]] = field(default_factory=dict)
    flow_accumulation: Optional[ScalarField] = None
    
    # Semantic data (populated by Layer 5)
    semantic_index: Dict[str, Any] = field(default_factory=dict)
    
    # =========================================================================
    # LLM-FRIENDLY PRE-COMPUTED DATA
    # =========================================================================
    
    # Natural language context (after pipeline, semantics)
    terrain_narrative: str = ""
    region_descriptions: Dict[str, str] = field(default_factory=dict)  # "north": "steep mountainous terrain"
    feature_descriptions: Dict[str, str] = field(default_factory=dict)  # feature_id → human description
    
    # Pre-computed tactical answers
    tactical_index: Dict[str, Any] = field(default_factory=dict)
    
    # Statistical summaries
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    
    # =========================================================================
    # INTERNAL: Lazy-loaded spatial indexes (not serialized)
    # =========================================================================
    
    _spatial_index: Any = field(default=None, init=False, repr=False)
    _index_built: bool = field(default=False, init=False, repr=False)
    
    # =========================================================================
    # QUERY INTERFACE
    # =========================================================================
    
    def find_by_type(self, feature_type: type) -> List[ClassifiedFeature]:
        """Return all features of specified class."""
        type_map = {
            PeakFeature: self.peaks,
            RidgeFeature: self.ridges,
            ValleyFeature: self.valleys,
            SaddleFeature: self.saddles,
            FlatZoneFeature: self.flat_zones,
        }
        return type_map.get(feature_type, [])
    
    def find_visible_from(self, feature_id: str) -> Set[str]:
        """Return IDs of features visible from given feature."""
        return self.visibility_graph.get(feature_id, set())
    
    def find_path(self, start_id: str, end_id: str, 
                  traversability: Traversability = Traversability.FREE) -> Optional[List[str]]:
        """
        Find least-cost path between features respecting slope constraints.
        Uses search *fast* algorithm on connectivity_graph.
        """
        if not self.connectivity_graph:
            return None
        
        # Simple BFS
        # TODO: upgrade to better algorithm
        
        from collections import deque
        
        if start_id not in self.connectivity_graph or end_id not in self.connectivity_graph:
            return None
        
        queue = deque([(start_id, [start_id])])
        visited = {start_id}
        
        while queue:
            current, path = queue.popleft()
            
            if current == end_id:
                return path
            
            for neighbor in self.connectivity_graph.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def query(self, **filters) -> List[ClassifiedFeature]:
        """
        Flexible query interface.
        
        Examples:
          terrain.query(type=PeakFeature, min_prominence=50.0)
          terrain.query(type=RidgeFeature, min_length=100)
          terrain.query(type=FlatZoneFeature, min_area_m2=5000)
          terrain.query(type=PeakFeature, defensive_rating__gt=0.7)
        """
        # Extract filter parameters
        feature_type = filters.pop('type', None)
        if feature_type is None:
            raise ValueError("Query must include 'type' parameter")
        
        # Get feature list
        features = self.find_by_type(feature_type)
        if not features:
            return []
        
        # Apply filters
        results = []
        for f in features:
            match = True
            
            for key, value in filters.items():
                # Handle comparison operators in key names (e.g., "prominence__gt")
                if '__' in key:
                    field_name, op = key.split('__')
                else:
                    field_name, op = key, 'eq'
                
                # Get the attribute value
                if hasattr(f, field_name):
                    attr_value = getattr(f, field_name)
                elif field_name in f.metadata:
                    attr_value = f.metadata[field_name]
                else:
                    match = False
                    break
                
                # Apply comparison
                if op == 'eq':
                    if attr_value != value:
                        match = False
                elif op == 'gt':
                    if not (attr_value > value):
                        match = False
                elif op == 'gte':
                    if not (attr_value >= value):
                        match = False
                elif op == 'lt':
                    if not (attr_value < value):
                        match = False
                elif op == 'lte':
                    if not (attr_value <= value):
                        match = False
                elif op == 'contains':
                    if value not in attr_value:
                        match = False
                else:
                    raise ValueError(f"Unknown operator: {op}")
                
                if not match:
                    break
            
            if match:
                results.append(f)
        
        return results
    
    # =========================================================================
    # LLM-FRIENDLY METHODS
    # =========================================================================
    
    def get_feature_by_id(self, feature_id: str) -> Optional[ClassifiedFeature]:
        """Retrieve a feature by its unique ID."""
        for feature_list in [self.peaks, self.ridges, self.valleys, self.saddles, self.flat_zones]:
            for f in feature_list:
                if f.feature_id == feature_id:
                    return f
        return None
    
    def get_context_window(self, max_features: int = 10) -> str:
        """
        Generate a compact text summary for LLM consumption.
        Returns a structured prompt-ready context string.
        """
        lines = []
        
        # Header
        lines.append("TERRAIN ANALYSIS SUMMARY")
        lines.append("=" * 50)
        
        # Narrative if available
        if self.terrain_narrative:
            lines.append(f"\n{self.terrain_narrative}\n")
        
        # Statistics
        if self.summary_stats:
            lines.append("STATISTICS")
            lines.append("-" * 30)
            elev = self.summary_stats.get('elevation', {})
            if elev:
                lines.append(f"Elevation: {elev.get('min', 0):.0f}m - {elev.get('max', 0):.0f}m (Δ{elev.get('range', 0):.0f}m)")
            slope = self.summary_stats.get('slope', {})
            if slope:
                lines.append(f"Average slope: {slope.get('mean', 0):.1f}°")
            lines.append(f"Features: {len(self.peaks)} peaks, {len(self.ridges)} ridges, {len(self.valleys)} valleys")
        
        # Key features (top N by prominence/size)
        lines.append("\nKEY FEATURES")
        lines.append("-" * 30)
        
        # Top peaks
        if self.peaks:
            sorted_peaks = sorted(self.peaks, key=lambda p: p.prominence, reverse=True)[:max_features//2]
            lines.append("Most prominent peaks:")
            for p in sorted_peaks:
                desc = self.feature_descriptions.get(p.feature_id, f"peak at {p.centroid}")
                lines.append(f"  • {desc} ({p.prominence:.0f}m prominence)")
        
        # Top ridges
        if self.ridges:
            sorted_ridges = sorted(self.ridges, key=lambda r: len(r.spine_points), reverse=True)[:max_features//2]
            lines.append("\nLongest ridges:")
            for r in sorted_ridges:
                desc = self.feature_descriptions.get(r.feature_id, f"ridge at {r.centroid}")
                lines.append(f"  • {desc}")
        
        # Regional descriptions
        if self.region_descriptions:
            lines.append("\nREGIONAL CHARACTER")
            lines.append("-" * 30)
            for region, desc in self.region_descriptions.items():
                lines.append(f"  {region.capitalize()}: {desc}")
        
        # Tactical notes
        if self.tactical_index:
            lines.append("\nTACTICAL NOTES")
            lines.append("-" * 30)
            obs = self.tactical_index.get('observation_posts', [])[:3]
            if obs:
                lines.append("Recommended observation points:")
                for o in obs:
                    lines.append(f"  • {o.get('description', 'Unknown')}")
            
            choke = self.tactical_index.get('chokepoints', [])[:3]
            if choke:
                lines.append("\nNatural chokepoints:")
                for c in choke:
                    lines.append(f"  • {c.get('description', 'Unknown')}")
        
        return "\n".join(lines)
    
    def describe_feature(self, feature_id: str) -> str:
        """Get human-readable description of a specific feature."""
        if feature_id in self.feature_descriptions:
            return self.feature_descriptions[feature_id]
        
        feature = self.get_feature_by_id(feature_id)
        if feature is None: 
            return f"Unknown feature: {feature_id}"
        
        if isinstance(feature, PeakFeature):
            return f"Peak at {feature.centroid} with {feature.prominence:.0f}m prominence, {feature.avg_slope:.0f}° slopes"
        elif isinstance(feature, RidgeFeature):
            return f"Ridge spanning {len(feature.spine_points)} points, adjacent to {len(feature.adjacent_watersheds)} watersheds"
        elif isinstance(feature, ValleyFeature):
            return f"Valley along {len(feature.spine_points)} points"
        elif isinstance(feature, SaddleFeature):
            return f"Saddle at {feature.centroid} connecting ridges and valleys"
        elif isinstance(feature, FlatZoneFeature):
            return f"Flat zone of {feature.area_pixels} pixels, max slope {feature.get_max_slope:.0f}°"
        
        return f"{type(feature).__name__} at {feature.centroid}"
    
    def _build_spatial_index(self):
        """Build spatial indexes for fast proximity queries."""
        if self._index_built:
            return
        
        try:
            from scipy.spatial import KDTree
            self._spatial_index = {}
            
            if self.peaks:
                peak_coords = [p.centroid for p in self.peaks]
                self._spatial_index['peaks'] = KDTree(peak_coords)
            
            # TODO: Add more feature types as we go...
            
            self._index_built = True
        except ImportError:
            pass  # scipy not available, skip indexing
    
    def find_nearby(self, feature_id: str, radius_m: float, feature_type: Optional[type] = None) -> List[ClassifiedFeature]:
        """
        Find features within radius of a given feature.
        
        Args:
            feature_id: Target feature ID
            radius_m: Search radius in meters
            feature_type: Optional filter by feature type
        """
        target = self.get_feature_by_id(feature_id)
        if target is None:
            return []
        
        self._build_spatial_index()
        
        if self._spatial_index is None:
            return []
        
        # Convert radius to pixels
        radius_px = radius_m / self.source_heightmap.config.horizontal_scale
        
        results = []
        target_px = target.centroid
        
        # Check each feature type
        type_map = {
            PeakFeature: self.peaks,
            RidgeFeature: self.ridges,
            ValleyFeature: self.valleys,
            SaddleFeature: self.saddles,
            FlatZoneFeature: self.flat_zones,
        }
        
        for ftype, flist in type_map.items():
            if feature_type is not None and ftype != feature_type:
                continue
            
            for f in flist:
                if f.feature_id == feature_id:
                    continue
                
                # Calculate Euclidean distance in pixels
                dx = f.centroid[0] - target_px[0]
                dy = f.centroid[1] - target_px[1]
                dist_px = (dx*dx + dy*dy) ** 0.5
                
                if dist_px <= radius_px:
                    results.append(f)
        
        return results
        
        
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
        #print(f'[cache] hashing: {combined_sum }')
        return combined_sum 
    
    def exists(self, cache_id: str, name: str, extension: str) -> bool:
        """Check if a cache file exists."""
        file_path = self.cache_dir / f"{cache_id}.{self.layer_name}.{name}.{extension}"
        file_exists = file_path.exists()
        #print(f'[cache] seek: {file_path}, exists={file_exists}')
        return file_exists
    
    def save_json(self, cache_id: str, name: str, data: dict) -> None:
        """Save metadata as JSON."""
        self._ensure_dir()
        path = self.cache_dir / f"{cache_id}.{self.layer_name}.{name}.json"
        #print(f'[cache] writing: {path}...')
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_json(self, cache_id: str, name: str) -> dict:
        """Load metadata JSON."""
        path = self.cache_dir / f"{cache_id}.{self.layer_name}.{name}.json"
        #print(f'[cache] reading: {path}...')
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
        
    def save_h5(self, cache_id: str, name: str, data: dict) -> None:
        """
        Save a structured dict of numpy arrays and string lists to HDF5.
        
        data format:
            { 'group/dataset_name': np.ndarray or List[str] }
        
        Example:
            { 'peaks/centroids': np.array(...), 'peaks/feature_ids': ['uuid1', ...] }
        """
        self._ensure_dir()
        path = self.cache_dir / f"{cache_id}.{self.layer_name}.{name}.h5"
        print(f'[cache] writing: {path}...')
        with h5py.File(path, 'w') as f:
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value, compression='gzip', compression_opts=4)
                elif isinstance(value, list):
                    # String lists — encode to fixed-length bytes for HDF5 compatibility
                    encoded = np.array([s.encode('utf-8') for s in value],
                                       dtype=h5py.special_dtype(vlen=str))
                    f.create_dataset(key, data=encoded)
                else:
                    raise TypeError(f"[cache] unsupported type for key '{key}': {type(value)}")

    def load_h5(self, cache_id: str, name: str) -> dict:
        """
        Load HDF5 file back into a flat dict of arrays and string lists.
        Reconstructs the same structure passed to save_h5.
        """
        path = self.cache_dir / f"{cache_id}.{self.layer_name}.{name}.h5"
        #print(f'[cache] reading: {path}...')
        result = {}
        with h5py.File(path, 'r') as f:
            def _visit(key, obj):
                if isinstance(obj, h5py.Dataset):
                    raw = obj[()]
                    # Decode byte strings back to Python str
                    if raw.dtype.kind in ('O', 'S'):
                        raw = [v.decode('utf-8') if isinstance(v, bytes) else v
                               for v in raw]
                    result[key] = raw
            f.visititems(_visit)
        return result