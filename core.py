import h5py
import hashlib
import json
import numpy as np

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
    adaptive_percentile: float = 58.0
    
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
    peak_min_shoulder_samples: int = 10         # [5-50] minimum pixels in shoulder region
    min_peak_size_px: int = 1                   # [1-10] pixels, minimum peak footprint
    peak_min_prominence_m: float = 10.0          # [1-50] meters, minimum height above saddle
    peak_nms_radius_px: int = 30               # [1-100] pixels, non-maximum suppression radius [tweak this when saddles are too surpressed]
    peak_shoulder_convex_ratio: float = 0.08    # [0.01-0.3] n% convex pixels in annular ring
    peak_annular_inner_m: float = 2.0           # [2-15] meters, inner shoulder radius
    peak_annular_outer_m: float = 40.0          # [1-50] meters, outer shoulder radius
    peak_smooth_sigma: float = 2.5              # ndimage.gaussian_filter(z, sigma)
    
    # --- Ridges & Valleys (linear features) ---
    min_ridge_length_px: int = 15                # [2-100] pixels, minimum ridge length
    min_valley_length_px: int = 10               # [2-20] pixels, minimum valley length
    
    # --- Ridge Smoothing ---
    smooth_spine_window: int = 5                # [1-10] Ridge spine smoothing window
    
    # --- Flat Zones (traversable areas) ---
    min_flat_zone_size_px: int = 80            # [50-500] pixels, minimum flat area
    flat_zone_slope_threshold_deg: float = 5.0  # [1-10] degrees, max slope for "flat"
    
    # --- Saddles (passes between peaks) ---
    saddle_k_min_threshold: float = 0.00020     # [2.00e-4/m²] minimum |K| for saddle (1/m²)
    saddle_confidence_threshold: float = 0.50    # [0.0-1.0] normalized confidence (1.0 = all)

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
    stream_accumulation_threshold_px: int = 120     # [10-500] pixels
    feature_snap_distance_px: int = 20             # [5-20] pixels
    snap_method: str = "nearest"                   # ["nearest", "flow_directed", "none"]
    include_edge_basins: bool = True               # include basins draining off-map

    # --- Connectivity ---
    
    # Connection Radius/Pixels [Performance tweaker]
    # Radius	| Pixels per source	    | Total expansions | Performance
    # 50m       | ~1,963	            | 1.47M	           | Default
    # 40m       | ~1,256	            | 942K	           | 36% reduction
    # 30m       | ~706	                | 530K	           | 64% reduction
    # 25m       | ~490	                | 368K	           | 75% reduction
    connection_radius_m: float = 40.0              # [1-1000] meters
    
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
    verbose_interval: int = 8                   # [2-64] i % n == 0 { iterative_logs }
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
                # assumes 'Heigtmap' object is available
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
        
# ----------------------------------------------------
#   Auto cache mechanic
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