import numpy as np

from enum import auto, Enum
from typing import TypeAlias
from typing import Dict, Any
from typing import Optional
from typing import List, Set
from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar
from dataclasses import dataclass, field, KW_ONLY
from uuid import uuid4

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



# Template Profiles
class Template(Enum):
    ARMA_2 = "ARMA_2"                   # Typical: 1-5m resolution
    ARMA_3 = "ARMA_3"                   # Typical: 1-10m resolution  
    WORLD_OF_TANKS = "WORLD OF TANKS"   # Vehicle-focused, 1-2m resolution
    WAR_THUNDER = "WARTHUNDER"          # Mixed, 1-5m resolution
    CUSTOM = "CUSTOM"

@dataclass
class TScaling:
    baseline: Template
    horizontal_scale_m_per_px: float
    vertical_scale_m_per_unit: float
    vehicle_climb_angle_deg: float
    infantry_climb_angle_deg: float
    
    @classmethod
    def for_template(cls, _t: Template) -> 'TScaling':
        presets = {
            Template.ARMA_3: cls(
                baseline=_t,
                horizontal_scale_m_per_px=2.0,    # ArmA maps: 2-5m typical
                vertical_scale_m_per_unit=0.2,    # 0.2m per grayscale unit
                vehicle_climb_angle_deg=30.0,     # ArmA vehicles
                infantry_climb_angle_deg=45.0     # Infantry can climb steeper
            ),
            Template.WORLD_OF_TANKS: cls(
                baseline=_t,
                horizontal_scale_m_per_px=1.0,    # WoT: detailed 1m resolution
                vertical_scale_m_per_unit=0.1,    # 0.1m precision
                vehicle_climb_angle_deg=25.0,     # Tanks have limits
                infantry_climb_angle_deg=45.0
            ),
            Template.WAR_THUNDER: cls(
                baseline=_t,
                horizontal_scale_m_per_px=1.5,    # Mixed resolution
                vertical_scale_m_per_unit=0.15,
                vehicle_climb_angle_deg=30.0,
                infantry_climb_angle_deg=45.0
            ),
            Template.ARMA_2: cls(
                baseline=_t,
                horizontal_scale_m_per_px=5.0,    # Older maps larger scale
                vertical_scale_m_per_unit=0.25,
                vehicle_climb_angle_deg=25.0,
                infantry_climb_angle_deg=45.0
            ),
            Template.CUSTOM: cls(
                baseline=_t,
                horizontal_scale_m_per_px=2.0,    # Custom
                vertical_scale_m_per_unit=0.25,
                vehicle_climb_angle_deg=25.0,
                infantry_climb_angle_deg=45.0
            )
        }
        return presets.get(_t, presets[Template.ARMA_3])

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
    min_peak_size_px: int = 1                   # [1-10] pixels, minimum peak footprint
    peak_min_prominence_m: float = 5.0          # [1-50] meters, minimum height above saddle
    peak_nms_radius_px: int = 15                # [5-30] pixels, non-maximum suppression radius
    peak_shoulder_convex_ratio: float = 0.05    # [0.01-0.3] convex pixels in annular ring
    peak_annular_inner_m: float = 5.0           # [2-15] meters, inner shoulder radius
    peak_annular_outer_m: float = 12.0          # [5-25] meters, outer shoulder radius
    peak_smooth_sigma: float = 1.5              # ndimage.gaussian_filter(z, sigma)
    
    # --- Ridges & Valleys (linear features) ---
    min_ridge_length_px: int = 10                # [3-20] pixels, minimum ridge length
    min_valley_length_px: int = 10               # [3-20] pixels, minimum valley length
    
    # --- Flat Zones (traversable areas) ---
    min_flat_zone_size_px: int = 450            # [50-500] pixels, minimum flat area
    flat_zone_slope_threshold_deg: float = 3.0  # [1-10] degrees, max slope for "flat"
    
    # --- Saddles (passes between peaks) ---
    saddle_k_min_threshold: float = 0.00020     # [1e-6-1e-2] minimum |K| for saddle (1/m²)
    saddle_confidence_threshold: float = 1.0    # [0.0-1.0] normalized confidence (1.0 = all)

    # --- Sea Level ---
    exclude_below_reference: bool = True        # [True/False] exclude sea domain features
    elevation_reference_m: float = 8.0         # meters
        
    # --- General Topology ---
    border_margin_px: int = 10                  # [5-30] pixels, ignore edges
    prominence_search_radius_m: float = 100.0   # [50-500] meters, search radius for prominence
    
    # =========================================================================
    # LAYER 4: RELATIONAL - Connectivity & Flow
    # =========================================================================
    
    # --- Visibility ---
    visibility_max_range_m: float = 1200.0      # [200-3000] meters, max line-of-sight distance
    viewshed_sample_step_px: int = 8            # [1-50] pixels, step size for ray casting (performance)
    visibility_sample_radius: int = 5           # [3-15] pixels, sampling radius for viewshed
    
    # --- Flow Network ---
    flow_step_px: int = 10                      # [5-20] pixels, step size for flow accumulation
    flow_neighbor_distance_px: int = 50         # [20-100] pixels, max distance to downstream feature
    
    # --- Connectivity ---
    connection_radius_m: float = 75.0          # [50-500] meters, feature connection radius
    vehicle_climb_angle: float = 25.0           # [20-45] degrees, max slope for vehicles
    cliff_threshold_degrees: float = 45.0       # [30-60] degrees, impassable terrain
    
    # --- Watersheds ---
    watershed_min_area_m2: float = 150.0       # [500-10000] m², minimum watershed area
    watershed_sample_step_px: int = 10          # [5-20] pixels, step size for watershed delineation
    
    # =========================================================================
    # LAYER 5: SEMANTICS
    # =========================================================================
    
    baseline: Template = Template.ARMA_3       # [ARMA_2/ARMA_3/WAR_THUNDER/WORLD_OF_TANKS/CUSTOM]
    max_feature_coverage: float = 0.5           # Max 50% of map for any single feature
    
    # --- Defensive Positions ---
    
    # classification thresholds
    threshold_major_peak: float = 15.0          # major peaks [m]
    threshold_minor_peak: float = 5.0           # minor peaks [m]
    saddle_elevation_high: float = 30.0         # saddle elevation high [m]
    saddle_elevation_low: float = 10.0          # saddle elevation low [m]
    valley_avg_slope: float = 15.0              # valley avg slope [m]
        
    defensive_min_prominence_m: float = 8.0     # [5-30] meters, minimum height advantage
    defensive_min_elevation_m: float = 5.0      # [2-20] meters, minimum absolute height
    defensive_max_slope_deg: float = 25.0       # [15-35] degrees, max slope for defense
    defensive_min_visibility: int = 5           # [3-15] number of visible features
    
    # --- Observation Points ---
    observation_min_prominence_m: float = 10.0  # [5-50] meters, minimum prominence
    observation_min_visibility: int = 10        # [5-30] number of visible features
    
    # --- Assembly Areas ---
    assembly_min_area_m2: float = 2000.0        # [500-10000] square meters, staging area
    assembly_max_slope_deg: float = 5.0         # [2-10] degrees, max slope for assembly
    
    # --- Chokepoints ---
    chokepoint_min_connectivity: int = 2        # [2-5] minimum connections to be chokepoint
    cover_min_width_m: float = 5.0              # [2-15] meters, minimum cover width
    analysis_scales: Dict[str, int] = field(default_factory=lambda: {
        "micro": 3,     # [1-5] pixels, noise/small rocks
        "meso": 15,     # [10-30] pixels, gullies/ridges
        "macro": 50     # [30-100] pixels, mountains/valleys
    })
    
    # =========================================================================
    # RUNTIME
    # =========================================================================
    
    verbose: bool = True                        # [True/False] enable debug logging
    
    # FUTURE UPDATE SEE PROPOSAL: CACHE AWARE PIPELINE (CACHE.PY
    save_intermediates: bool = False            # [True/False] save intermediate files 
    
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
    metadata: Optional[dict] = None  # Optional: georeferencing, units, etc.
    
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
              traversable_by=VehicleProfile(tank=True)
          )
        """
        pass
        