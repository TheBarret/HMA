import numpy as np

from enum import auto, Enum
from typing import TypeAlias
from typing import Dict, Any
from typing import Optional
from typing import List, Set
from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar, Protocol
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

# Typed bundle passed between layers — explicit over opaque dicts
LayerBundle: TypeAlias = Dict[str, Any]  # Keys defined per-layer in output_schema

# Feature classification

class CurvatureType(Enum):
    CONVEX = auto()
    CONCAVE = auto()
    FLAT = auto()
    SADDLE = auto()
    
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
        if not isinstance(self.data, np.ndarray):
            return False
        if self.data.ndim != 2:
            return False
        if self.data.dtype != np.uint8:
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
        return self.data.shape[::-1]  # Return (width, height)
    
    def elevation_at(self, pixel: PixelCoord) -> Optional[float]:
        """Safe elevation lookup with bounds checking."""
        x, y = pixel
        if 0 <= x < self.shape[0] and 0 <= y < self.shape[1]:
            return self.data[y, x]  # Note: numpy uses [row, col] = [y, x]
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
#  Pipeline Configuration
#

@dataclass
class PipelineConfig:
    """Global configuration for the analysis pipeline."""
    
    # Layer 0: Calibration
    noise_reduction_sigma: float = 1.0  # Gaussian blur radius
    
    # Layer 1: Slope thresholds
    cliff_threshold_degrees: float = 45.0
    gentle_slope_threshold_degrees: float = 15.0
    
    # Layer 2: Curvature sensitivity
    curvature_epsilon: float = 0.01  # Threshold for "near-zero" curvature
    
    # Layer 5: Domain rules
    vehicle_climb_angle: float = 30.0  # Max slope for traversability
    visibility_sample_radius: int = 8  # Rays per degree for viewshed
    
    # Scale-space analysis
    analysis_scales: Dict[str, int] = field(default_factory=lambda: {
        "micro": 3,    # 3px radius for fine detail
        "meso": 15,    # 15px radius for tactical features  
        "macro": 50,   # 50px radius for operational features
    })
    
    
    # Curvature tuning (for game maps)
    curvature_epsilon_h_factor: float = 0.5      # std multiplier for H
    curvature_epsilon_k_factor: float = 0.5      # std multiplier for K  
    curvature_epsilon_h_min: float = 1e-5        # Minimum H threshold (1/m)
    curvature_epsilon_k_min: float = 1e-5        # Minimum K threshold (1/m²)
    
    def is_traversable(self, slope_degrees: float) -> Traversability:
        """Classify slope by vehicle constraints."""
        if slope_degrees > self.cliff_threshold_degrees:
            return Traversability.BLOCKED
        elif slope_degrees > self.vehicle_climb_angle:
            return Traversability.DIFFICULT
        return Traversability.FREE
        
        
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
        

#
#  Layer Stubs (Contracts Only)
#


class Layer0_Calibration(PipelineLayer[Heightmap]):
    """Normalize raw image → calibrated mathematical surface."""
    
    def execute(self, input_data: dict) -> Heightmap:
        # input_data: {"raw_array": np.ndarray, "config": NormalizationConfig}
        # Output: Heightmap with noise-reduced, normalized data
        pass
    
    @property
    def output_schema(self) -> dict:
        return {"type": "Heightmap", "fields": ["data", "config", "origin"]}

class Layer1_LocalGeometry(PipelineLayer[Dict[str, ScalarField]]):
    """Compute first derivatives: slope magnitude and aspect direction."""
    
    def execute(self, input_data: Heightmap) -> Dict[str, ScalarField]:
        # Output: {"slope": 2D array of degrees, "aspect": 2D array of radians}
        pass
    
    @property
    def output_schema(self) -> dict:
        return {
            "slope": {"type": "ScalarField", "range": [0, 90], "unit": "degrees"},
            "aspect": {"type": "ScalarField", "range": [0, 6.28318], "unit": "radians"}  # 0–2π full compass
        }

class Layer2_RegionalGeometry(PipelineLayer[Dict[str, ScalarField]]):
    """
    Compute second derivatives: curvature classification.
    
    Takes the calibrated Heightmap directly — curvature is ∂²z/∂x² and ∂²z/∂y²,
    computed from the clean elevation surface, NOT derived from the slope/aspect maps.
    Deriving from Layer 1 output would compound numerical error across two finite-difference steps.
    """
    
    def execute(self, input_data: Heightmap) -> Dict[str, ScalarField]:
        # Input: calibrated Heightmap from Layer 0 (same source as Layer 1)
        # Output: {"curvature": 2D array, "curvature_type": 2D array of CurvatureType}
        pass
    
    @property
    def output_schema(self) -> dict:
        return {
            "curvature": {"type": "ScalarField", "unit": "1/meters"},
            "gaussian_curvature": {"type": "ScalarField", "unit": "1/meters²"},  # ADD THIS
            "curvature_type": {"type": "CategoricalField", "values": ["CONVEX", "CONCAVE", "FLAT", "SADDLE"]}
        }


class Layer3_TopologicalFeatures(PipelineLayer[List["TerrainFeature"]]):
    """
    Resolve continuous fields into discrete terrain structures.
    
    Requires the full bundle from all prior layers:
      - heightmap:      raw elevation for prominence calculation
      - slope:          needed to distinguish ridge (convex + steep) from plateau (convex + flat)
      - aspect:         needed for saddle orientation and flow-seed direction
      - curvature:      primary signal for ridge/valley/saddle classification
      - curvature_type: pre-classified categorical field for threshold-free feature seeding

    Output changes type: this is the first layer that produces *discrete objects*
    (points and polylines) rather than continuous scalar fields.
    """
    
    def execute(self, input_data: LayerBundle) -> List["TerrainFeature"]:
        # input_data keys: "heightmap", "slope", "aspect", "curvature", "curvature_type"
        # Output: flat list of TerrainFeature subclasses (PeakFeature, RidgeFeature, etc.)
        pass
    
    @property
    def output_schema(self) -> dict:
        return {
            "type": "List[TerrainFeature]",
            "feature_types": ["PeakFeature", "ValleyFeature", "RidgeFeature", "SaddleFeature"],
            "geometry": "points and polylines in PixelCoord space"
        }

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
   
    def is_traversable(self, config: PipelineConfig) -> Traversability:  # Add this
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
        # Valleys are often wet/difficult, so add penalty
        base = config.is_traversable(self.avg_slope)
        if base == Traversability.FREE:
            return Traversability.DIFFICULT  # Valleys default to difficult
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

class Layer4_Relational(PipelineLayer[Dict[str, Any]]):
    """
    Build relational graphs from discrete features.

    Requires the full bundle from all prior layers — features alone are insufficient:
      - features:       discrete objects from Layer 3 (nodes in every graph)
      - heightmap:      elevation surface for line-of-sight ray casting (viewshed)
      - slope/aspect:   flow direction is re-derived from aspect, not from features
      - curvature:      watershed boundary seeds follow ridge lines (positive curvature)

    Output is graph-structured, not scalar fields: visibility, flow, and connectivity
    are all relations between features, not per-pixel values.
    """

    def execute(self, input_data: LayerBundle) -> Dict[str, Any]:
        # input_data keys: "features", "heightmap", "slope", "aspect", "curvature"
        # Output:
        #   "visibility_graph":    Dict[str, Set[str]]   — feature_id → set of visible feature_ids
        #   "flow_network":        Dict[str, List[str]]  — feature_id → ordered downstream feature_ids
        #   "connectivity_graph":  Dict[str, Set[str]]   — feature_id → adjacent traversable feature_ids
        #   "watersheds":          Dict[str, Set[str]]   — basin_id → set of member feature_ids
        #   "flow_accumulation":   ScalarField           — per-pixel accumulated upstream area
        pass

    @property
    def output_schema(self) -> dict:
        return {
            "visibility_graph":   {"type": "Dict[str, Set[str]]",  "description": "feature_id → visible feature_ids"},
            "flow_network":       {"type": "Dict[str, List[str]]", "description": "feature_id → downstream feature_ids"},
            "connectivity_graph": {"type": "Dict[str, Set[str]]",  "description": "feature_id → traversable neighbours"},
            "watersheds":         {"type": "Dict[str, Set[str]]",  "description": "basin_id → member feature_ids"},
            "flow_accumulation":  {"type": "ScalarField",          "description": "per-pixel upstream area in pixels²"},
        }


class Layer5_Semantics(PipelineLayer["AnalyzedTerrain"]):
    """
    Assemble all prior outputs into the final AnalyzedTerrain object.

    This layer applies domain thresholds and classification rules to produce
    human-meaningful terrain categories. It is the only layer allowed to
    construct AnalyzedTerrain — all previous layers remain domain-agnostic.

    Requires the full bundle from all prior layers:
      - features:            discrete objects for population into typed lists
      - visibility_graph,
        flow_network,
        connectivity_graph,
        watersheds,
        flow_accumulation:   relational outputs from Layer 4
      - heightmap:           retained as source reference in AnalyzedTerrain
    """

    def execute(self, input_data: LayerBundle) -> "AnalyzedTerrain":
        # input_data keys: all Layer 3 + Layer 4 outputs, plus "heightmap"
        # Output: fully populated AnalyzedTerrain instance
        pass

    @property
    def output_schema(self) -> dict:
        return {
            "type": "AnalyzedTerrain",
            "fields": [
                "source_heightmap", "peaks", "valleys", "ridges", "saddles", "flat_zones",
                "visibility_graph", "flow_network", "connectivity_graph",
                "watersheds", "flow_accumulation",
            ]
        }

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
        
@dataclass(frozen=True)
class ScaledTransform:
    """Serializable pixel → world coordinate transform."""
    scale: float
    offset_x: float = 0.0
    offset_y: float = 0.0

    def __call__(self, pixel: PixelCoord) -> WorldCoord:
        return (pixel[0] * self.scale + self.offset_x,
                pixel[1] * self.scale + self.offset_y)
                

@dataclass(frozen=True)
class AffineTransform:
    """
    Serializable pixel → world coordinate transform (full affine, 6-parameter).
    Replaces the affine lambda closure which cannot be pickled or saved to disk.
    Parameters match the [a b tx; c d ty] convention.
    """
    a: float; b: float; tx: float
    c: float; d: float; ty: float

    def __call__(self, pixel) -> tuple:
        x, y = pixel
        return (self.a * x + self.b * y + self.tx,
                self.c * x + self.d * y + self.ty)
                
"""

    Config Wrappers
   
    TODO:
    - Add more game templates (ideally themed combat simulators/games)
"""

class GameType(Enum):
    """Supported game types for automatic scaling"""
    ARMA_2 = "ARMA_2"                   # Typical: 1-5m resolution
    ARMA_3 = "ARMA_3"                   # Typical: 1-10m resolution  
    WORLD_OF_TANKS = "WORLD OF TANKS"   # Vehicle-focused, 1-2m resolution
    WAR_THUNDER = "WARTHUNDER"          # Mixed, 1-5m resolution
    CUSTOM = "CUSTOM"

@dataclass
class GameScaling:
    """Game-specific scaling presets"""
    game_type: GameType
    horizontal_scale_m_per_px: float
    vertical_scale_m_per_unit: float
    vehicle_climb_angle_deg: float
    infantry_climb_angle_deg: float
    
    @classmethod
    def for_game(cls, game: GameType) -> 'GameScaling':
        presets = {
            GameType.ARMA_3: cls(
                game_type=game,
                horizontal_scale_m_per_px=2.0,    # ArmA maps: 2-5m typical
                vertical_scale_m_per_unit=0.2,    # 0.2m per grayscale unit
                vehicle_climb_angle_deg=30.0,     # ArmA vehicles
                infantry_climb_angle_deg=45.0     # Infantry can climb steeper
            ),
            GameType.WORLD_OF_TANKS: cls(
                game_type=game,
                horizontal_scale_m_per_px=1.0,    # WoT: detailed 1m resolution
                vertical_scale_m_per_unit=0.1,    # 0.1m precision
                vehicle_climb_angle_deg=25.0,     # Tanks have limits
                infantry_climb_angle_deg=45.0
            ),
            GameType.WAR_THUNDER: cls(
                game_type=game,
                horizontal_scale_m_per_px=1.5,    # Mixed resolution
                vertical_scale_m_per_unit=0.15,
                vehicle_climb_angle_deg=30.0,
                infantry_climb_angle_deg=45.0
            ),
            GameType.ARMA_2: cls(
                game_type=game,
                horizontal_scale_m_per_px=5.0,    # Older maps larger scale
                vertical_scale_m_per_unit=0.25,
                vehicle_climb_angle_deg=25.0,
                infantry_climb_angle_deg=45.0
            )
        }
        return presets.get(game, presets[GameType.ARMA_3])
        
@dataclass
class PipelineConfig:
    """Global configuration for the analysis pipeline."""
    
    # Layer 0: Calibration 
    horizontal_scale: float = 2.0
    vertical_scale: float = 0.2
    sea_level_offset: float = 10.0
    noise_reduction_sigma: float = 1.2
    
    # Layer 1: Gradient 
    gradient_method: str = "central_difference"  # or "sobel"
    flat_threshold_deg: float = 1.0
    
    # Layer 2: Curvature
    curvature_epsilon: float = 0.0001            # static fallback
    
    adaptive_epsilon = True
    curvature_epsilon_h_factor = 0.6  # (0.6 = rollign hills)
    curvature_epsilon_k_factor = 0.6
    curvature_epsilon_h_min = 1.38e-03
    curvature_epsilon_k_min = 3.70e-05
    
    
    # Layer 3: Topology 
    peak_annular_inner_m: float = 5.0
    peak_annular_outer_m: float = 12.0
    peak_confidence_threshold: float = 0.3
    min_peak_size_px: int = 2            # keep 1 — local_maxima returns 1px plateaus
    peak_min_prominence_m: float = 2.0   # real quality gate for peaks (meters)
    peak_nms_radius_px: int = 15         # suppress duplicate peaks within this radius
    min_ridge_length_px: int = 6
    min_valley_length_px: int = 6
    min_saddle_size_px: int = 50         # unused now (topographic approach)
    min_flat_zone_size_px: int = 50
    saddle_k_min_threshold: float = 5e-5
    saddle_confidence_threshold: float = 0.1
    flat_zone_slope_threshold_deg: float = 5.0
    border_margin_px: int = 10
    prominence_search_radius_m: float = 100.0
    
    # Layer 4: Relational 
    visibility_max_range_m: float = 1200.0
    connection_radius_m: float = 200.0
    viewshed_sample_step_px: int = 5
    flow_step_px: int = 10
    flow_neighbor_distance_px: int = 50
    
    # Layer 5: Tactical 
    game_type: GameType = GameType.ARMA_3
    defensive_min_prominence_m: float = 8.0
    defensive_min_elevation_m: float = 5.0
    defensive_max_slope_deg: float = 25.0
    defensive_min_visibility: int = 5
    observation_min_prominence_m: float = 10.0
    observation_min_visibility: int = 10
    assembly_min_area_m2: float = 2000.0
    assembly_max_slope_deg: float = 5.0
    chokepoint_min_connectivity: int = 2
    cover_min_width_m: float = 5.0
    
    # Existing fields
    cliff_threshold_degrees: float = 45.0
    gentle_slope_threshold_degrees: float = 15.0
    vehicle_climb_angle: float = 30.0
    visibility_sample_radius: int = 8
    analysis_scales: Dict[str, int] = field(default_factory=lambda: {
        "micro": 3, "meso": 15, "macro": 50
    })
    
    # Runtime
    verbose: bool = True
    save_intermediates: bool = False