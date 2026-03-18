import numpy as np

from enum import StrEnum, auto
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
class CurvatureType(StrEnum):
    CONVEX = auto()    # Positive: ridges, peaks
    CONCAVE = auto()   # Negative: valleys, pits  
    FLAT = auto()      # Near-zero: plains, uniform slopes
    SADDLE = auto()    # Mixed: mountain passes

class Traversability(StrEnum):
    FREE = auto()      # Slope < vehicle limit
    DIFFICULT = auto() # Slope near limit
    BLOCKED = auto()   # Slope > limit or cliff
    
    
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
#  Feature Abstractions
# 

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
    prominence: float                        # Height above surrounding saddle in meters
    visibility_radius: Optional[float] = None  # Meters
    
    @property
    def curvature_type(self) -> CurvatureType:
        return CurvatureType.CONVEX
    
    def is_traversable(self, config: PipelineConfig) -> Traversability:
        # Peaks are usually accessible only if slope permits
        return config.is_traversable(self.avg_slope)
    
    def contains_point(self, pixel: PixelCoord) -> bool:
        # Implement boundary check (e.g., convex hull test)
        pass

@dataclass  
class RidgeFeature(ClassifiedFeature):
    """Linear convex feature connecting peaks."""
    _: KW_ONLY
    spine_points: List[PixelCoord]    # Polyline defining ridge crest
    connected_peaks: Set[str]         # Feature IDs of connected peaks
    
    @property
    def curvature_type(self) -> CurvatureType:
        return CurvatureType.CONVEX
    

# ADDITONAL: ... ValleyFeature, SaddleFeature, etc. ...


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