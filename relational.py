"""
Relational Analysis

Builds graphs connecting terrain features:
- Visibility graph: which features can see each other
- Flow network: water flow between features
- Connectivity graph: traversable paths between features
- Watersheds: drainage basins
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import KDTree
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import warnings
import heapq

from core import (
    Heightmap, ScalarField, PipelineConfig, PipelineLayer,
    TerrainFeature, ClassifiedFeature,
    PeakFeature, RidgeFeature, ValleyFeature, SaddleFeature, FlatZoneFeature,
    PixelCoord, Traversability, LayerBundle
)


class Layer4_Relational(PipelineLayer[Dict[str, any]]):
    """
    Build relational graphs from discrete features.
    
    Outputs:
    - visibility_graph: feature_id → set of visible feature_ids
    - flow_network: feature_id → ordered downstream feature_ids
    - connectivity_graph: feature_id → adjacent traversable feature_ids
    - watersheds: basin_id → set of member feature_ids
    - flow_accumulation: per-pixel accumulated upstream area
    """
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self._viewshed_sample_step = 5   # Sample every 5th pixel for performance
        self._connection_radius_m = 200   # Max distance for connectivity (meters)
        self._visibility_max_range_m = 1500  # Hard cap on visibility check distance
        # Without a range cap, O(F²) Bresenham walks across the full map stall indefinitely.
        # 1500m covers most tactical engagement ranges and keeps pair count manageable.
        
    def execute(self, input_data: LayerBundle) -> Dict[str, any]:
        """
        Build relational graphs from features.
        
        Args:
            input_data: LayerBundle containing:
                - features: List[TerrainFeature] from Layer 3
                - heightmap: Heightmap from Layer 0
                - slope: Slope field from Layer 1
                - aspect: Aspect field from Layer 1
                - curvature: Curvature field from Layer 2
                
        Returns:
            Dictionary with graphs and watershed data
        """
        features = input_data['features']
        heightmap = input_data['heightmap']
        slope = input_data.get('slope')
        aspect = input_data.get('aspect')
        curvature = input_data.get('curvature')
        
        # Build feature index for quick lookup
        self._feature_index = self._build_feature_index(features)
        
        # 1. Visibility Graph (line-of-sight between features)
        print("\n=== Building Visibility Graph ===")
        visibility_graph = self._build_visibility_graph(features, heightmap)
        
        # 2. Flow Network (water flow direction between features)
        print("\n=== Building Flow Network ===")
        flow_network, flow_accumulation = self._build_flow_network(
            features, heightmap, aspect
        )
        
        # 3. Connectivity Graph (traversable paths)
        print("\n=== Building Connectivity Graph ===")
        connectivity_graph = self._build_connectivity_graph(
            features, heightmap, slope
        )
        
        # 4. Watersheds (drainage basins)
        print("\n=== Delineating Watersheds ===")
        watersheds = self._delineate_watersheds(
            features, heightmap, aspect, curvature
        )
        
        # Log statistics
        self._log_graph_statistics(
            visibility_graph, flow_network, connectivity_graph, watersheds
        )
        
        return {
            "visibility_graph": visibility_graph,
            "flow_network": flow_network,
            "connectivity_graph": connectivity_graph,
            "watersheds": watersheds,
            "flow_accumulation": flow_accumulation
        }
    
    def _build_feature_index(self, features: List[TerrainFeature]) -> Dict[str, TerrainFeature]:
        """Build dictionary mapping feature_id to feature."""
        return {f.feature_id: f for f in features}
    
    def _build_visibility_graph(self, features: List[TerrainFeature], 
                                heightmap: Heightmap) -> Dict[str, Set[str]]:
        """
        Build visibility graph using line-of-sight between feature centroids.
        
        Two features are visible if the line between them never goes below
        the terrain elevation at any point.
        """
        visibility = {f.feature_id: set() for f in features}
        
        # Extract centroids
        centroids = []
        feature_ids = []
        for f in features:
            x, y = f.centroid
            centroids.append((x, y))
            feature_ids.append(f.feature_id)
        
        # For performance, limit to peaks and saddles for initial implementation
        # (Full visibility for all features can be expensive)
        important_features = [
            f for f in features 
            if isinstance(f, (PeakFeature, SaddleFeature))
        ]
        
        important_ids = {f.feature_id for f in important_features}
        
        # Pre-compute max range in pixels for fast distance rejection
        max_range_px = self._visibility_max_range_m / heightmap.config.horizontal_scale

        # Check visibility between each pair of important features
        # Distance pre-filter eliminates the vast majority of pairs before
        # the expensive pixel-by-pixel Bresenham walk even begins.
        n = len(important_features)
        for i in range(n):
            for j in range(i + 1, n):
                f1 = important_features[i]
                f2 = important_features[j]

                # Fast Euclidean distance reject — skip pairs beyond max range
                dx = f1.centroid[0] - f2.centroid[0]
                dy = f1.centroid[1] - f2.centroid[1]
                if (dx*dx + dy*dy) > max_range_px * max_range_px:
                    continue

                if self._check_visibility(f1.centroid, f2.centroid, heightmap):
                    visibility[f1.feature_id].add(f2.feature_id)
                    visibility[f2.feature_id].add(f1.feature_id)
        
        return visibility
    
    def _check_visibility(self, p1: PixelCoord, p2: PixelCoord, 
                          heightmap: Heightmap) -> bool:
        """
        Check if two points are mutually visible using Bresenham line algorithm.
        """
        x1, y1 = p1
        x2, y2 = p2
        
        # Get elevation at endpoints
        z1 = heightmap.data[y1, x1]
        z2 = heightmap.data[y2, x2]
        
        # Calculate line parameters
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        x, y = x1, y1
        
        while (x, y) != (x2, y2):
            # Calculate interpolated elevation at this point along line
            t = np.sqrt((x - x1)**2 + (y - y1)**2) / np.sqrt(dx**2 + dy**2)
            interpolated_z = z1 + t * (z2 - z1)
            
            # Get actual terrain elevation
            terrain_z = heightmap.data[y, x]
            
            # If terrain blocks view, return False
            if terrain_z > interpolated_z:
                return False
            
            # Bresenham algorithm
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return True
    
    def _build_flow_network(self, features: List[TerrainFeature],
                            heightmap: Heightmap,
                            aspect: Optional[np.ndarray]) -> Tuple[Dict[str, List[str]], ScalarField]:
        """
        Build flow network based on aspect (flow direction).
        
        Returns:
            flow_network: feature_id → list of downstream feature_ids (ordered)
            flow_accumulation: per-pixel accumulated upstream area
        """
        flow_network = {f.feature_id: [] for f in features}
        
        # If no aspect, return empty network
        if aspect is None:
            warnings.warn("Aspect not available, flow network will be empty")
            return flow_network, None
        
        # Build feature KD-tree for nearest neighbor search
        feature_coords = [(f.centroid[0], f.centroid[1]) for f in features]
        feature_ids = [f.feature_id for f in features]
        kdtree = KDTree(feature_coords)
        
        # For each pixel, determine flow direction and accumulate
        flow_accumulation = np.zeros(heightmap.shape, dtype=np.float32)
        
        # Simplified: accumulate based on aspect direction
        # (Full flow accumulation would require D8 or D-infinity algorithm)
        rows, cols = heightmap.shape[1], heightmap.shape[0]  # Note: shape is (height, width)
        
        # Convert aspect to flow direction vectors
        # aspect: 0 = North, π/2 = East, π = South, 3π/2 = West
        dx = -np.sin(aspect)  # East-West component
        dy = -np.cos(aspect)  # North-South component
        
        # Find downstream connections between features
        for f in features:
            x, y = f.centroid
            
            # Check aspect at centroid to find flow direction
            if 0 <= x < cols and 0 <= y < rows:
                a = aspect[y, x]
                # Flow direction vector
                flow_x = x - np.sin(a) * 10  # 10px step
                flow_y = y - np.cos(a) * 10
                
                # Find nearest feature in flow direction
                distances, indices = kdtree.query([(flow_x, flow_y)])
                
                if distances[0] < 50:  # Within 50 pixels
                    downstream_id = feature_ids[indices[0]]
                    if downstream_id != f.feature_id:
                        flow_network[f.feature_id].append(downstream_id)
        
        return flow_network, flow_accumulation
    
    def _build_connectivity_graph(self, features: List[TerrainFeature],
                                   heightmap: Heightmap,
                                   slope: Optional[np.ndarray]) -> Dict[str, Set[str]]:
        """
        Build connectivity graph based on traversability between features.
        
        Features are connected if there exists a traversable path between them
        respecting the vehicle's slope limits.
        """
        connectivity = {f.feature_id: set() for f in features}
        
        # If no slope, use simple distance-based connectivity
        if slope is None:
            warnings.warn("Slope not available, using distance-based connectivity")
            return self._distance_based_connectivity(features)
        
        # Build feature KD-tree
        feature_coords = [(f.centroid[0], f.centroid[1]) for f in features]
        feature_ids = [f.feature_id for f in features]
        kdtree = KDTree(feature_coords)
        
        # Convert connection radius from meters to pixels
        radius_px = int(self._connection_radius_m / heightmap.config.horizontal_scale)
        
        for i, f in enumerate(features):
            x, y = f.centroid
            
            # Find nearby features within radius
            distances, indices = kdtree.query([(x, y)], k=min(20, len(features)))
            
            for dist, idx in zip(distances[0], indices[0]):
                if idx == i or dist == 0:
                    continue
                
                if dist > radius_px:
                    continue
                
                other = features[idx]
                
                # Check if path is traversable
                if self._is_path_traversable(f, other, heightmap, slope):
                    connectivity[f.feature_id].add(other.feature_id)
        
        return connectivity
    
    def _distance_based_connectivity(self, features: List[TerrainFeature]) -> Dict[str, Set[str]]:
        """Fallback: connect features within a simple distance threshold."""
        connectivity = {f.feature_id: set() for f in features}
        
        feature_coords = [(f.centroid[0], f.centroid[1]) for f in features]
        feature_ids = [f.feature_id for f in features]
        kdtree = KDTree(feature_coords)
        
        for i, f in enumerate(features):
            distances, indices = kdtree.query([feature_coords[i]], k=10)
            for dist, idx in zip(distances[0], indices[0]):
                if idx != i and dist < 50:  # 50 pixels threshold
                    connectivity[f.feature_id].add(feature_ids[idx])
        
        return connectivity
    
    def _is_path_traversable(self, f1: TerrainFeature, f2: TerrainFeature,
                              heightmap: Heightmap, slope: np.ndarray) -> bool:
        """
        Check if the straight-line path between features is traversable.
        
        The path is traversable if all points along the line have slope
        within vehicle limits.
        """
        x1, y1 = f1.centroid
        x2, y2 = f2.centroid
        
        # Bresenham line algorithm
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        x, y = x1, y1
        
        while (x, y) != (x2, y2):
            # Check slope at this point
            if 0 <= y < slope.shape[0] and 0 <= x < slope.shape[1]:
                if slope[y, x] > self.config.vehicle_climb_angle:
                    return False  # Too steep
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return True
    
    def _delineate_watersheds(self, features: List[TerrainFeature],
                                heightmap: Heightmap,
                                aspect: Optional[np.ndarray],
                                curvature: Optional[np.ndarray]) -> Dict[str, Set[str]]:
        """
        Delineate watersheds based on aspect and curvature.
        
        Each watershed is a drainage basin that flows to a common outlet.
        """
        watersheds = {}
        
        if aspect is None:
            warnings.warn("Aspect not available, watershed delineation skipped")
            return watersheds
        
        # Find valley features (these are potential watershed outlets)
        valleys = [f for f in features if isinstance(f, ValleyFeature)]
        
        if not valleys:
            return watersheds
        
        # For each valley, create a watershed
        for valley in valleys:
            basin_id = valley.feature_id
            basin_members = set()
            
            # Simplified: find features that flow toward this valley
            # In a full implementation, this would use flow accumulation
            # and watershed labeling algorithms
            
            # Add features within a certain distance
            x, y = valley.centroid
            radius = 100  # pixels
            
            for f in features:
                fx, fy = f.centroid
                if np.sqrt((fx - x)**2 + (fy - y)**2) < radius:
                    basin_members.add(f.feature_id)
            
            if basin_members:
                watersheds[basin_id] = basin_members
        
        return watersheds
    
    def _log_graph_statistics(self, visibility: Dict, flow: Dict, 
                               connectivity: Dict, watersheds: Dict) -> None:
        """Log graph statistics for debugging."""
        print("\n=== Graph Statistics ===")
        
        # Visibility graph
        vis_edges = sum(len(v) for v in visibility.values())
        vis_nodes = len([v for v in visibility.values() if v])
        print(f"Visibility: {vis_nodes} connected nodes, {vis_edges // 2} edges")
        
        # Flow network
        flow_edges = sum(len(v) for v in flow.values())
        flow_nodes = len([v for v in flow.values() if v])
        print(f"Flow Network: {flow_nodes} nodes with downstream, {flow_edges} edges")
        
        # Connectivity graph
        conn_edges = sum(len(v) for v in connectivity.values())
        conn_nodes = len([v for v in connectivity.values() if v])
        print(f"Connectivity: {conn_nodes} connected nodes, {conn_edges // 2} edges")
        
        # Watersheds
        print(f"Watersheds: {len(watersheds)} basins")
        if watersheds:
            avg_basin_size = np.mean([len(b) for b in watersheds.values()])
            print(f"  Average basin size: {avg_basin_size:.1f} features")
    
    @property
    def output_schema(self) -> dict:
        return {
            "visibility_graph": {"type": "Dict[str, Set[str]]", 
                                 "description": "feature_id → visible feature_ids"},
            "flow_network": {"type": "Dict[str, List[str]]", 
                            "description": "feature_id → downstream feature_ids"},
            "connectivity_graph": {"type": "Dict[str, Set[str]]", 
                                  "description": "feature_id → adjacent traversable feature_ids"},
            "watersheds": {"type": "Dict[str, Set[str]]", 
                          "description": "basin_id → member feature_ids"},
            "flow_accumulation": {"type": "ScalarField", 
                                 "description": "per-pixel upstream area"}
        }