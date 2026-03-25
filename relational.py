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
    PixelCoord, Traversability, LayerBundle, GameType
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
        self.game_type = config.game_type
        self._visibility_max_range_m = config.visibility_max_range_m
        self._connection_radius_m = config.connection_radius_m
        self._viewshed_sample_step = config.viewshed_sample_step_px
        self._flow_step_px = config.flow_step_px
        self._flow_neighbor_distance_px = config.flow_neighbor_distance_px
    
    def _log(self, msg: str) -> None:
        """Helper for verbose logging"""
        if self.config.verbose:
            print(f"[Relational] {msg}")
    
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
        self._log("Building Visibility Graph...")
        visibility_graph = self._build_visibility_graph(features, heightmap)
        
        # 2. Flow Network (water flow direction between features)
        self._log("Building Flow Network...")
        flow_network, flow_accumulation = self._build_flow_network(
            features, heightmap, aspect
        )
        
        # 3. Connectivity Graph (traversable paths)
        self._log("Building Connectivity Graph...")
        connectivity_graph = self._build_connectivity_graph(
            features, heightmap, slope
        )
        
        # 4. Watersheds (drainage basins)
        self._log("Delineating Watersheds")
        #watersheds = self._delineate_watersheds(features, heightmap, aspect, curvature)
        # first, compute flow direction if we have aspect
        if aspect is not None:
            flow_dir = self._compute_flow_direction(heightmap)
            
            # Identify outlets (valley features or lowest points)
            outlets = self._identify_watershed_outlets(features, heightmap, flow_dir)
            
            # Delineate watersheds using flow direction
            watershed_labels = self._delineate_watersheds(heightmap, flow_dir, outlets)
            
            # Convert labeled watersheds to feature sets
            watersheds = self._watershed_labels_to_features(watershed_labels, features)
        else:
            self._log(f"Warning: Aspect not available, watershed delineation skipped")
            watersheds = {}
        
        # Log statistics
        self._log_graph_statistics(visibility_graph, flow_network, connectivity_graph, watersheds)
        
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
        
        if not features:
            self._log("No features available, visibility network will be empty")
            return {}
            
        visibility = {f.feature_id: set() for f in features}
        
        # Extract centroids
        centroids = []
        feature_ids = []
        for f in features:
            x, y = f.centroid
            centroids.append((x, y))
            feature_ids.append(f.feature_id)
        
        # Visibility is only meaningful between peaks — they are the observation/defensive
        # nodes. Saddle-to-saddle LOS is not tactically useful and with 100s of saddles
        # the O(n²) Bresenham cost dominates total pipeline time.
        # Include peaks, ridges, and saddles for visibility
        important_features = [f for f in features if isinstance(f, (PeakFeature, RidgeFeature, SaddleFeature))]
        
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
        
        Uses config.viewshed_sample_step_px to skip pixels for performance.
        Larger step = faster but may miss small blockers.
        
        Args:
            p1, p2: Pixel coordinates of the two points
            heightmap: Heightmap with elevation data
        
        Returns:
            True if line of sight is clear, False if terrain blocks view
        """
        x1, y1 = p1
        x2, y2 = p2
        
        rows, cols = heightmap.data.shape
        
        # Boundary check - ensure points are within map
        if not (0 <= x1 < cols and 0 <= y1 < rows and 
                0 <= x2 < cols and 0 <= y2 < rows):
            return False
        
        # Get elevation at endpoints
        z1 = heightmap.data[y1, x1]
        z2 = heightmap.data[y2, x2]
        
        # Calculate line length in pixels
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        distance = max(dx, dy)
        
        # If points are adjacent or same, always visible
        if distance <= 1:
            return True
        
        # Get step size from config (default to 1 if not set)
        step = max(1, getattr(self.config, 'viewshed_sample_step_px', 1))
        
        # Adjust step to ensure at least 2 samples
        step = min(step, distance // 2)
        
        # Pre-calculate line parameters
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        
        # Use parameter t for interpolation (0 to 1)
        for t in np.linspace(0, 1, max(2, distance // step + 1)):
            # Skip endpoints (already checked)
            if t == 0 or t == 1:
                continue
            
            # Calculate point along line
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            
            # Clamp to map bounds (safety)
            x = max(0, min(cols - 1, x))
            y = max(0, min(rows - 1, y))
            
            # Skip if we're at an endpoint due to rounding
            if (x, y) == (x1, y1) or (x, y) == (x2, y2):
                continue
            
            # Calculate interpolated elevation at this point
            # Linear interpolation along line (simpler and faster)
            interpolated_z = z1 + t * (z2 - z1)
            
            # Get actual terrain elevation
            terrain_z = heightmap.data[y, x]
            
            # If terrain blocks view, return False
            # Add small epsilon (0.1m) to handle numerical precision
            if terrain_z > interpolated_z + 0.1:
                return False
        
        return True
    
    def _identify_watershed_outlets(self, features: List[TerrainFeature], 
                                 heightmap: Heightmap, 
                                 flow_dir: np.ndarray) -> List[Tuple[int, int]]:
        """
        Identify watershed outlets from features and flow patterns.
        
        Outlets are typically:
        - Valley features (lowest points in drainage)
        - Local minima in flow accumulation
        - Feature points with no outgoing flow
        """
        outlets = []
        
        # Method 1: Use valley features as outlets
        valleys = [f for f in features if isinstance(f, ValleyFeature)]
        for valley in valleys:
            x, y = valley.centroid
            outlets.append((int(x), int(y)))
        
        # Method 2: Find local minima in flow direction (cells with no outflow)
        rows, cols = heightmap.shape
        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                if flow_dir[y, x] == 0:  # No flow direction (pit or flat)
                    # Check if it's a local minimum
                    current_z = heightmap.data[y, x]
                    is_min = True
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < rows and 0 <= nx < cols:
                                if heightmap.data[ny, nx] < current_z:
                                    is_min = False
                                    break
                        if not is_min:
                            break
                    
                    if is_min and len(outlets) < 100:  # Limit outlets
                        outlets.append((x, y))
        
        # Remove duplicate outlets (same pixel)
        outlets = list(set(outlets))
        
        self._log(f"Identified {len(outlets)} watershed outlets")
        return outlets
    
    def _compute_flow_direction(self, heightmap: Heightmap) -> np.ndarray:
        """
        D8 flow direction algorithm.
        
        For each pixel, find the steepest downhill neighbor.
        
        Returns:
            flow_dir: 2D array with values 0-8
                0 = flat/no flow
                1 = North      (N)
                2 = Northeast  (NE)
                3 = East       (E)
                4 = Southeast  (SE)
                5 = South      (S)
                6 = Southwest  (SW)
                7 = West       (W)
                8 = Northwest  (NW)
        """
        z = heightmap.data
        rows, cols = z.shape
        flow_dir = np.zeros((rows, cols), dtype=np.uint8)
        
        # 8-direction offsets and their corresponding flow direction codes
        # Order: N, NE, E, SE, S, SW, W, NW
        directions = [
            (-1, 0, 1),   # North
            (-1, 1, 2),   # Northeast
            (0, 1, 3),    # East
            (1, 1, 4),    # Southeast
            (1, 0, 5),    # South
            (1, -1, 6),   # Southwest
            (0, -1, 7),   # West
            (-1, -1, 8)   # Northwest
        ]
        
        # Skip border pixels (can't compute full neighborhood)
        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                current_elev = z[y, x]
                max_drop = 0
                max_dir = 0
                
                # Check all 8 neighbors
                for dy, dx, dir_code in directions:
                    ny, nx = y + dy, x + dx
                    neighbor_elev = z[ny, nx]
                    drop = current_elev - neighbor_elev
                    
                    # Only consider downhill directions
                    if drop > max_drop:
                        max_drop = drop
                        max_dir = dir_code
                
                # Assign flow direction if there's a downhill path
                if max_drop > 0:
                    flow_dir[y, x] = max_dir
                else:
                    flow_dir[y, x] = 0  # Flat or pit
        
        return flow_dir
    
    def _compute_flow_accumulation(self, flow_dir: np.ndarray) -> np.ndarray:
        """
        Compute flow accumulation from D8 flow direction.
        
        Accumulates the number of upstream cells that flow into each cell.
        """
        rows, cols = flow_dir.shape
        accumulation = np.ones((rows, cols), dtype=np.float32)  # Each cell counts itself
        
        # Define reverse mapping: which neighbors flow into this cell
        # For each direction code, which neighbor directions would point to this cell
        reverse_map = {
            1: 5,   # North: cells flowing South point here
            2: 6,   # Northeast: cells flowing Southwest point here
            3: 7,   # East: cells flowing West point here
            4: 8,   # Southeast: cells flowing Northwest point here
            5: 1,   # South: cells flowing North point here
            6: 2,   # Southwest: cells flowing Northeast point here
            7: 3,   # West: cells flowing East point here
            8: 4    # Northwest: cells flowing Southeast point here
        }
        
        # Direction offsets for finding upstream cells
        dir_offsets = {
            1: (-1, 0),   # North
            2: (-1, 1),   # Northeast
            3: (0, 1),    # East
            4: (1, 1),    # Southeast
            5: (1, 0),    # South
            6: (1, -1),   # Southwest
            7: (0, -1),   # West
            8: (-1, -1)   # Northwest
        }
        
        # Iterate until convergence (simple iterative approach)
        # For production, use topological order (flow direction is DAG)
        changed = True
        max_iter = rows * cols
        iter_count = 0
        
        while changed and iter_count < max_iter:
            changed = False
            for y in range(rows):
                for x in range(cols):
                    dir_code = flow_dir[y, x]
                    if dir_code > 0:
                        # Get downstream cell
                        dy, dx = dir_offsets[dir_code]
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < rows and 0 <= nx < cols:
                            # Add current accumulation to downstream
                            new_val = accumulation[y, x] + 1
                            if new_val > accumulation[ny, nx]:
                                accumulation[ny, nx] = new_val
                                changed = True
        
        return accumulation
    
    def _build_flow_network(self, features: List[TerrainFeature],
                            heightmap: Heightmap,
                            aspect: Optional[np.ndarray]) -> Tuple[Dict[str, List[str]], ScalarField]:
        """
        Build flow network based on aspect (flow direction).
        """
        
        if not features:
            self._log("No features available, flow network will be empty")
            return {}, None
            
        flow_network = {f.feature_id: [] for f in features}
        
        if aspect is None:
            self._log(f"Aspect not available, flow network will be empty")
            return flow_network, None
        
        # NEW: Use proper D8 flow direction instead of simplified model
        flow_dir = self._compute_flow_direction(heightmap)
        
        # Build feature KD-tree for nearest neighbor search
        feature_coords = [(f.centroid[0], f.centroid[1]) for f in features]
        feature_ids = [f.feature_id for f in features]
        kdtree = KDTree(feature_coords)
        
        # Initialize flow accumulation
        flow_accumulation = np.zeros(heightmap.shape, dtype=np.float32)
        
        # TODO: Implement proper flow accumulation from flow_dir
        # For now, use simplified model with flow_dir
        
        rows, cols = heightmap.shape
        for f in features:
            x, y = f.centroid
            if 0 <= x < cols and 0 <= y < rows:
                # Get flow direction at this point
                direction = flow_dir[y, x]
                if direction > 0:  # Has flow
                    # Direction encoding: 1-8 (N, NE, E, SE, S, SW, W, NW)
                    # Convert to step vector
                    dir_map = {
                        1: (0, -1),   # North
                        2: (1, -1),   # Northeast
                        3: (1, 0),    # East
                        4: (1, 1),    # Southeast
                        5: (0, 1),    # South
                        6: (-1, 1),   # Southwest
                        7: (-1, 0),   # West
                        8: (-1, -1)   # Northwest
                    }
                    dx, dy = dir_map.get(direction, (0, 0))
                    
                    # Flow destination
                    flow_x = x + dx * self._flow_step_px
                    flow_y = y + dy * self._flow_step_px
                    
                    # Find nearest feature in flow direction
                    distances, indices = kdtree.query([(flow_x, flow_y)])
                    
                    if distances[0] < self._flow_neighbor_distance_px:
                        downstream_id = feature_ids[indices[0]]
                        if downstream_id != f.feature_id:
                            flow_network[f.feature_id].append(downstream_id)
        
        return flow_network, flow_accumulation
    
    # ######################
    
    def _build_connectivity_graph(self, features: List[TerrainFeature],
                                   heightmap: Heightmap,
                                   slope: Optional[np.ndarray]) -> Dict[str, Set[str]]:
        """
        Build connectivity graph based on traversability between features.
        
        Features are connected if there exists a traversable path between them
        respecting the vehicle's slope limits.
        """
        
        if not features:
            self._log("No features available, connectivity network will be empty")
            return {}
        
        connectivity = {f.feature_id: set() for f in features}
        
        # If no slope, use simple distance-based connectivity
        if slope is None:
            self._log(f"Slope not available, using distance-based connectivity")
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
                
                # Check if path is traversable using config vehicle limit
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
        Check if the path between two features is traversable.
        
        Returns True if the entire path has slope <= vehicle_climb_angle.
        Paths with slopes > cliff_threshold are considered BLOCKED (False).
        
        Args:
            f1, f2: Features to check connectivity between
            heightmap: Heightmap for scale conversion
            slope: Slope field in degrees
        
        Returns:
            True if traversable, False if blocked or exceeds vehicle limits
        """
        x1, y1 = f1.centroid
        x2, y2 = f2.centroid
        
        # Get config thresholds
        max_slope = self.config.vehicle_climb_angle      # [20-45] degrees
        cliff_threshold = self.config.cliff_threshold_degrees  # [30-60] degrees
        
        # If distance > max range, not traversable
        dist_px = np.hypot(x2 - x1, y2 - y1)
        max_range_px = self._connection_radius_m / heightmap.config.horizontal_scale
        if dist_px > max_range_px:
            return False
        
        # Check slope along path using Bresenham
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        x, y = x1, y1
        
        while (x, y) != (x2, y2):
            # Skip the start point (already at feature)
            if (x, y) != (x1, y1):
                # Check bounds
                if 0 <= y < slope.shape[0] and 0 <= x < slope.shape[1]:
                    current_slope = slope[y, x]
                    
                    # Cliff: absolutely blocked
                    if current_slope > cliff_threshold:
                        return False
                    # Steep but within vehicle limits: still traversable (True)
                    if current_slope > max_slope:
                        return False
            
            # Bresenham algorithm
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        # Check the end point as well
        if 0 <= y2 < slope.shape[0] and 0 <= x2 < slope.shape[1]:
            if slope[y2, x2] > max_slope or slope[y2, x2] > cliff_threshold:
                return False
        
        return True

    def _check_path_slope(self, x1: int, y1: int, x2: int, y2: int,
                          slope: np.ndarray, max_slope: float) -> bool:
        """Check if all points along line have slope <= max_slope."""
        # Bresenham implementation
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        x, y = x1, y1
        while (x, y) != (x2, y2):
            if 0 <= y < slope.shape[0] and 0 <= x < slope.shape[1]:
                if slope[y, x] > max_slope:
                    return False
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return True
    
    def _delineate_watersheds(self, heightmap: Heightmap, 
                          flow_dir: np.ndarray, 
                          outlets: List[Tuple[int, int]]) -> np.ndarray:
        """
        Label watersheds using flow direction.
        
        Walk upstream from outlet points using reverse flow direction.
        
        Returns:
            watershed_labels: 2D array with integer labels for each watershed
        """
        rows, cols = heightmap.shape
        watershed_labels = np.zeros((rows, cols), dtype=np.int32)
        
        # Direction offsets for finding upstream cells
        # For each flow direction code, which neighbor directions would point here?
        reverse_dir_map = {
            1: 5,   # North: cells flowing South point here
            2: 6,   # Northeast: cells flowing Southwest point here
            3: 7,   # East: cells flowing West point here
            4: 8,   # Southeast: cells flowing Northwest point here
            5: 1,   # South: cells flowing North point here
            6: 2,   # Southwest: cells flowing Northeast point here
            7: 3,   # West: cells flowing East point here
            8: 4    # Northwest: cells flowing Southeast point here
        }
        
        # Direction offsets for neighbors
        neighbor_offsets = {
            1: (-1, 0),   # North
            2: (-1, 1),   # Northeast
            3: (0, 1),    # East
            4: (1, 1),    # Southeast
            5: (1, 0),    # South
            6: (1, -1),   # Southwest
            7: (0, -1),   # West
            8: (-1, -1)   # Northwest
        }
        
        # Process each outlet
        for outlet_id, (ox, oy) in enumerate(outlets, start=1):
            # BFS/DFS upstream from outlet
            stack = [(ox, oy)]
            watershed_labels[oy, ox] = outlet_id
            
            while stack:
                cx, cy = stack.pop()
                current_dir = flow_dir[cy, cx]
                
                # Find cells that flow into this cell
                # For each possible flow direction that could point to (cx, cy)
                for neighbor_dir, (dy, dx) in neighbor_offsets.items():
                    nx, ny = cx + dx, cy + dy  # Note: x,y order careful!
                    
                    if 0 <= nx < cols and 0 <= ny < rows:
                        # Check if neighbor flows to current cell
                        neighbor_flow = flow_dir[ny, nx]
                        if neighbor_flow > 0:
                            # Check if neighbor's flow direction points to current cell
                            if neighbor_flow == reverse_dir_map.get(current_dir, 0) or \
                               (current_dir == 0 and neighbor_flow > 0):
                                if watershed_labels[ny, nx] == 0:
                                    watershed_labels[ny, nx] = outlet_id
                                    stack.append((nx, ny))
        
        return watershed_labels


    def _watershed_labels_to_features(self, watershed_labels: np.ndarray,
                                       features: List[TerrainFeature]) -> Dict[str, Set[str]]:
        """
        Convert watershed labels to feature ID sets.
        
        Returns:
            Dict[basin_id, Set[feature_ids]] where basin_id is the outlet feature ID
        """
        watersheds = {}
        
        # Create a lookup for features by their centroid pixel
        feature_pixels = {}
        for f in features:
            x, y = f.centroid
            if 0 <= y < watershed_labels.shape[0] and 0 <= x < watershed_labels.shape[1]:
                feature_pixels[(int(x), int(y))] = f.feature_id
        
        # Group features by watershed label
        for (x, y), feature_id in feature_pixels.items():
            basin_id = watershed_labels[y, x]
            if basin_id > 0:
                basin_key = f"basin_{basin_id}"
                if basin_key not in watersheds:
                    watersheds[basin_key] = set()
                watersheds[basin_key].add(feature_id)
        
        return watersheds
        
    def _log_graph_statistics(self, visibility: Dict, flow: Dict, 
                               connectivity: Dict, watersheds: Dict) -> None:
        """Log graph statistics for debugging."""
        # Visibility graph
        vis_edges = sum(len(v) for v in visibility.values())
        vis_nodes = len([v for v in visibility.values() if v])
        self._log(f"Visibility: {vis_nodes} connected nodes, {vis_edges // 2} edges")
        
        # Flow network
        flow_edges = sum(len(v) for v in flow.values())
        flow_nodes = len([v for v in flow.values() if v])
        self._log(f"Flow Network: {flow_nodes} nodes with downstream, {flow_edges} edges")
        
        # Connectivity graph
        conn_edges = sum(len(v) for v in connectivity.values())
        conn_nodes = len([v for v in connectivity.values() if v])
        self._log(f"Connectivity: {conn_nodes} connected nodes, {conn_edges // 2} edges")
        
        # Watersheds
        self._log(f"Watersheds: {len(watersheds)} basins")
        if watersheds:
            avg_basin_size = np.mean([len(b) for b in watersheds.values()])
            self._log(f"Average basin size: {avg_basin_size:.1f} features")
    
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