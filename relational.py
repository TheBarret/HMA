"""
Layer 4: Relational Analysis

Builds graphs connecting terrain features:
- Visibility graph: which features can see each other
- Flow network: water flow between features
- Connectivity graph: traversable paths between features
- Watersheds: drainage basins
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import KDTree
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import deque

from core import (
    Heightmap, ScalarField, PipelineConfig, PipelineLayer,
    TerrainFeature, ClassifiedFeature,
    PeakFeature, RidgeFeature, ValleyFeature, SaddleFeature, FlatZoneFeature,
    PixelCoord, Traversability, LayerBundle
)


class Layer4_Relational(PipelineLayer[Dict[str, Any]]):
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
        self._visibility_max_range_px = config.visibility_max_range_m / config.horizontal_scale
        self._connection_radius_px = config.connection_radius_m / config.horizontal_scale
        self._viewshed_sample_step = config.viewshed_sample_step_px
        self._flow_step_px = config.flow_step_px
        self._flow_neighbor_distance_px = config.flow_neighbor_distance_px
        self._watershed_min_area_px = config.watershed_min_area_m2 / (config.horizontal_scale ** 2)
        self._vehicle_climb_angle = config.vehicle_climb_angle
        self._cliff_threshold = config.cliff_threshold_degrees
        self._visibility_epsilon_m = config.visibility_epsilon_m
        self._max_watershed_outlets = config.max_watershed_outlets
        self._watershed_area_estimate_factor = config.watershed_area_estimate_factor
        self._connectivity_max_neighbors = config.connectivity_max_neighbors
    
    def _log(self, msg: str) -> None:
        """Helper for verbose logging"""
        if self.config.verbose:
            print(f"[Relational] {msg}")
    
    def execute(self, input_data: LayerBundle) -> Dict[str, Any]:
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
        
        self._log(f"Building relational graphs | features={len(features)}")
        
        # Build feature index for quick lookup
        self._feature_index = {f.feature_id: f for f in features}
        self._feature_coords = [(f.centroid[0], f.centroid[1]) for f in features]
        self._feature_ids = [f.feature_id for f in features]
        self._feature_kdtree = KDTree(self._feature_coords) if features else None
        
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
        watersheds = self._delineate_watersheds(
            features, heightmap, aspect, curvature, flow_network
        )
        
        # Log statistics
        self._log_graph_statistics(visibility_graph, flow_network, connectivity_graph, watersheds)
        
        return {
            "visibility_graph": visibility_graph,
            "flow_network": flow_network,
            "connectivity_graph": connectivity_graph,
            "watersheds": watersheds,
            "flow_accumulation": flow_accumulation
        }
    
    # =========================================================================
    # 1. VISIBILITY GRAPH
    # =========================================================================
    
    def _build_visibility_graph(self, features: List[TerrainFeature], 
                                heightmap: Heightmap) -> Dict[str, Set[str]]:
        """
        Build visibility graph using line-of-sight between feature centroids.
        
        Only peaks, ridges, and saddles are considered for visibility.
        """
        if not features or len(features) < 2:
            return {f.feature_id: set() for f in features}
        
        visibility = {f.feature_id: set() for f in features}
        
        # Only important features for visibility (peaks, ridges, saddles)
        important = [f for f in features if isinstance(f, (PeakFeature, RidgeFeature, SaddleFeature))]
        
        if len(important) < 2:
            return visibility
        
        self._log(f"Checking visibility between {len(important)} important features")
        
        n = len(important)
        checked = 0
        
        for i in range(n):
            f1 = important[i]
            x1, y1 = f1.centroid
            
            for j in range(i + 1, n):
                f2 = important[j]
                x2, y2 = f2.centroid
                
                # Fast distance rejection
                dx = x1 - x2
                dy = y1 - y2
                if (dx * dx + dy * dy) > self._visibility_max_range_px ** 2:
                    continue
                
                if self._check_visibility((x1, y1), (x2, y2), heightmap):
                    visibility[f1.feature_id].add(f2.feature_id)
                    visibility[f2.feature_id].add(f1.feature_id)
                
                checked += 1
                if checked % 1000 == 0:
                    self._log(f"  Visibility checks: {checked}")
        
        self._log(f"Visibility complete | checks={checked}")
        return visibility
    
    def _check_visibility(self, p1: PixelCoord, p2: PixelCoord, 
                          heightmap: Heightmap) -> bool:
        """
        Check if two points are mutually visible using Bresenham line algorithm.
        """
        x1, y1 = p1
        x2, y2 = p2
        
        rows, cols = heightmap.data.shape
        
        # Boundary check
        if not (0 <= x1 < cols and 0 <= y1 < rows and 
                0 <= x2 < cols and 0 <= y2 < rows):
            return False
        
        z1 = heightmap.data[y1, x1]
        z2 = heightmap.data[y2, x2]
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        distance = max(dx, dy)
        
        if distance <= 1:
            return True
        
        # Bresenham line traversal
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        x, y = x1, y1
        
        # Sample step (skip pixels for performance)
        step = max(1, self._viewshed_sample_step)
        sample_counter = 0
        
        while (x, y) != (x2, y2):
            # Only check every Nth pixel
            sample_counter += 1
            if sample_counter >= step:
                sample_counter = 0
                
                if (x, y) != (x1, y1) and (x, y) != (x2, y2):
                    # Linear interpolation of line elevation
                    t = np.hypot(x - x1, y - y1) / distance
                    line_z = z1 + t * (z2 - z1)
                    terrain_z = heightmap.data[y, x]
                    
                    if terrain_z > line_z + self._visibility_epsilon_m:
                        return False
            
            # Bresenham update
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return True
    
    # =========================================================================
    # 2. FLOW NETWORK
    # =========================================================================
    
    def _build_flow_network(self, features: List[TerrainFeature],
                            heightmap: Heightmap,
                            aspect: Optional[np.ndarray]) -> Tuple[Dict[str, List[str]], ScalarField]:
        """
        Build flow network based on D8 flow direction.
        
        Returns:
            flow_network: feature_id → list of downstream feature_ids
            flow_accumulation: per-pixel upstream area (pixels²)
        """
        flow_network = {f.feature_id: [] for f in features}
        
        if not features or aspect is None:
            self._log("Flow network skipped: no features or no aspect")
            return flow_network, np.zeros(heightmap.shape, dtype=np.float32)
        
        # Compute D8 flow direction
        flow_dir = self._compute_flow_direction(heightmap)
        
        # Compute flow accumulation using topological order
        flow_accumulation = self._compute_flow_accumulation(flow_dir)
        
        # Build feature KD-tree for nearest neighbor
        if self._feature_kdtree is None:
            return flow_network, flow_accumulation
        
        rows, cols = heightmap.shape
        
        for f in features:
            x, y = f.centroid
            
            if not (0 <= x < cols and 0 <= y < rows):
                continue
            
            dir_code = flow_dir[y, x]
            if dir_code == 0:
                continue  # Flat or pit
            
            # Direction vectors (dx, dy) for D8 codes
            dir_vectors = {
                1: (0, -1),   # North
                2: (1, -1),   # Northeast
                3: (1, 0),    # East
                4: (1, 1),    # Southeast
                5: (0, 1),    # South
                6: (-1, 1),   # Southwest
                7: (-1, 0),   # West
                8: (-1, -1)   # Northwest
            }
            
            dx, dy = dir_vectors.get(dir_code, (0, 0))
            
            # Project flow direction
            flow_x = x + dx * self._flow_step_px
            flow_y = y + dy * self._flow_step_px
            
            # Find nearest feature to flow destination
            distances, indices = self._feature_kdtree.query([(flow_x, flow_y)])
            
            if distances[0] < self._flow_neighbor_distance_px:
                downstream_id = self._feature_ids[indices[0]]
                if downstream_id != f.feature_id:
                    flow_network[f.feature_id].append(downstream_id)
        
        return flow_network, flow_accumulation
    
    def _compute_flow_direction(self, heightmap: Heightmap) -> np.ndarray:
        """
        D8 flow direction algorithm.
        
        Returns:
            flow_dir: 2D array with values 0-8
                0 = flat/no flow
                1 = North, 2 = Northeast, 3 = East, 4 = Southeast,
                5 = South, 6 = Southwest, 7 = West, 8 = Northwest
        """
        z = heightmap.data
        rows, cols = z.shape
        flow_dir = np.zeros((rows, cols), dtype=np.uint8)
        
        # Direction offsets: (dy, dx, code)
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
        
        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                current = z[y, x]
                max_drop = 0
                best_dir = 0
                
                for dy, dx, code in directions:
                    ny, nx = y + dy, x + dx
                    drop = current - z[ny, nx]
                    
                    if drop > max_drop:
                        max_drop = drop
                        best_dir = code
                
                if max_drop > 0:
                    flow_dir[y, x] = best_dir
        
        return flow_dir
    
    def _compute_flow_accumulation(self, flow_dir: np.ndarray) -> ScalarField:
        """
        Compute flow accumulation using topological order.
        
        Uses float64 internally to prevent overflow, returns float32 for memory efficiency.
        
        Returns:
            accumulation: number of upstream cells (including self) as float32
        """
        rows, cols = flow_dir.shape
        accumulation = np.ones((rows, cols), dtype=np.float64)
        
        # Direction vectors for downstream movement
        dir_vectors = {
            1: (0, -1), 2: (1, -1), 3: (1, 0), 4: (1, 1),
            5: (0, 1), 6: (-1, 1), 7: (-1, 0), 8: (-1, -1)
        }
        
        # Build adjacency: cell → downstream cell
        downstream = {}
        for y in range(rows):
            for x in range(cols):
                code = flow_dir[y, x]
                if code > 0:
                    dx, dy = dir_vectors[code]
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < cols and 0 <= ny < rows:
                        downstream[(x, y)] = (nx, ny)
        
        # Count incoming edges (how many cells flow into each cell)
        in_degree = np.zeros((rows, cols), dtype=np.int32)
        for (x, y), (nx, ny) in downstream.items():
            in_degree[ny, nx] += 1
        
        # Initialize queue with cells that have no incoming flow (sources)
        queue = deque()
        for y in range(rows):
            for x in range(cols):
                if flow_dir[y, x] > 0 and in_degree[y, x] == 0:
                    queue.append((x, y))
        
        # Process in topological order
        while queue:
            x, y = queue.popleft()
            
            if (x, y) in downstream:
                nx, ny = downstream[(x, y)]
                
                # Add accumulation from current cell to downstream
                accumulation[ny, nx] += accumulation[y, x]
                
                # Decrement in-degree of downstream cell
                in_degree[ny, nx] -= 1
                
                # If downstream cell has no more incoming flows, add to queue
                if in_degree[ny, nx] == 0:
                    queue.append((nx, ny))
        
        # Cap extreme values to prevent float32 overflow
        max_accumulation = rows * cols
        accumulation = np.clip(accumulation, 0, max_accumulation)
        
        return accumulation.astype(np.float32)
    
    # =========================================================================
    # 3. CONNECTIVITY GRAPH
    # =========================================================================
    
    def _build_connectivity_graph(self, features: List[TerrainFeature],
                                   heightmap: Heightmap,
                                   slope: Optional[np.ndarray]) -> Dict[str, Set[str]]:
        """
        Build connectivity graph based on traversability between features.
        """
        connectivity = {f.feature_id: set() for f in features}
        
        if not features or len(features) < 2:
            return connectivity
        
        if slope is None:
            self._log("Slope not available, using distance-based connectivity")
            return self._distance_based_connectivity(features, heightmap)
        
        if self._feature_kdtree is None:
            return connectivity
        
        n = len(features)
        self._log(f"Checking connectivity between {n} features")
        
        checked = 0
        connected = 0
        
        for i, f1 in enumerate(features):
            x1, y1 = f1.centroid
            
            # Find nearby features within radius
            distances, indices = self._feature_kdtree.query([(x1, y1)], k=min(self._connectivity_max_neighbors, n))
            
            for dist, idx in zip(distances[0], indices[0]):
                if idx <= i or dist == 0:
                    continue
                
                if dist > self._connection_radius_px:
                    continue
                
                f2 = features[idx]
                
                if self._is_path_traversable(f1, f2, heightmap, slope):
                    connectivity[f1.feature_id].add(f2.feature_id)
                    connectivity[f2.feature_id].add(f1.feature_id)
                    connected += 1
                
                checked += 1
        
        self._log(f"Connectivity complete | checks={checked}, connections={connected}")
        return connectivity
    
    def _distance_based_connectivity(self, features: List[TerrainFeature],
                                      heightmap: Heightmap) -> Dict[str, Set[str]]:
        """
        Fallback: connect features within distance threshold.
        """
        connectivity = {f.feature_id: set() for f in features}
        
        if self._feature_kdtree is None:
            return connectivity
        
        n = len(features)
        
        for i, f1 in enumerate(features):
            distances, indices = self._feature_kdtree.query([self._feature_coords[i]], k=min(self._connectivity_max_neighbors, n))
            
            for dist, idx in zip(distances[0], indices[0]):
                if idx <= i or dist == 0:
                    continue
                
                if dist < self._connection_radius_px:
                    f2 = features[idx]
                    connectivity[f1.feature_id].add(f2.feature_id)
                    connectivity[f2.feature_id].add(f1.feature_id)
        
        return connectivity
    
    def _is_path_traversable(self, f1: TerrainFeature, f2: TerrainFeature,
                             heightmap: Heightmap, slope: np.ndarray) -> bool:
        """
        Check if the path between two features is traversable.
        
        Returns True if all points along path have slope <= vehicle_climb_angle
        and no points exceed cliff_threshold.
        """
        x1, y1 = f1.centroid
        x2, y2 = f2.centroid
        
        rows, cols = heightmap.shape
        
        # Bresenham line traversal
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        x, y = x1, y1
        
        while (x, y) != (x2, y2):
            if (x, y) != (x1, y1):
                if 0 <= y < rows and 0 <= x < cols:
                    s = slope[y, x]
                    if s > self._cliff_threshold or s > self._vehicle_climb_angle:
                        return False
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        # Check endpoint
        if 0 <= y2 < rows and 0 <= x2 < cols:
            if slope[y2, x2] > self._cliff_threshold or slope[y2, x2] > self._vehicle_climb_angle:
                return False
        
        return True
    
    # =========================================================================
    # 4. WATERSHEDS
    # =========================================================================
    
    def _delineate_watersheds(self, features: List[TerrainFeature],
                               heightmap: Heightmap,
                               aspect: Optional[np.ndarray],
                               curvature: Optional[np.ndarray],
                               flow_network: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        """
        Delineate watersheds using flow direction.
        
        Returns:
            Dict[basin_id, Set[feature_ids]] where basin_id is outlet feature ID
        """
        if not features or aspect is None:
            return {}
        
        # Compute flow direction
        flow_dir = self._compute_flow_direction(heightmap)
        
        # Identify outlets from valley features
        outlets = self._identify_outlets(features, heightmap, flow_dir)
        
        if not outlets:
            return {}
        
        # Label watersheds by upstream propagation
        watershed_labels = self._label_watersheds(heightmap, flow_dir, outlets)
        
        # Convert to feature sets using flow network
        watersheds = self._assign_features_to_watersheds(watershed_labels, features, flow_network)
        
        # Filter by minimum area
        watersheds = self._filter_watersheds_by_area(watersheds, features, heightmap)
        
        return watersheds
    
    def _identify_outlets(self, features: List[TerrainFeature],
                          heightmap: Heightmap,
                          flow_dir: np.ndarray) -> List[Tuple[int, int, str]]:
        """
        Identify watershed outlets from valley features and flow minima.
        
        Returns:
            List of (x, y, feature_id) for each outlet
        """
        outlets = []
        
        # Priority 1: Valley features (these are natural drainage outlets)
        valleys = [f for f in features if isinstance(f, ValleyFeature)]
        for valley in valleys:
            x, y = valley.centroid
            outlets.append((x, y, valley.feature_id))
        
        # Priority 2: Local flow minima (cells with no outflow)
        rows, cols = heightmap.shape
        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                if flow_dir[y, x] == 0:
                    # Check if it's a local minimum
                    current = heightmap.data[y, x]
                    is_min = True
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < rows and 0 <= nx < cols:
                                if heightmap.data[ny, nx] < current:
                                    is_min = False
                                    break
                        if not is_min:
                            break
                    
                    if is_min and len(outlets) < self._max_watershed_outlets:
                        outlets.append((x, y, f"outlet_{x}_{y}"))
        
        self._log(f"Identified {len(outlets)} watershed outlets")
        return outlets
    
    def _label_watersheds(self, heightmap: Heightmap,
                          flow_dir: np.ndarray,
                          outlets: List[Tuple[int, int, str]]) -> np.ndarray:
        """
        Label watersheds by propagating upstream from outlets.
        
        Returns:
            watershed_labels: 2D array with integer labels
        """
        rows, cols = heightmap.shape
        labels = np.zeros((rows, cols), dtype=np.int32)
        
        # Direction vectors for neighbor offsets
        neighbor_vectors = {
            1: (0, -1), 2: (1, -1), 3: (1, 0), 4: (1, 1),
            5: (0, 1), 6: (-1, 1), 7: (-1, 0), 8: (-1, -1)
        }
        
        # Reverse mapping: given a flow direction code, which neighbor codes flow into it?
        reverse_dir = {
            1: 5,   # North ← South
            2: 6,   # Northeast ← Southwest
            3: 7,   # East ← West
            4: 8,   # Southeast ← Northwest
            5: 1,   # South ← North
            6: 2,   # Southwest ← Northeast
            7: 3,   # West ← East
            8: 4    # Northwest ← Southeast
        }
        
        for outlet_id, (ox, oy, _) in enumerate(outlets, start=1):
            stack = [(ox, oy)]
            labels[oy, ox] = outlet_id
            
            while stack:
                cx, cy = stack.pop()
                current_dir = flow_dir[cy, cx]
                
                # Find all neighbors that flow into this cell
                for nb_code, (dx, dy) in neighbor_vectors.items():
                    nx, ny = cx + dx, cy + dy
                    
                    if 0 <= nx < cols and 0 <= ny < rows:
                        nb_dir = flow_dir[ny, nx]
                        
                        # Does this neighbor flow into current cell?
                        if nb_dir > 0:
                            # Get the direction from neighbor to current
                            to_current = reverse_dir.get(nb_dir, 0)
                            if to_current == current_dir or current_dir == 0:
                                if labels[ny, nx] == 0:
                                    labels[ny, nx] = outlet_id
                                    stack.append((nx, ny))
        
        return labels
    
    def _assign_features_to_watersheds(self, watershed_labels: np.ndarray,
                                        features: List[TerrainFeature],
                                        flow_network: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        """
        Assign features to watersheds using flow network.
        
        A feature belongs to the same watershed as its ultimate outlet.
        """
        # First, identify outlet features (features with no downstream)
        outlets = {f.feature_id for f in features if not flow_network.get(f.feature_id, [])}
        
        # Map each outlet to its watershed label from the pixel-based labeling
        outlet_basins = {}
        for f in features:
            if f.feature_id in outlets:
                x, y = f.centroid
                if 0 <= y < watershed_labels.shape[0] and 0 <= x < watershed_labels.shape[1]:
                    basin_id = watershed_labels[y, x]
                    if basin_id > 0:
                        outlet_basins[f.feature_id] = f"basin_{basin_id}"
        
        # If no outlets with valid basin labels, return empty
        if not outlet_basins:
            return {}
        
        # Propagate upstream: every feature flows to an outlet
        watersheds = {basin: set() for basin in outlet_basins.values()}
        
        # Build reverse flow network (downstream → upstream)
        upstream = {f.feature_id: [] for f in features}
        for fid, downstream_list in flow_network.items():
            for did in downstream_list:
                upstream[did].append(fid)
        
        # BFS/DFS from each outlet to collect all upstream features
        for outlet_id, basin_key in outlet_basins.items():
            stack = [outlet_id]
            visited = set()
            
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                watersheds[basin_key].add(current)
                
                # Add all features that flow into this one
                stack.extend(upstream.get(current, []))
        
        return watersheds
    
    def _filter_watersheds_by_area(self, watersheds: Dict[str, Set[str]],
                                    features: List[TerrainFeature],
                                    heightmap: Heightmap) -> Dict[str, Set[str]]:
        """
        Filter out watersheds smaller than minimum area.
        
        Returns:
            Filtered watersheds dictionary
        """
        if not watersheds:
            return watersheds
        
        filtered = {}
        cell_area_m2 = heightmap.config.horizontal_scale ** 2
        
        for basin_id, feature_ids in watersheds.items():
            # approximate area from number of features in basin
            area_estimate = len(feature_ids) * cell_area_m2 * self._watershed_area_estimate_factor
            
            if area_estimate >= self._watershed_min_area_px * cell_area_m2:
                filtered[basin_id] = feature_ids
        
        removed = len(watersheds) - len(filtered)
        if removed > 0:
            self._log(f"Filtered {removed} watersheds below min area")
        
        return filtered
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def _log_graph_statistics(self, visibility: Dict, flow: Dict, 
                               connectivity: Dict, watersheds: Dict) -> None:
        """Log graph statistics for debugging."""
        if not self.config.verbose:
            return
        
        vis_edges = sum(len(v) for v in visibility.values()) // 2
        vis_nodes = sum(1 for v in visibility.values() if v)
        self._log(f"Visibility: {vis_nodes} connected nodes, {vis_edges} edges")
        
        flow_edges = sum(len(v) for v in flow.values())
        flow_nodes = sum(1 for v in flow.values() if v)
        self._log(f"Flow Network: {flow_nodes} nodes with downstream, {flow_edges} edges")
        
        conn_edges = sum(len(v) for v in connectivity.values()) // 2
        conn_nodes = sum(1 for v in connectivity.values() if v)
        self._log(f"Connectivity: {conn_nodes} connected nodes, {conn_edges} edges")
        
        self._log(f"Watersheds: {len(watersheds)} basins")
        if watersheds:
            avg_size = np.mean([len(b) for b in watersheds.values()])
            self._log(f"Average basin size: {avg_size:.1f} features")
    
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
                                 "description": "per-pixel upstream area in pixels²"}
        }