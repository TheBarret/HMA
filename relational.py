"""
Layer 4: Relational Analysis
==================================================
Design:
Relations are NOT derived from feature heuristics.
Relations are derived from continuous mathematical fields, onto which features are mapped.

Internal steps:
1. Compute Pixel-Level Fields (Mathematics)
   - Flow Direction, Accumulation, Cost Surfaces
2. Extract Relational Structures (Topology)
   - Stream Networks, Watershed Boundaries, Visibility Lines
3. Map Discrete Features (Semantics)
   - Assign Features to Basins, Build Feature Graphs
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
    Builds relational graphs by mapping discrete features onto continuous relational fields.
    """

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        # configuration thresholds
        self._visibility_max_range_px = config.visibility_max_range_m / config.horizontal_scale
        self._connection_radius_px = config.connection_radius_m / config.horizontal_scale
        self._vehicle_climb_angle = config.vehicle_climb_angle
        self._cliff_threshold = config.cliff_threshold_degrees
        self._watershed_min_area_px = config.watershed_min_area_m2 / (config.horizontal_scale ** 2)
        # Recommended production values:
        # - 128x128 test maps: 50-100px
        # - 512x512 production: 200-500px
        # - 1024x1024+ large maps: 500-1000px
        self._stream_accumulation_threshold = config.stream_accumulation_threshold_px
        self._log(f"Config stream threshold: {config.stream_accumulation_threshold_px}px")

    def _log(self, msg: str) -> None:
        if self.config.verbose:
            print(f"[Relational] {msg}")

    def execute(self, input_data: LayerBundle) -> Dict[str, Any]:
        """
        ORCHESTRATOR: Executes the 3-phase relational analysis.
        
        LOGIC:
        1. We compute FIELDS (Flow, Cost) from Heightmap/Slope/Aspect.
        2. We extract STRUCTURES (Streams, Basins) from Fields.
        3. We map FEATURES (Layer 3 output) onto Structures to build GRAPHS.
        
        This ensures relations are grounded in geometry, not feature proximity heuristics.
        """
        features = input_data['features']
        heightmap = input_data['heightmap']
        slope = input_data.get('slope')
        aspect = input_data.get('aspect')
        curvature = input_data.get('curvature')
        
        self._log(f"Starting Relational Analysis | features={len(features)}")
                
        # =========================================================================
        # PHASE 1: PIXEL-LEVEL RELATIONAL FIELDS (Mathematics)
        # =========================================================================
        # Reasoning: Relations exist everywhere on the surface, not just at features.
        # We must compute the underlying physics (water flow, visibility, cost) first.
        
        self._log("Phase 1: Computing Relational Fields...")
        
        # A. Hydrological Field
        # Compute flow direction (D8 or DInf). Handle flat areas via priority-flood.
        # TODO:
        # USE CONFIG REF: self.config.flow_direction_method
        self._log(f"       - Compute flow direction...[{self.config.flow_direction_method}]")
        flow_direction = self._compute_flow_direction_field(heightmap, aspect)
        
        # Compute accumulation (upstream area). Required to define streams.
        self._log("       - Accumulation...")
        flow_accumulation = self._compute_flow_accumulation_field(flow_direction)
        
        # B. Traversability Field
        # Compute cost surface (slope + curvature). Required for pathfinding.
        self._log("       - Compute cost surface...")
        cost_surface = self._compute_traversability_cost_field(heightmap, slope, curvature)
        
               
        # =========================================================================
        # PHASE 2: RELATIONAL STRUCTURE EXTRACTION (Topology)
        # =========================================================================
        # Reasoning: Convert continuous fields into discrete relational objects.
        # A stream is where accumulation > threshold. A basin is where flow converges.
        
        self._log("Phase 2: Extracting Relational Structures...")
        
        # A. Stream Network (Pixel Graph)
        # Threshold accumulation to identify channel pixels. Connect them.
        stream_network_pixels = self._extract_stream_network(flow_accumulation)
        
        # B. Watershed Basins (Pixel Labels)
        # Identify outlets (true minima), label upstream areas
        watershed_labels, outlet_pixels = self._delineate_watershed_fields(flow_direction)
        
        # =========================================================================
        # PHASE 3: FEATURE-TO-STRUCTURE MAPPING (Semantics)
        # =========================================================================
        # Reasoning: Features are observers on the field. 
        # A peak belongs to a basin because its pixel lies in that basin label.
        # Two peaks are visible if the line between them clears the heightmap field.
        
        self._log("Phase 3: Mapping Features to Structures...")
        
        # A. Hydrological Graph (Feature -> Basin/Stream)
        # Assign each feature to a watershed based on its centroid's label.
        feature_watersheds = self._assign_features_to_basins(features, watershed_labels)
        
        # Build flow connections by tracing flow paths from feature pixels to outlets.
        feature_flow_network = self._build_feature_flow_graph(features, flow_direction, stream_network_pixels)
        
        # B. Visibility Graph (Feature -> Feature)
        # Check Line-of-Sight (LOS) using Bresenham without sampling skips.
        feature_visibility_graph = self._build_feature_visibility_graph(features, heightmap)
        
        # C. Connectivity Graph (Feature -> Feature)
        # Compute least-cost paths on the cost_surface between features.
        feature_connectivity_graph = self._build_feature_connectivity_graph(features, cost_surface)
        
        # =========================================================================
        # OUTPUT ASSEMBLY
        # =========================================================================
        # Return both the underlying fields (for Layer 5) and the graphs (for analysis).
        
        return {
            # Fields (for Semantic Layer 5)
            "flow_accumulation": flow_accumulation,
            "watershed_labels": watershed_labels,
            "cost_surface": cost_surface,
            
            # Graphs (for Relational Analysis)
            "visibility_graph": feature_visibility_graph,
            "flow_network": feature_flow_network,
            "connectivity_graph": feature_connectivity_graph,
            "watersheds": feature_watersheds,
            
            # Structures (Intermediate)
            "stream_network_pixels": stream_network_pixels,
            "outlet_pixels": outlet_pixels
        }

    # =========================================================================
    # PHASE 1: FIELD COMPUTATION 
    # =========================================================================

    def _encode_d8_direction(self, from_x: int, from_y: int, to_x: int, to_y: int) -> int:
        """
        Encode D8 direction code (1-8) from two coordinates.
        
        Coordinate system: x=column (right+), y=row (down+)
        """
        dx = to_x - from_x  # column change
        dy = to_y - from_y  # row change
        
        if dx == 1 and dy == 0:   return 1   # East
        if dx == 1 and dy == 1:   return 2   # Southeast
        if dx == 0 and dy == 1:   return 3   # South
        if dx == -1 and dy == 1:  return 4   # Southwest
        if dx == -1 and dy == 0:  return 5   # West
        if dx == -1 and dy == -1: return 6   # Northwest
        if dx == 0 and dy == -1:  return 7   # North
        if dx == 1 and dy == -1:  return 8   # Northeast
        return 0

    def _priority_flood_resolution(self, z: np.ndarray, flow_direction: np.ndarray, heightmap: Heightmap) -> np.ndarray:
        """
        Resolve flat areas using priority-flood algorithm.
        Propagates from boundary inward, ensuring water flows to edges.
        """
        h, w = z.shape
        resolved = flow_direction.copy()
        
        # D8 offsets for neighbor lookup
        OFFSETS = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1)
        ]
        
        import heapq
        pq = []
        visited = np.zeros(z.shape, dtype=bool)
        
        # Initialize: all boundary pixels enter the queue
        for y in [0, h-1]:
            for x in range(w):
                heapq.heappush(pq, (z[y, x], y, x))
                visited[y, x] = True
        
        for x in [0, w-1]:
            for y in range(1, h-1):
                heapq.heappush(pq, (z[y, x], y, x))
                visited[y, x] = True
        
        while pq:
            elev, y, x = heapq.heappop(pq)
            
            for dy, dx in OFFSETS:
                ny, nx = y + dy, x + dx
                
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if visited[ny, nx]:
                    continue
                
                visited[ny, nx] = True
                neighbor_z = z[ny, nx]
                
                # For flats (neighbor_z == elev), still assign direction
                if neighbor_z >= elev:
                    # Neighbor flows toward current cell
                    direction = self._encode_d8_direction(nx, ny, x, y)
                    resolved[ny, nx] = direction
                    heapq.heappush(pq, (neighbor_z, ny, nx))
                else:
                    # Neighbor is lower, already drains
                    heapq.heappush(pq, (neighbor_z, ny, nx))
        
        # Handle any remaining unassigned flat cells
        # For any cell still with direction 0, assign to nearest boundary via distance transform
        unassigned = np.where(resolved == 0)
        if len(unassigned[0]) > 0:
            from scipy.ndimage import distance_transform_edt
            boundary_mask = np.zeros_like(resolved, dtype=bool)
            boundary_mask[0, :] = boundary_mask[-1, :] = boundary_mask[:, 0] = boundary_mask[:, -1] = True
            distances, indices = distance_transform_edt(~boundary_mask, return_indices=True)
            
            for i in range(len(unassigned[0])):
                y, x = unassigned[0][i], unassigned[1][i]
                nearest_y, nearest_x = indices[0][y, x], indices[1][y, x]
                resolved[y, x] = self._encode_d8_direction(x, y, nearest_x, nearest_y)
        
        return resolved


    def _compute_flow_direction_field(self, heightmap: Heightmap, 
                                   aspect: Optional[np.ndarray]) -> np.ndarray:
        """
        COMPUTE: Flow Direction (D8 Algorithm + Priority-Flood)
        
        D8 Encoding (standard):
            64  128  1
            32   0   2
            16   8   4
        
        Simplified (1-8):
            7  8  1
            6  0  2
            5  4  3
        """
        z = heightmap.data
        h, w = z.shape
        cell_size = heightmap.config.horizontal_scale
        
        # D8 offsets: (dy, dx) where dy=row_change, dx=col_change
        # Direction codes: 1=E, 2=SE, 3=S, 4=SW, 5=W, 6=NW, 7=N, 8=NE
        D8_OFFSETS = [
            (0, 1),    # 1: East
            (1, 1),    # 2: Southeast
            (1, 0),    # 3: South
            (1, -1),   # 4: Southwest
            (0, -1),   # 5: West
            (-1, -1),  # 6: Northwest
            (-1, 0),   # 7: North
            (-1, 1),   # 8: Northeast
        ]
        
        _acc = 0
        flow_direction = np.zeros(z.shape, dtype=np.uint8)
        
        # Step 1: Compute raw D8 flow direction (steepest descent)
        for y in range(h):
            for x in range(w):
                center_z = z[y, x]
                max_drop = -np.inf
                best_dir = 0
                    
                for i, (dy, dx) in enumerate(D8_OFFSETS):
                    ny, nx = y + dy, x + dx
                    
                    # Boundary pixels flow off-map
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        if max_drop <= 0:
                            best_dir = i + 1
                            max_drop = 0.0
                        continue
                    
                    neighbor_z = z[ny, nx]
                    distance = np.sqrt(dx*dx + dy*dy) * cell_size
                    drop = (center_z - neighbor_z) / distance if distance > 0 else 0
                    
                    if drop > max_drop:
                        max_drop = drop
                        best_dir = i + 1
                
                flow_direction[y, x] = best_dir if max_drop > 0 else 0
        _acc += 1
        if _acc % 16 == 0:
                self._log(f"       -> y={y} of {h}")
                
        # Step 2: Priority-flood resolution for flat areas
        flow_direction = self._priority_flood_resolution(z, flow_direction, heightmap)
        
        outlets = np.sum(flow_direction == 0)
        self._log(f"Flow direction computed: {np.sum(flow_direction > 0)} pixels routed, {outlets} outlets")
        
        return flow_direction


    def _compute_flow_accumulation_field(self, flow_direction: np.ndarray) -> np.ndarray:
        """
        COMPUTE: Flow Accumulation via topological sort.
        FIX: Use indegree (upstream count), not outdegree (downstream flag).
        """
        from collections import deque
        
        h, w = flow_direction.shape
        accumulation = np.ones(flow_direction.shape, dtype=np.float32)  # Each cell counts itself
        
        D8_DECODE = {
            1: (0, 1), 2: (1, 1), 3: (1, 0), 4: (1, -1),
            5: (0, -1), 6: (-1, -1), 7: (-1, 0), 8: (-1, 1),
            0: (0, 0),
        }
        
        # Build reverse graph: which cells flow INTO each cell
        receivers = [[] for _ in range(h * w)]
        
        # Track indegree (count of upstream contributors), not outdegree
        indegree = np.zeros(flow_direction.shape, dtype=np.int32)
        
        for y in range(h):
            for x in range(w):
                code = flow_direction[y, x]
                if code == 0:
                    continue  # Outlet, no downstream
                
                dy, dx = D8_DECODE.get(code, (0, 0))
                ny, nx = y + dy, x + dx
                
                if 0 <= ny < h and 0 <= nx < w:
                    # Cell (y,x) flows INTO (ny,nx)
                    receivers[ny * w + nx].append((y, x))
                    indegree[ny, nx] += 1  # Count upstream contributors
        
        # Start with cells that have NO upstream contributors (sources)
        queue = deque()
        for y in range(h):
            for x in range(w):
                if indegree[y, x] == 0:  # Check indegree, not outdegree
                    queue.append((y, x))
        
        # Process in topological order (sources → sinks)
        while queue:
            y, x = queue.popleft()
            
            # Push this cell's accumulation to its DOWNSTREAM receiver
            code = flow_direction[y, x]
            if code != 0:
                dy, dx = D8_DECODE.get(code, (0, 0))
                ny, nx = y + dy, x + dx
                
                if 0 <= ny < h and 0 <= nx < w:
                    accumulation[ny, nx] += accumulation[y, x]  # Propagate accumulation
                    indegree[ny, nx] -= 1  # Decrement receiver's indegree
                    if indegree[ny, nx] == 0:  # Check indegree
                        queue.append((ny, nx))
        
        self._log(f"Flow accumulation computed: max={np.max(accumulation):.0f}px, mean={np.mean(accumulation):.1f}px")
        
        return accumulation

    def _compute_traversability_cost_field(self, heightmap: Heightmap, 
                                       slope: Optional[np.ndarray], 
                                       curvature: Optional[np.ndarray]) -> np.ndarray:
        """
        COMPUTE: Traversability Cost Surface
        
        Cost functions:
            - "tobler": Tobler's hiking function (time per meter)
            - "vehicle_quadratic": Quadratic penalty for slope (energy-based)
            - "custom": Placeholder for user-defined
        
        Cost = base_cost * (1 + slope_weight * slope_factor + curvature_weight * curvature_penalty + roughness_weight * roughness_penalty)
        """
        h, w = heightmap.shape
        cell_size = heightmap.config.horizontal_scale
        
        # Default: flat terrain cost (1.0 = easy)
        base_cost = np.ones((h, w), dtype=np.float32)
        
        if slope is None:
            self._log("Warning: slope not provided, cost surface defaults to uniform 1.0")
            return base_cost
        
        # Convert slope to radians for trig functions
        slope_rad = np.radians(slope)
        
        # =========================================================================
        # Apply selected cost function
        # =========================================================================
        
        if self.config.traversability_cost_function == "tobler":
            # Tobler's hiking function: speed = 6 * exp(-3.5 * |slope_rad + 0.05|)
            # Time = 1/speed (hours per km), normalized to seconds per meter
            speed_kmh = 6.0 * np.exp(-3.5 * np.abs(slope_rad + 0.05))
            speed_ms = speed_kmh / 3.6  # Convert km/h to m/s
            base_cost = 1.0 / np.maximum(speed_ms, 0.01)  # Time per meter (s/m)
            
        elif self.config.traversability_cost_function == "vehicle_quadratic":
            # Quadratic penalty: cost = 1 + (slope / max_slope)^2
            max_slope_rad = np.radians(self.config.vehicle_climb_angle)
            slope_ratio = np.clip(slope_rad / max_slope_rad, 0, 1)
            base_cost = 1.0 + slope_ratio ** 2
            
        elif self.config.traversability_cost_function == "custom":
            # Placeholder for custom implementation
            self._log("Custom cost function selected — using default slope-only cost")
            base_cost = 1.0 + (slope_rad / np.radians(self.config.vehicle_climb_angle))
            
        else:
            self._log(f"Unknown cost function: {self.config.traversability_cost_function}, using default")
            base_cost = 1.0 + (slope_rad / np.radians(self.config.vehicle_climb_angle))
        
        # =========================================================================
        # Apply curvature penalty (if available)
        # =========================================================================
        
        if curvature is not None and self.config.cost_curvature_weight > 0:
            # Positive curvature (convex) increases cost, negative (concave) decreases slightly
            curvature_penalty = 1.0 + self.config.cost_curvature_weight * np.maximum(curvature, 0)
            base_cost *= curvature_penalty
        
        # =========================================================================
        # Apply roughness penalty (if available)
        # =========================================================================
        
        if self.config.cost_roughness_weight > 0:
            # Roughness = local variance in elevation (could be passed from Layer 2)
            # For now, use simple 3x3 variance as placeholder
            from scipy.ndimage import generic_filter
            
            def local_variance(window):
                return np.var(window)
            
            roughness = generic_filter(heightmap.data, local_variance, size=3)
            roughness_norm = np.clip(roughness / 10.0, 0, 1)  # Normalize, assume 10m variance max
            base_cost *= (1.0 + self.config.cost_roughness_weight * roughness_norm)
        
        # =========================================================================
        # Normalize based on unit type
        # =========================================================================
        
        if self.config.traversability_unit == "time":
            # Already in seconds per meter, keep as-is
            pass
        elif self.config.traversability_unit == "energy":
            # Cost as energy expenditure (kJ per meter)
            # Approximate: 70kg human on slope
            mass_kg = 70.0
            g = 9.81
            energy_per_meter = mass_kg * g * np.sin(slope_rad)
            # Add base walking cost
            base_cost = energy_per_meter + 2.0  
        elif self.config.traversability_unit == "risk":
            # Cost as risk factor (1-10 scale)
            base_cost = np.clip(base_cost / 5.0, 1, 10)
        
        # Apply minimum cost floor
        base_cost = np.maximum(base_cost, 0.1)
        
        self._log(f"Cost surface computed: function={self.config.traversability_cost_function}, "
                  f"unit={self.config.traversability_unit}, "
                  f"range=[{base_cost.min():.2f}, {base_cost.max():.2f}], "
                  f"mean={base_cost.mean():.2f}")
        
        return base_cost.astype(np.float32)

    # =========================================================================
    # PHASE 2: STRUCTURE EXTRACTION 
    # =========================================================================

    def _extract_stream_network(self, flow_accumulation: np.ndarray) -> np.ndarray:
        """Extract stream network from accumulation threshold."""
        # Centralize skimage imports to handle potential ImportError once
        try:
            from skimage.morphology import skeletonize, remove_small_objects
            has_skimage = True
        except ImportError:
            self._log("Warning: skimage not available, using threshold only (no thinning/cleanup)")
            has_skimage = False

        threshold = self._stream_accumulation_threshold
        stream_mask = flow_accumulation >= threshold
        
        self._log(f"Stream network: {np.sum(stream_mask)} pixels ({100*np.sum(stream_mask)/stream_mask.size:.2f}%) above threshold={threshold}px")
        
        if np.sum(stream_mask) == 0:
            self._log("Warning: No streams detected!")
            return stream_mask
        
        # Only apply advanced morphology if skimage is present
        if has_skimage:
            stream_mask = skeletonize(stream_mask)
            # Ensure min_size is an integer; 10 is usually a good starting point
            stream_mask = remove_small_objects(stream_mask, min_size=self.config.skimage_stream_min_size_px)
        
        return stream_mask

    def _delineate_watershed_fields(self, flow_direction: np.ndarray) -> Tuple[np.ndarray, List[PixelCoord]]:
        """Delineate watersheds by propagating labels upstream from outlets."""
        from collections import deque
        from scipy.ndimage import generic_filter
        
        h, w = flow_direction.shape
        
        D8_DECODE = {
            1: (0, 1), 2: (1, 1), 3: (1, 0), 4: (1, -1),
            5: (0, -1), 6: (-1, -1), 7: (-1, 0), 8: (-1, 1),
            0: (0, 0),
        }
        
        # Step 1: Identify outlets
        outlets = []
        outlet_id_map = {}
        
        for y in range(h):
            for x in range(w):
                code = flow_direction[y, x]
                is_outlet = False
                
                if code == 0:
                    is_outlet = True
                else:
                    dy, dx = D8_DECODE.get(code, (0, 0))
                    ny, nx = y + dy, x + dx
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        is_outlet = True
                
                if is_outlet:
                    outlet_id = len(outlets)
                    outlets.append((x, y))
                    outlet_id_map[(y, x)] = outlet_id
        
        self._log(f"Watershed outlets identified: {len(outlets)}")
        
        # Step 2: Initialize label map
        watershed_labels = np.full(flow_direction.shape, -1, dtype=np.int32)
        
        for (y, x), outlet_id in outlet_id_map.items():
            watershed_labels[y, x] = outlet_id
        
        # Step 3: Build reverse graph
        receivers = [[] for _ in range(h * w)]
        
        for y in range(h):
            for x in range(w):
                code = flow_direction[y, x]
                if code == 0:
                    continue
                
                dy, dx = D8_DECODE.get(code, (0, 0))
                ny, nx = y + dy, x + dx
                
                if 0 <= ny < h and 0 <= nx < w:
                    receivers[ny * w + nx].append((y, x))
        
        # Step 4: BFS from outlets upstream
        queue = deque(outlet_id_map.keys())
        
        while queue:
            y, x = queue.popleft()
            current_label = watershed_labels[y, x]
            
            if current_label == -1:
                continue
            
            for upstream_y, upstream_x in receivers[y * w + x]:
                if watershed_labels[upstream_y, upstream_x] == -1:
                    watershed_labels[upstream_y, upstream_x] = current_label
                    queue.append((upstream_y, upstream_x))
        
        # After BFS, handle unlabeled pixels
        unlabeled = np.where(watershed_labels == -1)
        if len(unlabeled[0]) > 0:
            # Use nearest labeled neighbor
            from scipy.ndimage import distance_transform_edt
            labeled_mask = watershed_labels >= 0
            distances, indices = distance_transform_edt(~labeled_mask, return_indices=True)
            watershed_labels[unlabeled] = watershed_labels[tuple(indices[:, unlabeled[0], unlabeled[1]])]
        
        return watershed_labels, outlets

    # =========================================================================
    # PHASE 3: FEATURE MAPPING 
    # =========================================================================

    def _assign_features_to_basins(self, features: List[TerrainFeature], 
                               watershed_labels: np.ndarray) -> Dict[str, Set[str]]:
        """
        MAP: Features → Watersheds
        
        Config uses:
            - watershed_min_area_m2: Minimum basin area in square meters
            - include_edge_basins: Include basins that drain off-map
            - horizontal_scale: For area calculation
        """
        from collections import defaultdict
        h, w = watershed_labels.shape
        cell_area_m2 = self.config.horizontal_scale ** 2
        min_area_px = self._watershed_min_area_px  # Already converted in __init__
        
        # Step 1: Calculate area for each basin (pixel count)
        basin_ids, basin_counts = np.unique(watershed_labels, return_counts=True)
        
        # Filter by minimum area
        valid_basins = {}
        basin_areas_m2 = {}
        
        # debug trackers
        _acc = 0
        _volume = 0
        
        for bid, count in zip(basin_ids, basin_counts):
            # Skip invalid basins (negative labels)
            if bid < 0:
                continue
            
            area_m2 = count * cell_area_m2
            
            # Apply area filter
            if area_m2 >= self.config.watershed_min_area_m2:
                basin_key = f"basin_{bid}"
                valid_basins[bid] = basin_key
                basin_areas_m2[basin_key] = area_m2
                _volume += area_m2
                _acc += 1
        
        if len(valid_basins) == 0:
            self._log(f"Warning: No basins meet minimum area {self.config.watershed_min_area_m2:.0f} m²")
            return {}
            
        if self.config.verbose:
            self._log(f"Basins: {_acc} found with a total of {_volume}m²")
        
        # Step 2: Assign features to basins
        feature_watersheds = {basin_key: set() for basin_key in valid_basins.values()}
        
        for feature in features:
            x, y = feature.centroid
            
            # Boundary check
            if not (0 <= x < w and 0 <= y < h):
                continue
            
            basin_id = watershed_labels[y, x]
            
            if basin_id in valid_basins:
                basin_key = valid_basins[basin_id]
                feature_watersheds[basin_key].add(feature.feature_id)
                
                # Update feature attributes
                if hasattr(feature, 'watershed_id'):
                    feature.watershed_id = basin_key
                elif hasattr(feature, 'watershed_ids'):
                    if basin_key not in feature.watershed_ids:
                        feature.watershed_ids.append(basin_key)
                elif hasattr(feature, 'adjacent_watersheds'):
                    # For ridges
                    if basin_key not in feature.adjacent_watersheds:
                        feature.adjacent_watersheds.append(basin_key)
        
        # Step 3: Log results
        total_assigned = sum(len(features) for features in feature_watersheds.values())
        self._log(f"Assigned {total_assigned} features to {len(feature_watersheds)} basins")
        
        # Step 4: Filter out empty basins
        filtered_watersheds = {k: v for k, v in feature_watersheds.items() if v}
        
        return filtered_watersheds

    def _build_feature_flow_graph(self, features: List[TerrainFeature], 
                              flow_direction: np.ndarray,
                              stream_network: np.ndarray) -> Dict[str, List[str]]:
        """
        MAP: Features → Flow Network
        
        Traces flow paths from each feature centroid downstream.
        Connects features when one flows into another's catchment.
        
        Config uses:
            - feature_snap_distance_px: Max distance to snap to stream
            - snap_method: "nearest", "flow_directed", or "none"
            - include_edge_basins: Include outlets at map edges
        """
        h, w = flow_direction.shape
        flow_network = {f.feature_id: [] for f in features}
        
        # D8 decode for downstream movement
        D8_DECODE = {
            1: (0, 1), 2: (1, 1), 3: (1, 0), 4: (1, -1),
            5: (0, -1), 6: (-1, -1), 7: (-1, 0), 8: (-1, 1),
            0: (0, 0),
        }
        
        # Build spatial index for features
        feature_coords = [(f.centroid[0], f.centroid[1]) for f in features]
        feature_ids = [f.feature_id for f in features]
        feature_tree = KDTree(feature_coords) if features else None
        
        # Pre-compute stream network for faster lookup
        is_stream = stream_network.astype(bool)
        
        self._log(f"Building flow network: {len(features)} features, snap_distance={self.config.feature_snap_distance_px}px")
        
        for feature in features:
            x, y = feature.centroid
            
            # Start tracing from centroid
            trace_path = [(x, y)]
            current_x, current_y = x, y
            
            # Trace until we hit: boundary, stream, or another feature
            while True:
                # Check boundaries
                if not (0 <= current_x < w and 0 <= current_y < h):
                    break
                
                # Check if we hit a stream pixel
                if is_stream[current_y, current_x]:
                    # Snap to nearest feature if within distance
                    if feature_tree and self.config.snap_method != "none":
                        distances, indices = feature_tree.query([(current_x, current_y)])
                        if distances[0] <= self.config.feature_snap_distance_px:
                            downstream_id = feature_ids[indices[0]]
                            if downstream_id != feature.feature_id:
                                flow_network[feature.feature_id].append(downstream_id)
                            break
                    break
                
                # Get flow direction
                code = flow_direction[current_y, current_x]
                if code == 0:  # Sink or outlet
                    break
                
                dy, dx = D8_DECODE.get(code, (0, 0))
                next_x, next_y = current_x + dx, current_y + dy
                
                
                # Check if we've entered a cycle,
                # the issue is that for long flow paths (common in large basins), this becomes O(n²)
                
                # proposed:
                #trace_path_set = {(x, y)}  # Use set for O(1) lookup
                #if (next_x, next_y) in trace_path_set:
                #    break
                #trace_path_set.add((next_x, next_y))
                
                # This works too but needs an evaluation
                if (next_x, next_y) in trace_path:
                    self._log(f"Cycle detected in flow path for feature {feature.feature_id}")
                    break
                
                trace_path.append((next_x, next_y))
                current_x, current_y = next_x, next_y
                
                # Safety limit
                if len(trace_path) > h * w:
                    self._log(f"Flow path exceeded max length for feature {feature.feature_id}")
                    break
        
        # Log statistics
        edges = sum(len(v) for v in flow_network.values())
        connected_features = sum(1 for v in flow_network.values() if v)
        self._log(f"Flow network: {connected_features} features have downstream connections, {edges} total edges")
        
        return flow_network

    def _check_line_of_sight(self, x1: int, y1: int, z1: float,
                     x2: int, y2: int, z2: float,
                     z: np.ndarray) -> bool:
        """
        Bresenham line-of-sight check.
        
        Args:
            x1, y1: Start pixel coordinates
            z1: Start height (including observer offset)
            x2, y2: End pixel coordinates
            z2: End height (including target offset)
            z: Heightmap data
        
        Returns:
            True if line-of-sight is clear, False otherwise
        """
        rows, cols = z.shape
        
        # Boundary check
        if not (0 <= x1 < cols and 0 <= y1 < rows and 
                0 <= x2 < cols and 0 <= y2 < rows):
            return False
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        # Total Euclidean distance in pixels
        total_distance_px = np.hypot(dx, dy)
        
        x, y = x1, y1
        
        while (x, y) != (x2, y2):
            if (x, y) != (x1, y1):
                # Linear interpolation of line-of-sight height
                t = np.hypot(x - x1, y - y1) / max(total_distance_px, 1.0)
                line_z = z1 + t * (z2 - z1)
                terrain_z = z[y, x]
                
                # Check if terrain blocks view
                if terrain_z > line_z + self.config.visibility_epsilon_m:
                    return False
            
            # Bresenham step
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return True

    def _build_feature_visibility_graph(self, features: List[TerrainFeature], 
                                    heightmap: Heightmap) -> Dict[str, Set[str]]:
        """
        MAP: Features → Visibility Graph
        
        Config uses:
            - visibility_max_range_m: Maximum LOS distance
            - visibility_epsilon_m: LOS precision threshold
            - visibility_observer_height_m: Observer height offset
            - visibility_target_height_m: Target height offset
            - horizontal_scale: For pixel-to-meter conversion
            - verbose: For logging
        """
        visibility = {f.feature_id: set() for f in features}
        
        if len(features) < 2:
            return visibility
        
        z = heightmap.data
        rows, cols = z.shape
        cell_size = heightmap.config.horizontal_scale
        
        # Optional: Only consider dominant features for visibility (configurable)
        # Using peaks and ridges as primary visibility points
        important_features = [f for f in features if isinstance(f, (PeakFeature, RidgeFeature, SaddleFeature))]
        
        if len(important_features) < 2:
            self._log("Insufficient important features for visibility graph")
            return visibility
        
        self._log(f"Building visibility graph: {len(important_features)} features, "
                  f"max_range={self._visibility_max_range_px:.1f}px ({self.config.visibility_max_range_m:.0f}m)")
        
        # Pre-compute heights with observer/target offsets
        feature_heights = {}
        for f in features:
            x, y = f.centroid
            if 0 <= x < cols and 0 <= y < rows:
                base_height = z[y, x]
                # Different features might have different observation heights
                if isinstance(f, PeakFeature):
                    # Peaks: observer at top, targets at ground
                    observer_height = self.config.visibility_observer_height_m
                    target_height = self.config.visibility_target_height_m
                elif isinstance(f, RidgeFeature):
                    # Ridges: slightly lower observation
                    observer_height = self.config.visibility_observer_height_m * self.config.ridge_visibility_scale
                    target_height = self.config.visibility_target_height_m
                else:
                    observer_height = self.config.visibility_observer_height_m
                    target_height = self.config.visibility_target_height_m
                
                feature_heights[f.feature_id] = {
                    'observer': base_height + observer_height,
                    'target': base_height + target_height,
                    'x': x,
                    'y': y
                }
        
        # Check visibility between all important feature pairs
        n = len(important_features)
        checked = 0
        visible_pairs = 0
        
        for i in range(n):
            f1 = important_features[i]
            f1_data = feature_heights.get(f1.feature_id)
            if not f1_data:
                continue
            
            for j in range(i + 1, n):
                f2 = important_features[j]
                f2_data = feature_heights.get(f2.feature_id)
                if not f2_data:
                    continue
                
                # Fast distance rejection
                dx = (f1_data['x'] - f2_data['x']) * cell_size
                dy = (f1_data['y'] - f2_data['y']) * cell_size
                distance_m = np.sqrt(dx*dx + dy*dy)
                
                if distance_m > self.config.visibility_max_range_m:
                    continue
                
                checked += 1
                
                # Check visibility: from f1 observer to f2 target
                visible_1_to_2 = self._check_line_of_sight(
                    f1_data['x'], f1_data['y'], f1_data['observer'],
                    f2_data['x'], f2_data['y'], f2_data['target'],
                    z
                )
                
                # Problematic: assumes symmetry
                if visible_1_to_2:
                    visibility[f1.feature_id].add(f2.feature_id)
                    visibility[f2.feature_id].add(f1.feature_id)
                    visible_pairs += 1
                
                # Suggestion: do a check on both ends, only proceed when true
                #visible_1_to_2 = self._check_line_of_sight(f1_obs, f2_tgt, z)
                #visible_2_to_1 = self._check_line_of_sight(f2_obs, f1_tgt, z)
                #if visible_1_to_2:
                #    visibility[f1.feature_id].add(f2.feature_id)
                #if visible_2_to_1:
                #    visibility[f2.feature_id].add(f1.feature_id)
                
                if self.config.verbose and checked % 1024 == 1:
                    self._log(f"Visibility checking, verified={checked}, visible_pairs={visible_pairs}")
        
        self._log(f"Visibility graph complete: {checked} checks, {visible_pairs} visible pairs")
        
        return visibility

    def _build_feature_connectivity_graph(self, features: List[TerrainFeature], 
                                       cost_surface: np.ndarray) -> Dict[str, Set[str]]:
        import heapq
        from scipy.spatial import KDTree
        
        h, w = cost_surface.shape
        connectivity = {f.feature_id: set() for f in features}
        
        if len(features) < 2:
            return connectivity
        
        # Precompute global stats once
        min_cost = np.min(cost_surface)
        median_cost = np.median(cost_surface)
        
        # Build spatial index
        feature_coords = np.array([(f.centroid[0], f.centroid[1]) for f in features], dtype=np.int32)
        feature_ids = [f.feature_id for f in features]
        feature_tree = KDTree(feature_coords)
        
        # Precompute ALL neighbor relationships (one query instead of N)
        radius_px = self._connection_radius_px
        all_neighbors = feature_tree.query_ball_point(feature_coords, r=radius_px)
        
        # Map coordinates to feature index for O(1) lookup during Dijkstra
        coord_to_feature = {}
        for idx, (x, y) in enumerate(feature_coords):
            coord_to_feature[(x, y)] = idx
        
        max_cost = self.config.connectivity_max_cost
        heuristic_buffer = self.config.connectivity_heuristic_buffer
        
        # 8-direction neighbors
        NEIGHBORS = [
            (0, 1, 1.0), (1, 1, 1.414), (1, 0, 1.0), (1, -1, 1.414),
            (0, -1, 1.0), (-1, -1, 1.414), (-1, 0, 1.0), (-1, 1, 1.414),
        ]
        
        # Setup version-stamped g_scores (allocated once)
        if not hasattr(self, '_g_scores') or self._g_scores.shape != (h, w):
            self._g_scores = np.full((h, w), np.inf, dtype=np.float32)
            self._g_version = np.zeros((h, w), dtype=np.int32)
            self._current_version = 0
        
        self._log(f"Building connectivity graph: {len(features)} features, "
                  f"radius={radius_px:.1f}px, max_cost={max_cost:.1f}, buffer={heuristic_buffer}")
        
        total_connections = 0
        total_pairs_considered = 0
        
        for i, source_feature in enumerate(features):
            sx, sy = source_feature.centroid
            source_id = source_feature.feature_id
            
            # Use precomputed neighbors, filter to j > i
            candidate_indices = [j for j in all_neighbors[i] if j > i]
            
            if not candidate_indices:
                continue
            
            # Build target info for this source
            target_coords = set()
            target_info = {}
            
            for j in candidate_indices:
                x, y = feature_coords[j]
                target_coords.add((x, y))
                target_info[(x, y)] = (feature_ids[j], j)
            
            # Heuristic filter with configurable buffer
            filtered_targets = set()
            for x, y in target_coords:
                euclidean_dist = np.hypot(x - sx, y - sy)
                estimated_cost = euclidean_dist * median_cost * heuristic_buffer
                
                if estimated_cost <= max_cost:
                    filtered_targets.add((x, y))
            
            if not filtered_targets:
                continue
            
            total_pairs_considered += len(filtered_targets)
            
            # Multi-target Dijkstra
            # Possible issue: If Layer 4 instance runs multiple times in same session, 
            # version tracking works correctly. However, if you create multiple instances of the layer, 
            # each has its own _current_version counter starting at 0.
            # Only matters if you instantiate multiple Layer4 objects and reuse them. Single-instance pipeline is safe.
            
            self._current_version += 1
            version = self._current_version
            
            # Initialize source
            self._g_scores[sy, sx] = 0.0
            self._g_version[sy, sx] = version
            pq = [(0.0, sx, sy)]
            
            remaining_targets = set(filtered_targets)
            settled_costs = {}
            
            if i % 16 == 0:
                self._log(f"    #{i} [{source_id}] pairs={total_pairs_considered}")
            
            while pq and remaining_targets:
                cost, x, y = heapq.heappop(pq)
                
                # Version check - skip if stale
                if self._g_version[y, x] != version:
                    continue
                if cost > self._g_scores[y, x]:
                    continue
                if cost > max_cost:
                    continue
                
                # Check if this is a target (settle it, but keep expanding)
                if (x, y) in remaining_targets:
                    settled_costs[(x, y)] = cost
                    remaining_targets.discard((x, y))
                    # continue neighbor expansion
                
                # Explore neighbors
                for dy, dx, move_cost in NEIGHBORS:
                    nx, ny = x + dx, y + dy
                    
                    if not (0 <= nx < w and 0 <= ny < h):
                        continue
                    
                    step_cost = cost_surface[ny, nx] * move_cost
                    new_cost = cost + step_cost
                    
                    if new_cost > max_cost:
                        continue
                    
                    # Version-aware update
                    if (self._g_version[ny, nx] != version or 
                        new_cost < self._g_scores[ny, nx]):
                        self._g_scores[ny, nx] = new_cost
                        self._g_version[ny, nx] = version
                        heapq.heappush(pq, (new_cost, nx, ny))
            
            # Record connections
            for (tx, ty), path_cost in settled_costs.items():
                if path_cost <= max_cost:
                    target_id, _ = target_info[(tx, ty)]
                    connectivity[source_id].add(target_id)
                    connectivity[target_id].add(source_id)
                    total_connections += 1
              
        edges = sum(len(v) for v in connectivity.values()) // 2
        connected_features = sum(1 for v in connectivity.values() if v)
        
        self._log(f"Connectivity graph complete: {connected_features}/{len(features)} features connected, "
                  f"{edges} edges, {total_pairs_considered} pairs evaluated")
        
        return connectivity
    
    #     
    # HELPERS
    #
    
    def _astar_path_cost(self, cost_surface: np.ndarray, 
                     start: PixelCoord, 
                     goal: PixelCoord,
                     max_cost: float = float('inf')) -> Optional[float]:
        """
        A* pathfinding with Euclidean distance heuristic.
        
        OPTIMIZATIONS vs Dijkstra:
        1. Heuristic guides search toward goal (fewer explored pixels)
        2. Early termination when f_score > max_cost
        3. Bounded search: don't explore pixels beyond reasonable detour
        
        Returns total cost or None if unreachable/too expensive.
        """
        import heapq
        
        h, w = cost_surface.shape
        sx, sy = start
        gx, gy = goal
        
        # Quick check: same cell
        if (sx, sy) == (gx, gy):
            return 0.0
        
        # Quick check: out of bounds
        if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
            return None
        
        # 8-direction neighbors with movement costs
        NEIGHBORS = [
            (0, 1, 1.0),    # E
            (1, 1, 1.414),  # SE (√2)
            (1, 0, 1.0),    # S
            (1, -1, 1.414), # SW
            (0, -1, 1.0),   # W
            (-1, -1, 1.414),# NW
            (-1, 0, 1.0),   # N
            (-1, 1, 1.414), # NE
        ]
        
        # Heuristic: Euclidean distance * minimum possible cost
        min_cost = np.min(cost_surface)
        def heuristic(x, y):
            return np.hypot(x - gx, y - gy) * min_cost
        
        # Priority queue: (f_score, g_score, x, y)
        # f = g + h (total estimated cost)
        # g = actual cost so far
        initial_h = heuristic(sx, sy)
        pq = [(initial_h, 0.0, sx, sy)]
        
        # Track best g_score for each pixel
        g_scores = np.full((h, w), np.inf, dtype=np.float32)
        g_scores[sy, sx] = 0.0
        
        # Bounding box with buffer (don't search entire grid)
        
        # Change:
        # min_cost from Tobler on flat terrain is ~0.71 (from your debug log: range=[0.71, 120.35]).
        # With max_cost=1500, that gives search_radius ≈ 2112 — larger than any reasonable map. 
        
        search_radius = int(max_cost / min_cost) if min_cost > 0 else max(h, w)
        y_min, y_max = max(0, sy - search_radius), min(h, sy + search_radius)
        x_min, x_max = max(0, sx - search_radius), min(w, sx + search_radius)
        
        visited_count = 0
        max_visited = (x_max - x_min) * (y_max - y_min) * self.config.connectivity_max_visited_ratio
        
        while pq:
            f, g, x, y = heapq.heappop(pq)
            
            # Early termination: already found better path to this pixel
            if g > g_scores[y, x]:
                continue
            
            # Early termination: exceeded max cost
            if g > max_cost:
                continue
            
            # Early termination: reached goal
            if (x, y) == (gx, gy):
                return g
            
            # Safety limit: don't explore too many pixels
            visited_count += 1
            if visited_count > max_visited:
                return None
            
            # Explore neighbors
            for dy, dx, move_cost in NEIGHBORS:
                nx, ny = x + dx, y + dy
                
                # Bounds check (use pre-computed bounding box)
                if nx < x_min or nx >= x_max or ny < y_min or ny >= y_max:
                    continue
                
                # Cost to move to neighbor
                step_cost = cost_surface[ny, nx] * move_cost
                new_g = g + step_cost
                
                # Prune if already exceeds max_cost
                if new_g > max_cost:
                    continue
                
                # Update if better path found
                if new_g < g_scores[ny, nx]:
                    g_scores[ny, nx] = new_g
                    h = heuristic(nx, ny)
                    f = new_g + h
                    heapq.heappush(pq, (f, new_g, nx, ny))
        
        # No path found within max_cost
        return None
    
    def _dijkstra_path_cost(self, cost_surface: np.ndarray, 
                            start: PixelCoord, 
                            goal: PixelCoord,
                            max_cost: float = float('inf')) -> Optional[float]:
        """
        Compute least-cost path cost using Dijkstra's algorithm.
        
        Args:
            cost_surface: 2D array of traversal costs per pixel
            start: (x, y) start pixel
            goal: (x, y) goal pixel
            max_cost: Early termination threshold — if accumulated cost exceeds this,
                      stop exploring this path (pruning)
        
        Returns:
            Total accumulated cost, or None if unreachable or cost > max_cost
        """
        import heapq
        
        h, w = cost_surface.shape
        sx, sy = start
        gx, gy = goal
        
        # Quick check: same cell
        if (sx, sy) == (gx, gy):
            return 0.0
        
        # 8-direction neighbors
        NEIGHBORS = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1)
        ]
        
        def move_cost(dx, dy):
            """Euclidean distance for diagonal moves, 1.0 for cardinal."""
            return np.sqrt(2.0) if dx != 0 and dy != 0 else 1.0
        
        # Priority queue: (accumulated_cost, x, y, heuristic)
        pq = [(0.0, sx, sy)]
        visited = np.full((h, w), False, dtype=bool)
        dist = np.full((h, w), float('inf'))
        dist[sy, sx] = 0.0
        
        while pq:
            current_cost, x, y = heapq.heappop(pq)
            
            if visited[y, x]:
                continue
            
            # Early termination if we've already exceeded max_cost
            if current_cost > max_cost:
                continue
            
            visited[y, x] = True
            
            # Early exit if we reached the goal
            if (x, y) == (gx, gy):
                return current_cost
            
            # Explore neighbors
            for dx, dy in NEIGHBORS:
                nx, ny = x + dx, y + dy
                
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                
                if visited[ny, nx]:
                    continue
                
                # Cost to move to neighbor = cost at target * distance
                step_cost = cost_surface[ny, nx] * move_cost(dx, dy)
                new_cost = current_cost + step_cost
                
                # Prune if new_cost already exceeds max_cost
                if new_cost > max_cost:
                    continue
                
                if new_cost < dist[ny, nx]:
                    dist[ny, nx] = new_cost
                    heapq.heappush(pq, (new_cost, nx, ny))
        
        # No path found within max_cost
        return None

    @property
    def output_schema(self) -> dict:
        return {
            "visibility_graph": { "type": "Dict[str, Set[str]]", "description": "Feature visibility relations" },
            "flow_network": { "type": "Dict[str, List[str]]", "description": "Hydrological feature connections" },
            "connectivity_graph": { "type": "Dict[str, Set[str]]", "description": "Traversability connections" },
            "watersheds": { "type": "Dict[str, Set[str]]", "description": "Basin membership by feature" },
            "flow_accumulation": { "type": "ScalarField", "description": "Pixel-level upstream area" },
            "watershed_labels": { "type": "ScalarField", "description": "Pixel-level basin IDs" },
            "cost_surface": { "type": "ScalarField", "description": "Pixel-level traversal cost" }
        }