"""
Layer 5: Semantic Interpretation

Assembles all prior outputs into the final AnalyzedTerrain object.
Applies domain thresholds and classification rules to produce
tactical terrain categories for combat games (Arma, WoT, War Thunder).
"""

import numpy as np
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

from core import (
    Heightmap, ScalarField, PipelineConfig, PipelineLayer,
    TerrainFeature, ClassifiedFeature,
    PeakFeature, RidgeFeature, ValleyFeature, SaddleFeature, FlatZoneFeature,
    AnalyzedTerrain, LayerBundle, PixelCoord, WorldCoord, Template
)


class Layer5_Semantics(PipelineLayer[AnalyzedTerrain]):
    """
    Assemble all prior outputs into the final AnalyzedTerrain object.
    
    Applies game-specific tactical classification:
    - Defensive positions (peaks with good visibility and moderate slopes)
    - Observation posts (high prominence peaks with wide visibility)
    - Chokepoints (saddles that connect multiple regions)
    - Assembly areas (large flat zones near objectives)
    - Cover positions (wide ridges that block LOS)
    - Vehicle routes (traversable corridors)
    """
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.baseline = config.baseline
        
        # internal storage
        self.max_feature_area: float = 0.0
        self.total_area_m2: float = 0.0
        
        # Template thresholds
        self._log(f'using template: {self.baseline}')
        self._template_thresholds = self._get_template_thresholds()
        
        # Coverage filtering
        self._max_feature_coverage = self.config.max_feature_coverage
        self._log(f'max n-% feature coverage: {self._max_feature_coverage}')

    def _log(self, msg: str) -> None:
        """Helper for verbose logging"""
        if self.config.verbose:
            print(f"[Semantics] {msg}")
            
    def _get_template_thresholds(self) -> Dict[str, Any]:
        """Get game-specific tactical thresholds."""
        thresholds = {
            Template.ARMA_3: {
                "defensive_min_prominence_m": 8.0,
                "defensive_min_elevation_m": 5.0,
                "defensive_max_slope_deg": 25.0,
                "defensive_min_visibility": 5,
                "observation_min_prominence_m": 10.0,
                "observation_min_visibility": 10,
                "assembly_min_area_m2": 2000,
                "assembly_max_slope_deg": 5.0,
                "chokepoint_min_connectivity": 2,
                "cover_min_width_m": 5.0,
                "vehicle_route_max_slope_deg": 25.0,
            },
            Template.WORLD_OF_TANKS: {
                "defensive_min_prominence_m": 5.0,
                "defensive_min_elevation_m": 3.0,
                "defensive_max_slope_deg": 20.0,
                "defensive_min_visibility": 3,
                "observation_min_prominence_m": 8.0,
                "observation_min_visibility": 8,
                "assembly_min_area_m2": 1000,
                "assembly_max_slope_deg": 3.0,
                "chokepoint_min_connectivity": 2,
                "cover_min_width_m": 4.0,
                "vehicle_route_max_slope_deg": 20.0,
            },
            Template.WAR_THUNDER: {
                "defensive_min_prominence_m": 6.0,
                "defensive_min_elevation_m": 4.0,
                "defensive_max_slope_deg": 25.0,
                "defensive_min_visibility": 5,
                "observation_min_prominence_m": 10.0,
                "observation_min_visibility": 12,
                "assembly_min_area_m2": 1500,
                "assembly_max_slope_deg": 5.0,
                "chokepoint_min_connectivity": 2,
                "cover_min_width_m": 5.0,
                "vehicle_route_max_slope_deg": 25.0,
            },
            Template.ARMA_2: {
                "defensive_min_prominence_m": 10.0,
                "defensive_min_elevation_m": 6.0,
                "defensive_max_slope_deg": 25.0,
                "defensive_min_visibility": 5,
                "observation_min_prominence_m": 12.0,
                "observation_min_visibility": 8,
                "assembly_min_area_m2": 2500,
                "assembly_max_slope_deg": 5.0,
                "chokepoint_min_connectivity": 2,
                "cover_min_width_m": 6.0,
                "vehicle_route_max_slope_deg": 25.0,
            },
        }
        return thresholds.get(self.baseline, thresholds[Template.ARMA_3])
    
    def execute(self, input_data: LayerBundle) -> AnalyzedTerrain:
        """
        Assemble final AnalyzedTerrain with tactical semantic tags.
        
        Args:
            input_data: LayerBundle containing:
                - features: List[TerrainFeature] from Layer 3
                - visibility_graph: Dict[str, Set[str]] from Layer 4
                - connectivity_graph: Dict[str, Set[str]] from Layer 4
                - flow_network: Dict[str, List[str]] from Layer 4
                - watersheds: Dict[str, Set[str]] from Layer 4
                - flow_accumulation: ScalarField from Layer 4
                - heightmap: Heightmap from Layer 0
                - slope: Slope field from Layer 1
                
        Returns:
            AnalyzedTerrain: Fully populated terrain analysis with semantic tags
        """
        
        # Extract from bundle
        self._log('analysing data bundle...')
        features = input_data.get('features', [])
        if not features:
            self._log("No features found. critical for semantic index")
            return AnalyzedTerrain(
                source_heightmap=input_data['heightmap'],
                peaks=[], valleys=[], ridges=[], saddles=[], flat_zones=[],
                visibility_graph={}, flow_network={}, connectivity_graph={},
                watersheds={}, flow_accumulation=None,
                semantic_index=self._build_empty_semantic_index()
            )
        
        visibility = input_data.get('visibility_graph', {})
        flow_network = input_data.get('flow_network', {})
        connectivity = input_data.get('connectivity_graph', {})
        watersheds = input_data.get('watersheds', {})
        flow_accum = input_data.get('flow_accumulation', None)
        heightmap = input_data.get('heightmap')
        slope = input_data.get('slope', None)
        
        if heightmap is None:
            self._log("No heightmap found, critical for [AnalyzedTerrain]")
        
        # Separate features by type
        peaks = [f for f in features if isinstance(f, PeakFeature)]
        ridges = [f for f in features if isinstance(f, RidgeFeature)]
        valleys = [f for f in features if isinstance(f, ValleyFeature)]
        saddles = [f for f in features if isinstance(f, SaddleFeature)]
        flat_zones = [f for f in features if isinstance(f, FlatZoneFeature)]
        
        # Apply semantic classification with game-specific thresholds
        self._classify_features_semantically(
            peaks, ridges, valleys, saddles, flat_zones,
            heightmap, slope, visibility, connectivity, flow_network
        )
        
        # Build semantic index for tactical queries
        semantic_index = self._build_semantic_index(
            peaks, ridges, valleys, saddles, flat_zones,
            visibility, connectivity, flow_network, heightmap
        )
        
        # Create final AnalyzedTerrain object
        analyzed = AnalyzedTerrain(
            source_heightmap=heightmap,
            peaks=peaks,
            valleys=valleys,
            ridges=ridges,
            saddles=saddles,
            flat_zones=flat_zones,
            visibility_graph=visibility,
            flow_network=flow_network,
            connectivity_graph=connectivity,
            watersheds=watersheds,
            flow_accumulation=flow_accum,
            semantic_index=semantic_index
        )
        
        # Log semantic summary
        #self._log_semantic_summary(analyzed)
        
        return analyzed
    
    def _classify_features_semantically(self, peaks: List, ridges: List,
                                     valleys: List, saddles: List,
                                     flat_zones: List, heightmap: Heightmap,
                                     slope: Optional[np.ndarray],
                                     visibility: Dict,
                                     connectivity: Dict,
                                     flow_network: Dict) -> None:
        """
        Add semantic tags to features based on game-specific thresholds.
        
        Uses relational graphs (visibility, connectivity, flow_network) to classify
        features according to the foundation document: Layer 5 consumes the graphs
        built by Layer 4 to produce tactical meaning.
        """
        cell_size = heightmap.config.horizontal_scale
        
        
        thresholds = self._template_thresholds
        
        # store bounds + offset
        self.total_area_m2 = heightmap.shape[0] * heightmap.shape[1] * (cell_size ** 2)
        self.max_feature_area = self.total_area_m2 * self._max_feature_coverage
        
        
        # Build feature type map for efficient lookups
        feature_type_map = {}
        for f in peaks + ridges + valleys + saddles + flat_zones:
            if isinstance(f, PeakFeature):
                feature_type_map[f.feature_id] = 'peak'
            elif isinstance(f, RidgeFeature):
                feature_type_map[f.feature_id] = 'ridge'
            elif isinstance(f, ValleyFeature):
                feature_type_map[f.feature_id] = 'valley'
            elif isinstance(f, SaddleFeature):
                feature_type_map[f.feature_id] = 'saddle'
            elif isinstance(f, FlatZoneFeature):
                feature_type_map[f.feature_id] = 'flat'
        
        # --- Classify Peaks (Defensive Positions + Observation Posts) ---
        for peak in peaks:
            tags = []
            visible_count = len(visibility.get(peak.feature_id, []))
            
            # Defensive position scoring (requires visibility from peak)
            if (peak.prominence >= thresholds["defensive_min_prominence_m"] and
                peak.elevation_range[1] >= thresholds["defensive_min_elevation_m"] and
                peak.avg_slope <= thresholds["defensive_max_slope_deg"] and
                visible_count >= thresholds["defensive_min_visibility"]):
                
                tags.append("defensive_position")
                # Score based on prominence and visibility
                defensive_score = min(1.0, (peak.prominence / self.config.defensive_prominence_divisor) * 
                                           (visible_count / self.config.defensive_visibility_divisor)
                                      )
                peak.metadata['defensive_score'] = defensive_score
            
            # Observation post (requires high visibility from peak)
            if (peak.prominence >= thresholds["observation_min_prominence_m"] and
                visible_count >= thresholds["observation_min_visibility"]):
                tags.append("observation_post")
                peak.metadata['visibility_count'] = visible_count
                peak.metadata['observation_score'] = min(1.0, visible_count / self.config.observation_visibility_divisor)
            
            # Major/Minor classification
            if peak.prominence > self.config.threshold_major_peak:
                tags.append("major_peak")
            elif peak.prominence > self.config.threshold_minor_peak:
                tags.append("minor_peak")
            else:
                tags.append("hill")
            
            peak.metadata['semantic_tags'] = tags
        
        # --- Classify Ridges (Cover Positions) ---
        # Ridges provide cover based on width and connection to defensive peaks
        for ridge in ridges:
            tags = []
            width = ridge.metadata.get('width_meters', 0)
            
            if width >= thresholds["cover_min_width_m"]:
                tags.append("defensive_cover")
                ridge.metadata['cover_quality'] = min(1.0, width / self.config.cover_quality_width_divisor)
            else:
                tags.append("exposed_crest")
            
            # Tactical significance based on connectivity graph
            connected_to = connectivity.get(ridge.feature_id, set())
            connected_peaks = [cid for cid in connected_to 
                              if feature_type_map.get(cid) == 'peak']
            
            if len(connected_peaks) > 1:
                tags.append("ridge_line")
                ridge.metadata['tactical_significance'] = "high"
            elif len(connected_peaks) == 1:
                ridge.metadata['tactical_significance'] = "medium"
            else:
                ridge.metadata['tactical_significance'] = "low"
            
            ridge.metadata['semantic_tags'] = tags
        
        # --- Classify Valleys (Drainage + Ambush Potential) ---
        for valley in valleys:
            tags = []
            
            # Drainage significance from flow network
            # Count how many features flow into this valley
            upstream_features = []
            for src_id, targets in flow_network.items():
                if valley.feature_id in targets:
                    upstream_features.append(src_id)
            
            upstream_count = len(upstream_features)
            if upstream_count > self.config.drainage_major_threshold:  # High flow
                tags.append("major_drainage")
                valley.metadata['drainage_magnitude'] = upstream_count
            elif upstream_count > self.config.drainage_minor_threshold: # low flow
                tags.append("minor_drainage")
                valley.metadata['drainage_magnitude'] = upstream_count
            else:                                                       # default to gully 
                tags.append("gully")
            
            # Ambush potential: narrow valleys with steep sides
            # Use connectivity graph to find adjacent steep ridges
            connected_to = connectivity.get(valley.feature_id, set())
            adjacent_ridges = [cid for cid in connected_to 
                              if feature_type_map.get(cid) == 'ridge']
            
            avg_slope = valley.metadata.get('avg_slope', 10.0)
            # Ambush requires both steep slopes and constricting terrain (adjacent ridges)
            if avg_slope > self.config.valley_avg_slope and len(adjacent_ridges) >= 2:
                tags.append("ambush_potential")
                valley.metadata['ambush_rating'] = min(1.0, (avg_slope / self.config.ambush_slope_divisor) * 
                                                            (len(adjacent_ridges) / self.config.ambush_ridge_divisor)
                                                       )
            elif avg_slope > self.config.valley_avg_slope:
                tags.append("steep_ravine")
            
            valley.metadata['semantic_tags'] = tags
        
        # --- Classify Saddles (Chokepoints) ---
        # Chokepoint are two steep ridges + narrow flat valley
        # Use connectivity graph to count ridges and valleys connected to the saddle
        for saddle in saddles:
            tags = []
            
            # Get all features connected to this saddle from connectivity graph
            connected_to = connectivity.get(saddle.feature_id, set())
            
            # Count ridges and valleys separately
            connected_ridges = [cid for cid in connected_to 
                               if feature_type_map.get(cid) == 'ridge']
            connected_valleys = [cid for cid in connected_to 
                                if feature_type_map.get(cid) == 'valley']
            
            ridge_count = len(connected_ridges)
            valley_count = len(connected_valleys)
            total_connections = ridge_count + valley_count
            
            # Chokepoint requires at least 2 ridges (constriction) OR
            # combination of ridges and valleys that create a natural pass
            if total_connections >= thresholds["chokepoint_min_connectivity"]:
                tags.append("chokepoint")
                saddle.metadata['chokepoint_degree'] = total_connections
                saddle.metadata['ridge_connections'] = ridge_count
                saddle.metadata['valley_connections'] = valley_count
                
                # Higher tactical value if chokepoint connects multiple ridges
                if ridge_count >= 2:
                    saddle.metadata['tactical_value'] = "high"
                elif ridge_count == 1 and valley_count >= 1:
                    saddle.metadata['tactical_value'] = "medium"
                else:
                    saddle.metadata['tactical_value'] = "low"
            
            # Pass classification based on elevation
            if saddle.elevation > self.config.saddle_elevation_high:
                tags.append("high_pass")
            elif saddle.elevation > self.config.saddle_elevation_low:
                tags.append("low_pass")
            else:
                tags.append("col")
            
            saddle.metadata['semantic_tags'] = tags
        
        # --- Classify Flat Zones (Assembly Areas) ---
        for flat in flat_zones:
            tags = []
            area_m2 = flat.metadata.get('area_m2', 0)
            max_slope = flat.max_slope
            
            # Assembly area: large, flat zones suitable for staging
            if (area_m2 >= thresholds["assembly_min_area_m2"] and
                max_slope <= thresholds["assembly_max_slope_deg"]):
                
                if area_m2 > self.config.assembly_major_area_threshold_m2:
                    tags.append("major_assembly_area")
                else:
                    tags.append("assembly_area")
                
                flat.metadata['assembly_capacity'] = area_m2 / self.config.assembly_capacity_divisor
                
                # Check connectivity to nearby tactical features
                # Use the peaks list directly instead of _feature_index
                connected_to = connectivity.get(flat.feature_id, set())
                nearby_defensive = []
                for cid in connected_to:
                    # Find the feature object to check its tags
                    for peak in peaks:
                        if peak.feature_id == cid and 'defensive_position' in peak.metadata.get('semantic_tags', []):
                            nearby_defensive.append(cid)
                            break
                flat.metadata['defensive_coverage'] = len(nearby_defensive)
            else:
                tags.append("small_clearing")
            
            # Traversability rating based on slope
            if max_slope < self.config.trafficability_ideal_threshold_deg:
                tags.append("ideal_traffic")
                flat.metadata['trafficability_rating'] = 1.0
            elif max_slope < self.config.trafficability_good_threshold_deg:
                tags.append("good_traffic")
                flat.metadata['trafficability_rating'] = 0.8
            else:
                flat.metadata['trafficability_rating'] = 0.5
            
            flat.metadata['semantic_tags'] = tags
    
    def _build_semantic_index(self, peaks, ridges, valleys, saddles, flat_zones,
                           visibility, connectivity, flow_network, heightmap) -> Dict[str, Any]:
        """
        Build searchable semantic index for tactical queries.
        """
        if not peaks and not ridges and not valleys and not saddles and not flat_zones:
            self._log("No features available. Semantic index will be empty.")
            return self._build_empty_semantic_index()
        
        semantic_index = {
            'defensive_positions': [],
            'observation_posts': [],
            'chokepoints': [],
            'assembly_areas': [],
            'cover_positions': [],
            'ambush_positions': [],
            'vehicle_routes': [],
            'feature_summary': {
                'total_peaks': len(peaks),
                'total_ridges': len(ridges),
                'total_valleys': len(valleys),
                'total_saddles': len(saddles),
                'total_flat_zones': len(flat_zones),
                'total_area_m2': self.total_area_m2,
                'max_feature_coverage_m2': self.max_feature_area,
            }
        }
        
        thresholds = self._get_template_thresholds()
        self._log("building index...")
        
        # --- Defensive Positions ---
        self._log(f" -> Defensive Positions [{len(peaks)}]")
        for peak in peaks:
            # safe metadata access with default empty list
            tags = peak.metadata.get('semantic_tags', [])
            
            if 'defensive_position' in tags:
                visible_set = visibility.get(peak.feature_id, set())
                visible_count = len(visible_set) if visible_set else 0
                
                semantic_index['defensive_positions'].append({
                    'id': peak.feature_id,
                    'centroid': peak.centroid,
                    'elevation': peak.elevation_range[1],
                    'prominence': peak.prominence,
                    'defensive_score': peak.metadata.get('defensive_score', 0),
                    'visibility_count': visible_count
                })
        
        # --- Observation Posts ---
        self._log(f" -> Observation Positions [{len(peaks)}]")
        for peak in peaks:
            tags = peak.metadata.get('semantic_tags', [])
            
            if 'observation_post' in tags:
                visible_set = visibility.get(peak.feature_id, set())
                visible_count = len(visible_set) if visible_set else 0
                
                semantic_index['observation_posts'].append({
                    'id': peak.feature_id,
                    'centroid': peak.centroid,
                    'elevation': peak.elevation_range[1],
                    'prominence': peak.prominence,
                    'visibility_count': visible_count
                })
        
        # --- Chokepoints ---
        self._log(f" -> Chokepoint Positions [{len(saddles)}]")
        for saddle in saddles:
            tags = saddle.metadata.get('semantic_tags', [])
            if 'chokepoint' in tags:
                semantic_index['chokepoints'].append({
                    'id': saddle.feature_id,
                    'centroid': saddle.centroid,
                    'elevation': saddle.elevation,
                    'connectivity_degree': saddle.metadata.get('chokepoint_degree', 0)
                })
        
        # --- Assembly Areas ---
        self._log(f" -> Assembly Areas [{len(flat_zones)}]")
        for flat in flat_zones:
            tags = flat.metadata.get('semantic_tags', [])
            if 'assembly_area' in tags or 'major_assembly_area' in tags:
                semantic_index['assembly_areas'].append({
                    'id': flat.feature_id,
                    'centroid': flat.centroid,
                    'area_m2': flat.metadata.get('area_m2', 0),
                    'capacity': flat.metadata.get('assembly_capacity', 0),
                    'trafficability': flat.metadata.get('trafficability_rating', 0.5)
                })
        
        # --- Cover Positions ---
        self._log(f" -> Cover Positions [{len(ridges)}]")
        for ridge in ridges:
            tags = ridge.metadata.get('semantic_tags', [])
            
            if 'defensive_cover' in tags:
                semantic_index['cover_positions'].append({
                    'id': ridge.feature_id,
                    'spine': ridge.spine_points,
                    'width_m': ridge.metadata.get('width_meters', 0),
                    'quality': ridge.metadata.get('cover_quality', 0)
                })
        
        # --- Ambush Positions ---
        self._log(f" -> Ambush Positions [{len(valleys)}]")
        for valley in valleys:
            tags = valley.metadata.get('semantic_tags', [])
            
            if 'ambush_potential' in tags:
                semantic_index['ambush_positions'].append({
                    'id': valley.feature_id,
                    'spine': valley.spine_points,
                    'rating': valley.metadata.get('ambush_rating', 0),
                    'centroid': valley.centroid,
                })
        
        # --- Vehicle Routes ---
        vehicle_routes = self._extract_vehicle_routes(connectivity, heightmap)
        semantic_index['vehicle_routes'] = vehicle_routes
        
        return semantic_index


    def _build_empty_semantic_index(self) -> Dict[str, Any]:
        """Return empty semantic index structure when no features exist."""
        return {
            'defensive_positions': [],
            'observation_posts': [],
            'chokepoints': [],
            'assembly_areas': [],
            'cover_positions': [],
            'ambush_positions': [],
            'vehicle_routes': [],
            'feature_summary': {
                'total_peaks': 0,
                'total_ridges': 0,
                'total_valleys': 0,
                'total_saddles': 0,
                'total_flat_zones': 0,
            }
        }
    
    def _extract_vehicle_routes(self, connectivity: Dict, heightmap: Heightmap) -> List[Dict]:
        """
        Extract traversable vehicle routes from connectivity graph.
        """
        routes = []
        
        # Find connected components in connectivity graph
        visited = set()
        components = []
        
        for f_id in connectivity.keys():
            if f_id in visited:
                continue
            
            # BFS to find component
            component = set()
            stack = [f_id]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                for neighbor in connectivity.get(node, set()):
                    if neighbor not in visited:
                        stack.append(neighbor)
            
            if len(component) > 1:
                components.append(component)
        
        # Score each component by size and terrain suitability
        for comp in components:
            route = {
                'feature_ids': list(comp),
                'size': len(comp),
                'avg_elevation': 0.0,
                'min_slope': float('inf'),
                'max_slope': 0.0
            }
            routes.append(route)
        
        return routes
    
    @property
    def output_schema(self) -> dict:
        return {
            "type": "AnalyzedTerrain",
            "fields": [
                "source_heightmap", "peaks", "valleys", "ridges", "saddles", "flat_zones",
                "visibility_graph", "flow_network", "connectivity_graph",
                "watersheds", "flow_accumulation", "semantic_index"
            ],
            "description": "Complete terrain analysis with tactical semantic tags"
        }


# Vehicle profile for traversability analysis
@dataclass
class VehicleProfile:
    """Define vehicle mobility characteristics for terrain analysis."""
    name: str
    max_slope_deg: float
    max_water_depth_m: float = 0.0
    width_m: float = 2.5
    length_m: float = 5.0
    
    def can_traverse_slope(self, slope_deg: float) -> bool:
        return slope_deg <= self.max_slope_deg
    
    def __repr__(self):
        return f"VehicleProfile({self.name}, max_slope={self.max_slope_deg}°)"


# Predefined vehicle profiles for supported games
VEHICLE_PROFILES = {
    'infantry': VehicleProfile('infantry', max_slope_deg=45.0, max_water_depth_m=0.5),
    'light_wheeled': VehicleProfile('light_wheeled', max_slope_deg=25.0, max_water_depth_m=0.3),
    'heavy_wheeled': VehicleProfile('heavy_wheeled', max_slope_deg=20.0, max_water_depth_m=0.5),
    'tracked': VehicleProfile('tracked', max_slope_deg=35.0, max_water_depth_m=1.0),
    'tank': VehicleProfile('tank', max_slope_deg=30.0, max_water_depth_m=1.5)
}