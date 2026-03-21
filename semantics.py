"""
Layer 5: Semantic Interpretation

Assembles all prior outputs into the final AnalyzedTerrain object.
Applies domain thresholds and classification rules to produce
tactical terrain categories for combat games (Arma, WoT, War Thunder).
"""

import numpy as np
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

from core import (
    Heightmap, ScalarField, PipelineConfig, PipelineLayer,
    TerrainFeature, ClassifiedFeature,
    PeakFeature, RidgeFeature, ValleyFeature, SaddleFeature, FlatZoneFeature,
    Traversability, CurvatureType, AnalyzedTerrain, LayerBundle,
    PixelCoord, WorldCoord, GameType
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
    
    def __init__(self, config: PipelineConfig, game_type: GameType = None):
        super().__init__(config)
        self.game_type = config.game_type
        
        # Game-specific tactical thresholds
        self._tactical_thresholds = self._get_game_thresholds()
        
        # Coverage filtering
        self._max_feature_coverage = 0.5  # Max 50% of map for any single feature
        
        print(f"Layer5_Semantics initialized for {self.game_type.value}")
    
    def _get_game_thresholds(self) -> Dict[str, Any]:
        """Get game-specific tactical thresholds."""
        thresholds = {
            GameType.ARMA_3: {
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
            GameType.WORLD_OF_TANKS: {
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
            GameType.WAR_THUNDER: {
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
            GameType.ARMA_2: {
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
        return thresholds.get(self.game_type, thresholds[GameType.ARMA_3])
    
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
        features = input_data.get('features', [])
        visibility = input_data.get('visibility_graph', {})
        flow_network = input_data.get('flow_network', {})
        connectivity = input_data.get('connectivity_graph', {})
        watersheds = input_data.get('watersheds', {})
        flow_accum = input_data.get('flow_accumulation', None)
        heightmap = input_data.get('heightmap')
        slope = input_data.get('slope', None)
        
        if heightmap is None:
            raise ValueError("Heightmap required for AnalyzedTerrain")
        
        # Separate features by type
        peaks = [f for f in features if isinstance(f, PeakFeature)]
        ridges = [f for f in features if isinstance(f, RidgeFeature)]
        valleys = [f for f in features if isinstance(f, ValleyFeature)]
        saddles = [f for f in features if isinstance(f, SaddleFeature)]
        flat_zones = [f for f in features if isinstance(f, FlatZoneFeature)]
        
        # Apply semantic classification with game-specific thresholds
        self._classify_features_semantically(
            peaks, ridges, valleys, saddles, flat_zones,
            heightmap, slope, visibility, connectivity
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
        self._log_semantic_summary(analyzed)
        
        return analyzed
    
    def _classify_features_semantically(self, peaks: List, ridges: List,
                                         valleys: List, saddles: List,
                                         flat_zones: List, heightmap: Heightmap,
                                         slope: Optional[np.ndarray],
                                         visibility: Dict,
                                         connectivity: Dict) -> None:
        """
        Add semantic tags to features based on game-specific thresholds.
        """
        cell_size = heightmap.config.horizontal_scale
        total_area_m2 = heightmap.shape[0] * heightmap.shape[1] * (cell_size ** 2)
        max_feature_area = total_area_m2 * self._max_feature_coverage
        thresholds = self._tactical_thresholds
        
        # --- Classify Peaks (Defensive Positions + Observation Posts) ---
        for peak in peaks:
            tags = []
            visible_count = len(visibility.get(peak.feature_id, []))
            
            # Defensive position scoring
            if (peak.prominence >= thresholds["defensive_min_prominence_m"] and
                peak.elevation_range[1] >= thresholds["defensive_min_elevation_m"] and
                peak.avg_slope <= thresholds["defensive_max_slope_deg"] and
                visible_count >= thresholds["defensive_min_visibility"]):
                
                tags.append("defensive_position")
                # Score based on prominence and visibility
                defensive_score = min(1.0, (peak.prominence / 20.0) * (visible_count / 10.0))
                peak.metadata['defensive_score'] = defensive_score
            
            # Observation post (high visibility)
            if (peak.prominence >= thresholds["observation_min_prominence_m"] and
                visible_count >= thresholds["observation_min_visibility"]):
                tags.append("observation_post")
                peak.metadata['visibility_count'] = visible_count
                peak.metadata['observation_score'] = min(1.0, visible_count / 15.0)
            
            # Major/Minor classification
            if peak.prominence > 15.0:
                tags.append("major_peak")
            elif peak.prominence > 5.0:
                tags.append("minor_peak")
            else:
                tags.append("hill")
            
            peak.metadata['semantic_tags'] = tags
        
        # --- Classify Ridges (Cover Positions) ---
        for ridge in ridges:
            tags = []
            width = ridge.metadata.get('width_meters', 0)
            
            if width >= thresholds["cover_min_width_m"]:
                tags.append("defensive_cover")
                ridge.metadata['cover_quality'] = min(1.0, width / 10.0)
            else:
                tags.append("exposed_crest")
            
            # Tactical significance based on connected peaks
            if len(ridge.connected_peaks) > 1:
                tags.append("ridge_line")
                ridge.metadata['tactical_significance'] = "high"
            else:
                ridge.metadata['tactical_significance'] = "medium"
            
            ridge.metadata['semantic_tags'] = tags
        
        # --- Classify Valleys (Drainage + Ambush Potential) ---
        for valley in valleys:
            tags = []
            
            # Drainage significance
            if valley.drainage_area and valley.drainage_area > 10000:
                tags.append("major_drainage")
            elif valley.drainage_area and valley.drainage_area > 1000:
                tags.append("minor_drainage")
            else:
                tags.append("gully")
            
            # Ambush potential (narrow valleys with steep sides)
            avg_slope = valley.metadata.get('avg_slope', 10.0)
            if avg_slope > 15.0:
                tags.append("ambush_potential")
                valley.metadata['ambush_rating'] = min(1.0, avg_slope / 30.0)
            
            valley.metadata['semantic_tags'] = tags
        
        # --- Classify Saddles (Chokepoints) ---
        for saddle in saddles:
            tags = []
            conn_degree = len(connectivity.get(saddle.feature_id, []))
            
            # Chokepoint detection (connectivity-based)
            if conn_degree >= thresholds["chokepoint_min_connectivity"]:
                tags.append("chokepoint")
                saddle.metadata['chokepoint_degree'] = conn_degree
            
            # Pass classification
            if saddle.elevation > 30:
                tags.append("high_pass")
            elif saddle.elevation > 10:
                tags.append("low_pass")
            else:
                tags.append("col")
            
            saddle.metadata['semantic_tags'] = tags
        
        # --- Classify Flat Zones (Assembly Areas) ---
        for flat in flat_zones:
            tags = []
            area_m2 = flat.metadata.get('area_m2', 0)
            max_slope = flat.max_slope
            
            # Assembly area (large, flat)
            if (area_m2 >= thresholds["assembly_min_area_m2"] and
                max_slope <= thresholds["assembly_max_slope_deg"]):
                
                if area_m2 > 10000:
                    tags.append("major_assembly_area")
                else:
                    tags.append("assembly_area")
                
                flat.metadata['assembly_capacity'] = area_m2 / 100
            else:
                tags.append("small_clearing")
            
            # Traversability rating
            if max_slope < 5:
                tags.append("ideal_traffic")
                flat.metadata['trafficability_rating'] = 1.0
            elif max_slope < 15:
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
            }
        }
        
        # Defensive positions
        for peak in peaks:
            tags = peak.metadata.get('semantic_tags', [])
            if 'defensive_position' in tags:
                semantic_index['defensive_positions'].append({
                    'id': peak.feature_id,
                    'centroid': peak.centroid,
                    'elevation': peak.elevation_range[1],
                    'prominence': peak.prominence,
                    'defensive_score': peak.metadata.get('defensive_score', 0),
                    'visibility_count': len(visibility.get(peak.feature_id, []))
                })
        
        # Observation posts
        for peak in peaks:
            tags = peak.metadata.get('semantic_tags', [])
            if 'observation_post' in tags:
                semantic_index['observation_posts'].append({
                    'id': peak.feature_id,
                    'centroid': peak.centroid,
                    'elevation': peak.elevation_range[1],
                    'prominence': peak.prominence,
                    'visibility_count': len(visibility.get(peak.feature_id, []))
                })
        
        # Chokepoints
        for saddle in saddles:
            tags = saddle.metadata.get('semantic_tags', [])
            if 'chokepoint' in tags:
                semantic_index['chokepoints'].append({
                    'id': saddle.feature_id,
                    'centroid': saddle.centroid,
                    'elevation': saddle.elevation,
                    'connectivity_degree': saddle.metadata.get('chokepoint_degree', 0)
                })
        
        # Assembly areas
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
        
        # Cover positions
        for ridge in ridges:
            tags = ridge.metadata.get('semantic_tags', [])
            if 'defensive_cover' in tags:
                semantic_index['cover_positions'].append({
                    'id': ridge.feature_id,
                    'spine': ridge.spine_points,
                    'width_m': ridge.metadata.get('width_meters', 0),
                    'quality': ridge.metadata.get('cover_quality', 0)
                })
        
        # Ambush positions
        for valley in valleys:
            tags = valley.metadata.get('semantic_tags', [])
            if 'ambush_potential' in tags:
                semantic_index['ambush_positions'].append({
                    'id': valley.feature_id,
                    'spine': valley.spine_points,
                    'rating': valley.metadata.get('ambush_rating', 0)
                })
        
        # Vehicle routes (extract from connectivity graph)
        vehicle_routes = self._extract_vehicle_routes(connectivity, heightmap)
        semantic_index['vehicle_routes'] = vehicle_routes
        
        return semantic_index
    
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
    
    def _log_semantic_summary(self, analyzed: AnalyzedTerrain) -> None:
        """Log semantic classification summary."""
        print("\n" + "=" * 60)
        print("LAYER 5: SEMANTIC INTERPRETATION")
        print(f"Game Type: {self.game_type.value}")
        print("=" * 60)
        
        # Extract semantic tags
        defensive = [p for p in analyzed.peaks if 'defensive_position' in p.metadata.get('semantic_tags', [])]
        observation = [p for p in analyzed.peaks if 'observation_post' in p.metadata.get('semantic_tags', [])]
        chokepoints = [s for s in analyzed.saddles if 'chokepoint' in s.metadata.get('semantic_tags', [])]
        assembly = [f for f in analyzed.flat_zones if 'assembly_area' in f.metadata.get('semantic_tags', [])]
        cover = [r for r in analyzed.ridges if 'defensive_cover' in r.metadata.get('semantic_tags', [])]
        ambush = [v for v in analyzed.valleys if 'ambush_potential' in v.metadata.get('semantic_tags', [])]
        
        print(f"\nTactical Features:")
        print(f"  • Defensive Positions: {len(defensive)}")
        print(f"  • Observation Posts: {len(observation)}")
        print(f"  • Chokepoints: {len(chokepoints)}")
        print(f"  • Assembly Areas: {len(assembly)}")
        print(f"  • Cover Positions: {len(cover)}")
        print(f"  • Ambush Positions: {len(ambush)}")
        
        # Feature summary
        print(f"\nFeature Summary:")
        print(f"  • Peaks: {len(analyzed.peaks)}")
        print(f"  • Ridges: {len(analyzed.ridges)}")
        print(f"  • Valleys: {len(analyzed.valleys)}")
        print(f"  • Saddles: {len(analyzed.saddles)}")
        print(f"  • Flat Zones: {len(analyzed.flat_zones)}")
        
        # Graph connectivity
        if analyzed.visibility_graph:
            vis_edges = sum(len(v) for v in analyzed.visibility_graph.values())
            print(f"\nGraph Connectivity:")
            print(f"  • Visibility Edges: {vis_edges // 2}")
        
        if analyzed.connectivity_graph:
            conn_edges = sum(len(v) for v in analyzed.connectivity_graph.values())
            print(f"  • Connectivity Edges: {conn_edges // 2}")
        
        if analyzed.watersheds:
            print(f"  • Watersheds: {len(analyzed.watersheds)}")
        
        # Top defensive positions
        if defensive:
            print(f"\nTop Defensive Positions:")
            sorted_def = sorted(defensive, key=lambda p: p.metadata.get('defensive_score', 0), reverse=True)
            for p in sorted_def[:5]:
                score = p.metadata.get('defensive_score', 0)
                print(f"  • {p.centroid}: {p.prominence:.1f}m prominence, score={score:.2f}")
    
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