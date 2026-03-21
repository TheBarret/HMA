"""
Semantic Interpretation

Assembles all prior outputs into the final AnalyzedTerrain object.
Applies domain thresholds and classification rules to produce
human-meaningful terrain categories.
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
    PixelCoord, WorldCoord
)

class Layer5_Semantics(PipelineLayer[AnalyzedTerrain]):
    
    def __init__(self, config: PipelineConfig, terrain_type: str = "gentle"):
        super().__init__(config)
        
        # Terrain-type calibrated thresholds
        if terrain_type == "mountainous":
            self._defensive_height_min = 10.0
            self._defensive_prominence_min = 10.0
            self._defensive_slope_max = 25.0
            self._assembly_area_min = 5000
            self._assembly_area_max = float('inf')
            self._chokepoint_connectivity_min = 2
            self._cover_thickness = 5.0
            
        elif terrain_type == "gentle":
            # Lower thresholds for rolling terrain
            self._defensive_height_min = 5.0
            self._defensive_prominence_min = 3.0
            self._defensive_slope_max = 25.0
            self._assembly_area_min = 1000
            self._assembly_area_max = 50000
            self._chokepoint_connectivity_min = 1
            self._cover_thickness = 2.0
            
        else:  # "mixed" or default
            self._defensive_height_min = 8.0
            self._defensive_prominence_min = 5.0
            self._defensive_slope_max = 25.0
            self._assembly_area_min = 2000
            self._assembly_area_max = 200000
            self._chokepoint_connectivity_min = 1
            self._cover_thickness = 3.0
            
        # Also track coverage percentage for filtering
        self._max_feature_coverage = 0.5  # Max 50% of map for any single feature
        print(f'Layer5_Semantics profile: {terrain_type}')
        
    def execute(self, input_data: LayerBundle) -> AnalyzedTerrain:
        """
        Assemble final AnalyzedTerrain object with semantic tags.
        
        Args:
            input_data: LayerBundle containing:
                - features: List[TerrainFeature] from Layer 3
                - visibility_graph: Dict[str, Set[str]] from Layer 4
                - flow_network: Dict[str, List[str]] from Layer 4
                - connectivity_graph: Dict[str, Set[str]] from Layer 4
                - watersheds: Dict[str, Set[str]] from Layer 4
                - flow_accumulation: ScalarField from Layer 4
                - heightmap: Heightmap from Layer 0
                - slope: Slope field from Layer 1 (optional)
                
        Returns:
            AnalyzedTerrain: Fully populated terrain analysis object
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
        
        # Apply semantic classification to features
        self._classify_features_semantically(
            peaks, ridges, valleys, saddles, flat_zones, heightmap, slope, connectivity
        )
        
        # Build semantic indexes for tactical analysis
        semantic_index = self._build_semantic_index(
            peaks, ridges, valleys, saddles, flat_zones,
            visibility, connectivity, flow_network
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
                                         connectivity: Dict) -> None:
        """
        Add semantic tags to features based on terrain context.
        """
        cell_size = heightmap.config.horizontal_scale
        total_area_m2 = heightmap.shape[0] * heightmap.shape[1] * (cell_size ** 2)
        max_feature_area = total_area_m2 * self._max_feature_coverage
        
        # Classify peaks
        for peak in peaks:
            tags = []
            
            # Prominence-based classification
            if peak.prominence > 20:
                tags.append("major_peak")
            elif peak.prominence > 5:
                tags.append("minor_peak")
            else:
                tags.append("hill")
            
            # Defensive position potential (calibrated for terrain)
            if (peak.prominence > self._defensive_prominence_min and 
                peak.avg_slope < self._defensive_slope_max and
                peak.elevation_range[1] > self._defensive_height_min):
                tags.append("defensive_position")
                peak.metadata['defensive_rating'] = min(1.0, peak.prominence / 20)
            
            # Observation post potential
            if peak.prominence > 3:
                tags.append("observation_post")
                peak.metadata['visibility_potential'] = min(1.0, peak.prominence / 15)
            
            peak.metadata['semantic_tags'] = tags
        
        # Classify ridges
        for ridge in ridges:
            tags = []
            
            # Get ridge width in meters
            width = ridge.metadata.get('width_meters', 0)
            
            # Cover classification
            if width > self._cover_thickness:
                tags.append("defensive_cover")
                ridge.metadata['cover_quality'] = min(1.0, width / 10)
            else:
                tags.append("exposed_crest")
            
            # Tactical significance
            if len(ridge.connected_peaks) > 1:
                tags.append("ridge_line")
                ridge.metadata['tactical_significance'] = "high"
            else:
                ridge.metadata['tactical_significance'] = "medium"
            
            ridge.metadata['semantic_tags'] = tags
        
        # Classify valleys
        for valley in valleys:
            tags = []
            
            # Drainage significance
            if valley.drainage_area and valley.drainage_area > 10000:
                tags.append("major_drainage")
            elif valley.drainage_area and valley.drainage_area > 1000:
                tags.append("minor_drainage")
            else:
                tags.append("gully")
            
            # Trafficability (valleys are often difficult)
            tags.append("wet_area")
            valley.metadata['trafficability_penalty'] = 1.5
            
            valley.metadata['semantic_tags'] = tags
        
        # Classify saddles (using graph connectivity for chokepoints)
        for saddle in saddles:
            tags = []
            
            # Pass classification based on elevation
            if saddle.elevation > 30:
                tags.append("high_mountain_pass")
            elif saddle.elevation > 10:
                tags.append("low_pass")
            else:
                tags.append("col")
            
            # Check chokepoint from graph connectivity (will be set in _build_semantic_index)
            # For now, just initialize
            saddle.metadata['semantic_tags'] = tags
        
        # Classify flat zones (with area cap)
        for flat in flat_zones:
            tags = []
            area_m2 = flat.metadata.get('area_m2', 0)
            
            # Skip features that are too large (map-wide)
            if area_m2 > max_feature_area:
                tags.append("background_terrain")
            else:
                # Assembly area classification with upper bound
                if area_m2 > self._assembly_area_max:
                    tags.append("large_plain")
                elif area_m2 > self._assembly_area_min:
                    tags.append("assembly_area")
                    flat.metadata['assembly_capacity'] = area_m2 / 100
                elif area_m2 > 100:
                    tags.append("landing_zone")
                else:
                    tags.append("small_clearing")
                
                # Traversability rating
                if flat.max_slope < 5:
                    tags.append("ideal_traffic")
                    flat.metadata['trafficability_rating'] = 1.0
                elif flat.max_slope < 15:
                    tags.append("good_traffic")
                    flat.metadata['trafficability_rating'] = 0.8
                else:
                    flat.metadata['trafficability_rating'] = 0.5
            
            flat.metadata['semantic_tags'] = tags
    
    def _build_semantic_index(self, peaks, ridges, valleys, saddles, flat_zones,
                               visibility, connectivity, flow_network) -> Dict[str, Any]:
        """
        Build searchable semantic index with graph-based classification.
        """
        
        semantic_index = {
            'defensive_positions': [],
            'chokepoints': [],
            'observation_posts': [],
            'assembly_areas': [],
            'cover_positions': [],
            'critical_nodes': []
        }
        
        # Defensive positions (peaks with defensive rating)
        for peak in peaks:
            tags = peak.metadata.get('semantic_tags', [])
            if 'defensive_position' in tags:
                semantic_index['defensive_positions'].append({
                    'id': peak.feature_id,
                    'centroid': peak.centroid,
                    'elevation': peak.elevation_range[1],
                    'prominence': peak.prominence,
                    'rating': peak.metadata.get('defensive_rating', 0)
                })
        
        # Chokepoints: saddles that connect multiple features (from connectivity graph)
        for saddle in saddles:
            # Check connectivity degree from graph
            conn_degree = len(connectivity.get(saddle.feature_id, []))
            
            if conn_degree >= self._chokepoint_connectivity_min:
                # Add semantic tag to saddle
                if 'semantic_tags' not in saddle.metadata:
                    saddle.metadata['semantic_tags'] = []
                if 'chokepoint' not in saddle.metadata['semantic_tags']:
                    saddle.metadata['semantic_tags'].append('chokepoint')
                
                semantic_index['chokepoints'].append({
                    'id': saddle.feature_id,
                    'centroid': saddle.centroid,
                    'elevation': saddle.elevation,
                    'connectivity_score': conn_degree
                })
        
        # Observation posts (peaks with good visibility)
        for peak in peaks:
            visible_count = len(visibility.get(peak.feature_id, []))
            if visible_count > 5:
                semantic_index['observation_posts'].append({
                    'id': peak.feature_id,
                    'centroid': peak.centroid,
                    'visible_features': visible_count,
                    'visibility_score': peak.metadata.get('visibility_potential', 0)
                })
        
        # Assembly areas (large flat zones)
        for flat in flat_zones:
            tags = flat.metadata.get('semantic_tags', [])
            if 'assembly_area' in tags:
                semantic_index['assembly_areas'].append({
                    'id': flat.feature_id,
                    'centroid': flat.centroid,
                    'area_m2': flat.metadata.get('area_m2', 0),
                    'capacity': flat.metadata.get('assembly_capacity', 0)
                })
        
        # Cover positions (wide ridges)
        for ridge in ridges:
            tags = ridge.metadata.get('semantic_tags', [])
            if 'defensive_cover' in tags:
                semantic_index['cover_positions'].append({
                    'id': ridge.feature_id,
                    'spine': ridge.spine_points,
                    'width': ridge.metadata.get('width_meters', 0),
                    'quality': ridge.metadata.get('cover_quality', 0)
                })
        
        # Critical nodes (highly connected features)
        all_nodes = {**{p.feature_id: p for p in peaks},
                    **{s.feature_id: s for s in saddles}}
        
        for f_id, feature in all_nodes.items():
            conn_count = len(connectivity.get(f_id, []))
            vis_count = len(visibility.get(f_id, []))
            
            if conn_count > 3 or vis_count > 10:
                semantic_index['critical_nodes'].append({
                    'id': f_id,
                    'type': feature.__class__.__name__,
                    'centroid': feature.centroid,
                    'connectivity': conn_count,
                    'visibility': vis_count
                })
        
        return semantic_index
    
    def _log_semantic_summary(self, analyzed: AnalyzedTerrain) -> None:
        """Log semantic classification summary."""
        print("\n" + "=" * 60)
        print("LAYER 5: SEMANTIC INTERPRETATION")
        print("=" * 60)
        
        # Extract semantic tags from features
        defensive = []
        chokepoints = []
        observation = []
        assembly = []
        cover = []
        
        for peak in analyzed.peaks:
            tags = peak.metadata.get('semantic_tags', [])
            if 'defensive_position' in tags:
                defensive.append(peak)
            if 'observation_post' in tags:
                observation.append(peak)
        
        for saddle in analyzed.saddles:
            tags = saddle.metadata.get('semantic_tags', [])
            if 'chokepoint' in tags:
                chokepoints.append(saddle)
        
        for flat in analyzed.flat_zones:
            tags = flat.metadata.get('semantic_tags', [])
            if 'assembly_area' in tags:
                assembly.append(flat)
        
        for ridge in analyzed.ridges:
            tags = ridge.metadata.get('semantic_tags', [])
            if 'defensive_cover' in tags:
                cover.append(ridge)
        
        print(f"\nTactical Features:")
        print(f"  Defensive Positions: {len(defensive)}")
        print(f"  Observation Posts: {len(observation)}")
        print(f"  Chokepoints: {len(chokepoints)}")
        print(f"  Assembly Areas: {len(assembly)}")
        print(f"  Cover Positions: {len(cover)}")
        
        # Feature type summary
        print(f"\nFeature Summary:")
        print(f"  Peaks: {len(analyzed.peaks)}")
        print(f"  Ridges: {len(analyzed.ridges)}")
        print(f"  Valleys: {len(analyzed.valleys)}")
        print(f"  Saddles: {len(analyzed.saddles)}")
        print(f"  Flat Zones: {len(analyzed.flat_zones)}")
        
        # Graph connectivity summary
        if analyzed.visibility_graph:
            vis_edges = sum(len(v) for v in analyzed.visibility_graph.values())
            print(f"\nGraph Connectivity:")
            print(f"  Visibility Graph: {vis_edges // 2} edges")
        
        if analyzed.connectivity_graph:
            conn_edges = sum(len(v) for v in analyzed.connectivity_graph.values())
            print(f"  Connectivity Graph: {conn_edges // 2} edges")
        
        if analyzed.watersheds:
            print(f"  Watersheds: {len(analyzed.watersheds)}")
        
        # Query example
        if defensive:
            print(f"\nExample Query: Find all defensive positions with prominence > 30m")
            defensive_high = [p for p in defensive if p.prominence > 30]
            for p in defensive_high[:5]:
                print(f"  Peak at {p.centroid}: {p.prominence:.1f}m prominence, rating={p.metadata.get('defensive_rating', 0):.2f}")

    @property
    def output_schema(self) -> dict:
        """Schema for pipeline validation."""
        return {
            "type": "AnalyzedTerrain",
            "fields": [
                "source_heightmap",
                "peaks",
                "valleys",
                "ridges",
                "saddles",
                "flat_zones",
                "visibility_graph",
                "flow_network",
                "connectivity_graph",
                "watersheds",
                "flow_accumulation"
            ],
            "description": "Complete terrain analysis with semantic tags"
        }


# Additional utility: Vehicle-specific traversability analysis
class VehicleProfile:
    """
    Define vehicle mobility characteristics for terrain analysis.
    """
    
    def __init__(self, name: str, max_slope: float, max_water_depth: float = 0.0,
                 width: float = 2.5, length: float = 5.0):
        self.name = name
        self.max_slope = max_slope
        self.max_water_depth = max_water_depth
        self.width = width
        self.length = length
    
    def can_traverse(self, feature: ClassifiedFeature) -> bool:
        """Check if this vehicle can traverse a given feature."""
        if feature.avg_slope > self.max_slope:
            return False
        
        # Check water crossing for valleys
        if isinstance(feature, ValleyFeature):
            # Valleys may have water accumulation
            if feature.drainage_area and feature.drainage_area > 1000:
                return self.max_water_depth > 0.5
        
        return True
    
    def __repr__(self):
        return f"VehicleProfile({self.name}, max_slope={self.max_slope}°)"


# Predefined vehicle profiles
VEHICLE_PROFILES = {
    'infantry': VehicleProfile('infantry', max_slope=45.0, max_water_depth=0.5),
    'light_wheeled': VehicleProfile('light_wheeled', max_slope=25.0, max_water_depth=0.3),
    'heavy_wheeled': VehicleProfile('heavy_wheeled', max_slope=20.0, max_water_depth=0.5),
    'tracked': VehicleProfile('tracked', max_slope=35.0, max_water_depth=1.0),
    'tank': VehicleProfile('tank', max_slope=30.0, max_water_depth=1.5)
}