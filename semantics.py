"""
Layer 5: Semantic Interpretation
"""

import numpy as np
from typing import Dict, List, Set, Optional, Tuple, Any

from core import (
    Heightmap, ScalarField, PipelineConfig, PipelineLayer,
    TerrainFeature, ClassifiedFeature,
    PeakFeature, RidgeFeature, ValleyFeature, SaddleFeature, FlatZoneFeature,
    AnalyzedTerrain, LayerBundle, PixelCoord
)


class Layer5_Semantics(PipelineLayer[AnalyzedTerrain]):
    """
    Assemble all prior outputs into the final AnalyzedTerrain object.
    
    Applies tactical classification:
    - Defensive positions (peaks with good visibility and moderate slopes)
    - Observation posts (high prominence peaks with wide visibility)
    - Chokepoints (saddles that connect multiple regions)
    - Assembly areas (large flat zones near objectives)
    - Cover positions (wide ridges that block LOS)
    """
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self._max_feature_coverage = self.config.max_feature_coverage

    def _log(self, msg: str) -> None:
        """Helper for verbose logging"""
        if self.config.verbose:
            print(f"[Semantics] {msg}")
    
    def execute(self, input_data: LayerBundle) -> AnalyzedTerrain:
        """
        Assemble final AnalyzedTerrain with tactical semantic tags.
        """
        features = input_data.get('features', [])
        if not features:
            self._log("No features found, aborting...")
            return # empty terrain
        
        # data set
        heightmap       = input_data.get('heightmap')
        slope           = input_data.get('slope', None)
        visibility      = input_data.get('visibility_graph', {})
        flow_network    = input_data.get('flow_network', {})
        connectivity    = input_data.get('connectivity_graph', {})
        watersheds      = input_data.get('watersheds', {})
        flow_accum      = input_data.get('flow_accumulation', None)
        
        if heightmap is None:
            self._log("No heightmap found, aborting...")
            return # empty terrain
        
        # Separate features by type
        peaks = [f for f in features if isinstance(f, PeakFeature)]
        ridges = [f for f in features if isinstance(f, RidgeFeature)]
        valleys = [f for f in features if isinstance(f, ValleyFeature)]
        saddles = [f for f in features if isinstance(f, SaddleFeature)]
        flat_zones = [f for f in features if isinstance(f, FlatZoneFeature)]
        
        self._log(f"Features: peaks={len(peaks)}, ridges={len(ridges)}, valleys={len(valleys)}, saddles={len(saddles)}, flats={len(flat_zones)}")
        self._log(f'Threshold coverage: {self._max_feature_coverage * 100}%')
        
        #return AnalyzedTerrain(...)