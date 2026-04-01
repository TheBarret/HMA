import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from core import PipelineConfig, PeakFeature, RidgeFeature, ValleyFeature, SaddleFeature, FlatZoneFeature
from calibration import Layer0_Calibration
from lgeometry import Layer1_LocalGeometry
from rgeometry import Layer2_RegionalGeometry
from topological import Layer3_TopologicalFeatures
from relational import Layer4_Relational
from semantics import Layer5_Semantics
from visualizer import render

def run_pipeline(png_path: str, config: PipelineConfig, selfcheck: bool = False):
    #""" self check """
    if selfcheck:
        from tools import SelfTest
        self_test = SelfTest(config)
        self_test.run()
        return
    
    from PIL import Image
    raw = np.array(Image.open(png_path).convert('L'))
    
    # Layer 0
    layer0 = Layer0_Calibration(config)
    heightmap = layer0.execute(raw)
    
    # Layer 1
    layer1 = Layer1_LocalGeometry(config)
    slope_aspect = layer1.execute(heightmap)
    
    # Layer 2
    layer2 = Layer2_RegionalGeometry(config)
    curvature = layer2.execute(heightmap)
    
    # Layer 3
    layer3 = Layer3_TopologicalFeatures(config)
    bundle = {
        'heightmap': heightmap,
        'slope': slope_aspect['slope'],
        'aspect': slope_aspect['aspect'],
        'curvature': curvature['curvature'],
        'gaussian_curvature': curvature['gaussian_curvature'],
        'curvature_type': curvature['curvature_type'],
    }
    features = layer3.execute(bundle)
    bundle['features'] = features
    
    # Layer 4
    layer4 = Layer4_Relational(config)
    relational = layer4.execute(bundle)
    bundle.update(relational)
    
    # visualize
    map_name = Path(png_path).stem
    render(
        bundle=bundle,
        features=features,
        relational=relational,
        map_name=map_name,
        save_path=f"{map_name}_topology.png",
        dpi=180
    )
    return
    



# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    # Setup config
    config = PipelineConfig()
    config.verbose = True
    run_pipeline('./assets/3point.png', config)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    main()