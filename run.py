import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from core import PipelineConfig
from calibration import Layer0_Calibration
from lgeometry import Layer1_LocalGeometry
from rgeometry import Layer2_RegionalGeometry
from topological import Layer3_TopologicalFeatures
from relational import Layer4_Relational
from semantics import Layer5_Semantics


"""
    HMA Staging script (testrig)
   
"""

def run_pipeline(png_path: str, config: PipelineConfig) -> Dict:
    """Execute full pipeline, return all outputs."""
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
    
    # Layer 5
    layer5 = Layer5_Semantics(config)
    analyzed = layer5.execute(bundle)
    
    return {
        'heightmap': heightmap.data,
        'features': features,
        'relational': relational,
        'analyzed': analyzed,
        'semantic_index': analyzed.semantic_index,
    }



# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    # Setup config
    config = PipelineConfig()
    config.verbose = True
    run_pipeline('.//assets//3point.png', config)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    main()