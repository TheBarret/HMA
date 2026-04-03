import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from core import ( PipelineConfig, PeakFeature, RidgeFeature, ValleyFeature, SaddleFeature, FlatZoneFeature,Traversability )

# context objects
from context import AnalyzedTerrain

from calibration  import Layer0_Calibration
from lgeometry    import Layer1_LocalGeometry
from rgeometry    import Layer2_RegionalGeometry
from topological  import Layer3_TopologicalFeatures
from relational   import Layer4_Relational


def run_pipeline(png_path: str,
                 config: PipelineConfig,
                 generate_plot: bool = True,
                 self_check: bool = False):

    if self_check:
        from tools import SelfTest
        SelfTest(config).run()
        return

    from PIL import Image
    raw = np.array(Image.open(png_path).convert('L'))

    # --- Layer 0 --------------------------------------------------------------
    heightmap = Layer0_Calibration(config).execute(raw)

    # --- Layer 1 --------------------------------------------------------------
    slope_aspect = Layer1_LocalGeometry(config).execute(heightmap)

    # --- Layer 2 --------------------------------------------------------------
    curvature = Layer2_RegionalGeometry(config).execute(heightmap)

    # --- Layer 3 --------------------------------------------------------------
    bundle = {
        'heightmap'         : heightmap,
        'slope'             : slope_aspect['slope'],
        'aspect'            : slope_aspect['aspect'],
        'curvature'         : curvature['curvature'],
        'gaussian_curvature': curvature['gaussian_curvature'],
        'curvature_type'    : curvature['curvature_type'],
    }
    bundle['features'] = Layer3_TopologicalFeatures(config).execute(bundle)

    # --- Layer 4 --------------------------------------------------------------
    relational = Layer4_Relational(config).execute(bundle)
    bundle.update(relational)

    # --- Sort features into typed lists ---------------------------------------
    all_features = bundle['features']

    peaks      = [f for f in all_features if isinstance(f, PeakFeature)]
    ridges     = [f for f in all_features if isinstance(f, RidgeFeature)]
    valleys    = [f for f in all_features if isinstance(f, ValleyFeature)]
    saddles    = [f for f in all_features if isinstance(f, SaddleFeature)]
    flat_zones = [f for f in all_features if isinstance(f, FlatZoneFeature)]

    # --- Assemble AnalyzedTerrain ---------------------------------------------
    terrain = AnalyzedTerrain(
        source_heightmap   = heightmap,
        peaks              = peaks,
        ridges             = ridges,
        valleys            = valleys,
        saddles            = saddles,
        flat_zones         = flat_zones,
        visibility_graph   = relational.get('visibility_graph',   {}),
        flow_network       = relational.get('flow_network',       {}),
        connectivity_graph = relational.get('connectivity_graph', {}),
        watersheds         = relational.get('watersheds',         {}),
        flow_accumulation  = relational.get('flow_accumulation',  None),
    )

    if terrain:
        from shell import launch
        launch(terrain, map_name="terrain")
        
    # --- Spatial pings --------------------------------------------------------
    #x, y, r = 250, 250, 300
    #print(f"-> Terrain.ping({x},{y}, radius={r}mtr)")
   
    #_nearby = terrain.ping_at(x, y, r)
    #_peaks = terrain.ping_at(x, y, r, PeakFeature)
    #_valleys = terrain.ping_at(x, y, r, ValleyFeature)
    #print(f"    - total : {len(_nearby)}")
    #print(f"    - peaks : {len(_peaks)}")
    #print(f"    - valley: {len(_valleys)}")

    # --- Query: upstream valleys ----------------------------------------------
    #if _valleys:
    #    outlet_id        = _valleys[0].feature_id
    #    upstream_valleys = (terrain.query()
    #                        .select(ValleyFeature)
    #                        .upstream_of(outlet_id)
    #                        .execute())
    #    print(f"    - upstream 'outlet_{outlet_id[:8]}': {len(upstream_valleys)} valleys found")
        
    # --- major peaks ---------------------------------------------------
    #major_peaks = (terrain.query()
    #               .select(PeakFeature)
    #               .where(prominence__gt=50)
    #               .order_by("prominence", descending=True)
    #               .execute())

    #print(f"    - prominence > 50m: {len(major_peaks)}")
    #if major_peaks:
    #    neighbours = terrain.ping_from(major_peaks[0].feature_id, radius_m=300)
    #    print(f"    - within 300m: {len(neighbours)} features")

    # --- Describe features ----------------------------------------------------
    #if major_peaks:
    #    peak = major_peaks[0]
    #    print(f"\nFull:  {terrain.describe(peak.feature_id)}")
    #    print(f"Brief: {terrain.describe(peak.feature_id, brief=True)}")
    #
    #if ridges:
    #    print(f"Full:  {terrain.describe(ridges[0].feature_id)}")

    # --- Traversability -------------------------------------------------------
    #tank          = config.VEHICLE_PROFILES['tank']
    #passable_flats = [f for f in flat_zones
    #                  if f.is_traversable(config) == Traversability.FREE]
    #print(f"\nFlat zones passable by all vehicles: {len(passable_flats)}")
    #print(f"Tank max slope: {tank.max_slope_deg}°")

    # --- Visualizer -----------------------------------------------------------
    #if generate_plot:
    #    from visualizer import render
    #    map_name = Path(png_path).stem
    #    render(
    #        bundle    = bundle,
    #        features  = all_features,
    #        relational = relational,
    #        map_name  = map_name,
    #        save_path = f"{map_name}_topology.png",
    #        dpi       = 180,
    #    )
    #return terrain


# =============================================================================
#  CLI
# =============================================================================

def main():
    config         = PipelineConfig()
    config.verbose = True
    run_pipeline('./assets/spike.png', config)


if __name__ == '__main__':
    main()