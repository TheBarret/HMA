"""
Layer 3: Topological Features — Ground Truth Validation

Uses synthetic_terrain.py to generate heightmaps with known features.
Compares detected features against ground truth.
"""

import numpy as np
from typing import Tuple, List

from core import PipelineConfig, GameType, PeakFeature, RidgeFeature, ValleyFeature, SaddleFeature, FlatZoneFeature
from factory import SyntheticTerrain, FeatureType, GroundTruthFeature
from calibration import Layer0_Calibration
from lgeometry import Layer1_LocalGeometry
from rgeometry import Layer2_RegionalGeometry
from topological import Layer3_TopologicalFeatures


# ─── Pipeline Runner ─────────────────────────────────────────────────────────

def run_full_pipeline(terrain_result, config: PipelineConfig):
    """Run all layers on synthetic terrain."""
    hm = terrain_result.heightmap
    
    #layer0 = Layer0_Calibration(config)
    layer1 = Layer1_LocalGeometry(config)
    layer2 = Layer2_RegionalGeometry(config)
    layer3 = Layer3_TopologicalFeatures(config)
    
    slope_aspect = layer1.execute(hm)
    curvature = layer2.execute(hm)
    
    bundle = {
        'heightmap': hm,
        'slope': slope_aspect['slope'],
        'aspect': slope_aspect['aspect'],
        'curvature': curvature['curvature'],
        'gaussian_curvature': curvature['gaussian_curvature'],
        'curvature_type': curvature['curvature_type'],
    }
    
    features = layer3.execute(bundle)
    
    # Separate by type
    peaks = [f for f in features if isinstance(f, PeakFeature)]
    ridges = [f for f in features if isinstance(f, RidgeFeature)]
    valleys = [f for f in features if isinstance(f, ValleyFeature)]
    saddles = [f for f in features if isinstance(f, SaddleFeature)]
    flat_zones = [f for f in features if isinstance(f, FlatZoneFeature)]
    
    return {
        'peaks': peaks,
        'ridges': ridges,
        'valleys': valleys,
        'saddles': saddles,
        'flat_zones': flat_zones,
    }


# ─── Validation Helpers ─────────────────────────────────────────────────────

def validate_peak(ground_truth: GroundTruthFeature, detected, tolerance_px=5, tolerance_height=0.2):
    """Check if detected peak matches ground truth."""
    if detected is None:
        return False, "no peak detected"
    
    pos_error = np.hypot(detected.centroid[0] - ground_truth.centroid[0],
                         detected.centroid[1] - ground_truth.centroid[1])
    if pos_error > tolerance_px:
        return False, f"position error {pos_error:.1f}px > {tolerance_px}"
    
    expected_height = ground_truth.properties.get('height', 0)
    height_error = abs(detected.prominence - expected_height) / expected_height
    if height_error > tolerance_height:
        return False, f"height error {height_error:.0%} > {tolerance_height:.0%}"
    
    return True, ""


# ─── Individual Tests ───────────────────────────────────────────────────────

def t_single_peak() -> Tuple[bool, str, str, str]:
    """Test: Single paraboloid peak detection."""
    terrain = SyntheticTerrain(size=128, h_scale=2.0, v_scale=0.2, sea_level=10.0)
    terrain.add_peak(x=64, y=64, height=40.0, radius=12.0, shape="paraboloid")
    result = terrain.build()
    
    terrain.save_png('t_single_peak.png')
    
    config = PipelineConfig(game_type=GameType.CUSTOM)
    config.curvature_epsilon_h_min = 1e-4
    config.curvature_epsilon_k_min = 1e-5
    config.peak_min_prominence_m = 1.0
    config.peak_shoulder_convex_ratio = 0.15
    
    detected = run_full_pipeline(result, config)
    ground_truth = result.ground_truth[0]
    
    if len(detected['peaks']) == 0:
        return False, "1 peak", "0 peaks detected", "Peak detection failed"
    
    ok, msg = validate_peak(ground_truth, detected['peaks'][0])
    return ok, "peak at (64,64), height 25m", msg, msg


def t_ridge() -> Tuple[bool, str, str, str]:
    """Test: Ridge detection."""
    terrain = SyntheticTerrain(size=128, h_scale=2.0, v_scale=0.2, sea_level=10.0)
    terrain.add_ridge(x1=30, y1=64, x2=98, y2=64, height=40.0, width=5.0)
    result = terrain.build()
    
    terrain.save_png('t_ridge.png')
    
    config = PipelineConfig(game_type=GameType.CUSTOM)
    config.min_ridge_length_px = 6
    
    detected = run_full_pipeline(result, config)
    
    if len(detected['ridges']) == 0:
        return False, "≥1 ridge", "0 ridges", "Ridge detection failed"
    
    # Check ridge length
    ridge = detected['ridges'][0]
    spine_len = len(ridge.spine_points)
    if spine_len < 40:  # Expected ~68px from 30 to 98
        return False, f"ridge length >40px", f"length={spine_len}px", "Ridge too short"
    
    return True, "ridge detected, length >40px", f"length={spine_len}px", ""


def t_saddle() -> Tuple[bool, str, str, str]:
    """Test: Saddle between two peaks."""
    terrain = SyntheticTerrain(size=128, h_scale=2.0, v_scale=0.2, sea_level=10.0)
    terrain.add_peak(x=48, y=64, height=40.0, radius=12.0, shape="paraboloid")
    terrain.add_peak(x=80, y=64, height=40.0, radius=12.0, shape="paraboloid")
    # Use a small, localized saddle so it doesn't globally inflate the elevation floor.
    # influence_radius=10 keeps the hyperboloid confined to the gap between the two peaks.
    terrain.add_saddle(x=64, y=64, height=24.0, a=0.001, b=0.001, influence_radius=10.0)
    result = terrain.build()
    
    terrain.save_png('t_saddle.png')
    
    config = PipelineConfig(game_type=GameType.CUSTOM)
    config.curvature_epsilon_h_min = 1e-4
    config.curvature_epsilon_k_min = 1e-5
    
    detected = run_full_pipeline(result, config)
    
    peaks_ok = len(detected['peaks']) >= 2
    saddles_ok = len(detected['saddles']) >= 1
    
    if not peaks_ok:
        return False, "≥2 peaks", f"{len(detected['peaks'])} peaks", "Peak detection failed"
    if not saddles_ok:
        return False, "≥1 saddle", f"{len(detected['saddles'])} saddles", "Saddle detection failed"
    
    return True, "2 peaks, 1 saddle", f"peaks={len(detected['peaks'])}, saddles={len(detected['saddles'])}", ""


def t_flat_zone() -> Tuple[bool, str, str, str]:
    """Test: Flat zone detection."""
    terrain = SyntheticTerrain(size=128, h_scale=2.0, v_scale=0.2, sea_level=10.0)
    terrain.add_flat_zone(x_min=20, x_max=50, y_min=20, y_max=50, elevation=10.0)
    result = terrain.build()
    
    terrain.save_png('t_flatzone.png')
    
    config = PipelineConfig(game_type=GameType.CUSTOM)
    config.min_flat_zone_size_px = 50
    
    detected = run_full_pipeline(result, config)
    
    if len(detected['flat_zones']) == 0:
        return False, "≥1 flat zone", "0 flat zones", "Flat zone detection failed"
    
    # Check area (30×30 = 900px, min is 50px)
    area = detected['flat_zones'][0].area_pixels
    if area < 900 * 0.5:  # Allow 50% area loss due to boundaries
        return False, f"area ≥450px", f"area={area}px", "Flat zone area too small"
    
    return True, "flat zone detected", f"area={area}px", ""


# ─── Runner ─────────────────────────────────────────────────────────────────

def run():
    print("\n[L3] Topological Features — Ground Truth Validation")
    print("-" * 60)
    
    tests = [
        ("single_peak", t_single_peak),
        ("ridge", t_ridge),
        ("saddle", t_saddle),
        ("flat_zone", t_flat_zone),
    ]
    
    all_ok = True
    for name, fn in tests:
        print("=" * 60)
        ok, exp, got, why = fn()
        status = "GOOD" if ok else "FAIL"
        print(f"[L3|{name:16}].......[{status}]", end="")
        if not ok:
            all_ok = False
            print(f"\n\texpected: {exp}\n\tobtained: {got}\n\twhy:      {why}")
        else:
            print()
    
    print("-" * 60)
    print(f"[SUMMARY] {'PASS' if all_ok else 'FAILED'}\n")
    return all_ok


if __name__ == "__main__":
    run()