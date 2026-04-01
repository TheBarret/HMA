"""
Self-test module for HMA pipeline.
Verifies feature detection on synthetic terrain. Prints warnings on mismatch.
"""

import numpy as np
from typing import List, Tuple, Optional

from core import PipelineConfig
from factory import SyntheticTerrain
from calibration import Layer0_Calibration
from lgeometry import Layer1_LocalGeometry
from rgeometry import Layer2_RegionalGeometry
from topological import Layer3_TopologicalFeatures


class SelfTest:
    """Quick self-test of feature detection. Runs at startup, warns only."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.config.exclude_below_reference = False
        self._log("Test parameters: exclude_below_reference=False")
        self.warnings = []
    
    def _log(self, msg: str) -> None:
        """Helper for verbose logging"""
        if self.config.verbose:
            print(f"[SelfTest] {msg}")
    
    def _run_layer3(self, heightmap) -> List:
        """Run pipeline up to Layer 3 and return features."""
        # Layer 0: already calibrated from factory
        # Layer 1: Slope & Aspect
        l1 = Layer1_LocalGeometry(self.config)
        l1_result = l1.execute(heightmap)
        
        # Layer 2: Curvature
        l2 = Layer2_RegionalGeometry(self.config)
        l2_result = l2.execute(heightmap)
        
        # Layer 3: Topological Features
        bundle = {
            'heightmap': heightmap,
            'slope': l1_result['slope'],
            'aspect': l1_result['aspect'],
            'curvature': l2_result['curvature'],
            'curvature_type': l2_result['curvature_type'],
            'gaussian_curvature': l2_result['gaussian_curvature']
        }
        
        l3 = Layer3_TopologicalFeatures(self.config)
        return l3.execute(bundle)
    
    def _count_features(self, features: List) -> dict:
        """Count features by type."""
        from core import PeakFeature, RidgeFeature, ValleyFeature, SaddleFeature, FlatZoneFeature
        
        return {
            'peaks': sum(1 for f in features if isinstance(f, PeakFeature)),
            'ridges': sum(1 for f in features if isinstance(f, RidgeFeature)),
            'valleys': sum(1 for f in features if isinstance(f, ValleyFeature)),
            'saddles': sum(1 for f in features if isinstance(f, SaddleFeature)),
            'flat_zones': sum(1 for f in features if isinstance(f, FlatZoneFeature)),
        }
    
    def test_peak(self) -> bool:
        """Single peak at center."""
        self._log("Testing peak detection...")
        
        gen = SyntheticTerrain(self.config, size_px=128)
        gen.add_peak(0.0, 0.0, height_m=50.0, radius_m=12.0, shape="gaussian")
        result = gen.build()
        
        features = self._run_layer3(result.heightmap)
        counts = self._count_features(features)
        
        ok = counts['peaks'] >= 1
        if not ok:
            self.warnings.append(f"PEAK: expected ≥1, got {counts['peaks']}")
        
        self._log("PASSED" if ok else "FAILED")
        return ok
    
    def test_ridge(self) -> bool:
        """Horizontal ridge across map."""
        self._log("Testing ridge detection...")
        
        gen = SyntheticTerrain(self.config, size_px=128)
        half = (128 * self.config.horizontal_scale) / 2
        gen.add_ridge(-half + 10, 0.0, half - 10, 0.0, height_m=40.0, width_m=4.0)
        result = gen.build()
        
        features = self._run_layer3(result.heightmap)
        counts = self._count_features(features)
        
        ok = counts['ridges'] >= 1
        if not ok:
            self.warnings.append(f"RIDGE: expected ≥1, got {counts['ridges']}")
        
        self._log("PASSED" if ok else "FAILED")
        return ok
    
    def test_valley(self) -> bool:
        """Valley line across map."""
        self._log("Testing valley detection...")
        
        gen = SyntheticTerrain(self.config, size_px=128)
        half = (128 * self.config.horizontal_scale) / 2
        gen.add_valley(-half + 10, 0.0, half - 10, 0.0, depth_m=30.0, width_m=5.0)
        result = gen.build()
        
        features = self._run_layer3(result.heightmap)
        counts = self._count_features(features)
        
        ok = counts['valleys'] >= 1
        if not ok:
            self.warnings.append(f"VALLEY: expected ≥1, got {counts['valleys']}")
        
        self._log("PASSED" if ok else "FAILED")
        return ok
    
    def test_saddle(self) -> bool:
        """Three peaks: center tall, two shorter sides."""
        self._log("Testing saddle detection...")
        
        gen = SyntheticTerrain(self.config, size_px=128)
        half = (128 * self.config.horizontal_scale) / 2
        gen.add_peak(0.0, 0.0, height_m=60.0, radius_m=10.0)
        gen.add_peak(-half * 0.5, 0.0, height_m=40.0, radius_m=8.0)
        gen.add_peak(half * 0.5, 0.0, height_m=40.0, radius_m=8.0)
        result = gen.build()
        
        features = self._run_layer3(result.heightmap)
        counts = self._count_features(features)
        
        # Expect at least 2 peaks and 1 saddle
        ok = counts['peaks'] >= 2 and counts['saddles'] >= 1
        if not ok:
            self.warnings.append(f"SADDLE: expected ≥2 peaks, ≥1 saddle, got {counts['peaks']} peaks, {counts['saddles']} saddles")
        
        self._log("PASSED" if ok else "FAILED")
        return ok
    
    def test_flat(self) -> bool:
        """Flat terrain with gentle tilt."""
        self._log("Testing flat detection...")
        
        gen = SyntheticTerrain(self.config, size_px=128)
        # No additional features, just base tilt
        result = gen.build()
        
        features = self._run_layer3(result.heightmap)
        counts = self._count_features(features)
        
        # Expect no peaks/ridges/saddles, but should detect flat zone
        ok = (counts['peaks'] == 0 and 
              counts['ridges'] == 0 and 
              counts['saddles'] == 0 and 
              counts['flat_zones'] >= 1)
        
        if not ok:
            self.warnings.append(f"FLAT: expected no peaks/ridges/saddles, 1+ flat zone, got peaks={counts['peaks']}, ridges={counts['ridges']}, saddles={counts['saddles']}, flat={counts['flat_zones']}")
        
        self._log("PASSED" if ok else "FAILED")
        return ok
    
    def run(self) -> bool:
        """Run all tests. Returns True if all pass."""
        self._log("Running...")
        
        tests = [
            self.test_peak,
            self.test_ridge,
            self.test_valley,
            self.test_saddle,
            self.test_flat,
        ]
        
        passed = 0
        for test in tests:
            if test():
                passed += 1
        
        self._log(f"Result: {passed}/{len(tests)} passed")
        
        if self.warnings:
            for w in self.warnings:
                self._log(f"  - {w}")
               
        self.config.exclude_below_reference = True
        return passed == len(tests)
