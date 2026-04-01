import json
import numpy as np
from dataclasses import dataclass, field
from dataclasses import asdict
from typing import Dict, Any, Tuple, List, Optional
from PIL import Image
from pathlib import Path


from core import Heightmap, PipelineConfig, RawImageInput
from calibration import Layer0_Calibration
from lgeometry import Layer1_LocalGeometry
from rgeometry import Layer2_RegionalGeometry
from topological import Layer3_TopologicalFeatures
from relational import Layer4_Relational

#
# Define Test Outputs
#

CONFIG_INFLUENCE = {
    # Layer 0
    'horizontal_scale': ['elevation_range', 'elevation_center'],
    'vertical_scale': ['elevation_center', 'elevation_min', 'elevation_max'],
    'noise_reduction_sigma': ['elevation_std', 'slope_std_deg', 'curvature_flat_ratio'],
    'sea_level_offset': ['elevation_min', 'elevation_mean'],
    
    # Layer 1
    'flat_threshold_deg': ['aspect_undefined_ratio', 'slope_mean_deg'],
    'gradient_method': ['slope_max_deg', 'slope_std_deg'],
    
    # Layer 2
    'adaptive_epsilon': ['curvature_convex_ratio', 'curvature_flat_ratio'],
    'adaptive_percentile': ['curvature_cylindrical_convex_ratio', 'curvature_saddle_ratio'],
    'curvature_epsilon_h_factor': ['curvature_convex_ratio', 'curvature_flat_ratio'],
    'curvature_epsilon_k_factor': ['curvature_saddle_ratio'],
    
    # Layer 3
    'peak_confidence': ['peak_count'],
    'peak_nms_radius_px': ['peak_count'],
    'peak_min_prominence_m': ['peak_count'],
    'min_ridge_length_px': ['ridge_count'],
    'min_valley_length_px': ['valley_count'],
    'saddle_confidence_threshold': ['saddle_count'],
    'saddle_k_min_threshold': ['saddle_count'],
    'min_flat_zone_size_px': ['flat_zone_count'],
    'flat_zone_slope_threshold_deg': ['flat_zone_count'],
}

@dataclass
class MetricRange:
    """Expected range for a metric, with tolerance."""
    min: float
    max: float
    tolerance: float = 0.0  # suggestion: soft boundary for warnings
    
    def contains(self, value: float) -> Tuple[bool, str]:
        if self.min - self.tolerance <= value <= self.max + self.tolerance:
            if self.min <= value <= self.max:
                return True, f"{value:.3f} = [{self.min}, {self.max}]"
            else:
                return True, f"{value:.3f} = [{self.min-self.tolerance}, {self.max+self.tolerance}] (tolerance)"
        return False, f"{value:.3f} != [{self.min}, {self.max}]"

@dataclass
class TuningSuggestion:
    """Actionable recommendation for config adjustment."""
    parameter: str
    current_value: float
    suggested_value: float
    direction: str  # "increase", "decrease"
    reason: str
    confidence: float  # 0-1 how certain this will fix

@dataclass
class TestCase:
    """Ground truth definition for a synthetic terrain test."""
    name: str                      # [t_peak, t_flat, t_ridge, t_saddle]
    png_path: str                  # path: '.\assets\t_<feature>.png'
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    
    # Expected outputs per layer
    layer0: Dict[str, MetricRange] = field(default_factory=dict)  # elevation stats
    layer1: Dict[str, MetricRange] = field(default_factory=dict)  # slope/aspect stats
    layer2: Dict[str, MetricRange] = field(default_factory=dict)  # curvature stats
    layer3: Dict[str, Any] = field(default_factory=dict)          # feature counts/locations
    
    # Ground truth feature specs (for Layer 3 validation)
    expected_features: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Example: {'peaks': {'count': 1, 'centroid_px': (64, 64), 'prominence_m': (45, 55)}}

def diagnose_failure(case_name: str, errors: List[str]) -> List[TuningSuggestion]:
    """Map specific errors to config adjustments."""
    
    suggestions = []
    
    # Peak case diagnostics
    if case_name == "t_peak":
        if any("peak_count" in e for e in errors):
            if "expected 1, got 0" in str(errors):
                suggestions.append(TuningSuggestion(
                    parameter='peak_confidence',
                    current_value=None,
                    suggested_value=15.0,
                    direction='decrease',
                    reason='No peak detected - confidence threshold too high',
                    confidence=0.9
                ))
            elif "expected 1, got >1" in str(errors):
                suggestions.append(TuningSuggestion(
                    parameter='peak_nms_radius_px',
                    current_value=None,
                    suggested_value=60,  # Larger radius = more suppression
                    direction='increase',
                    reason='Multiple peaks detected - NMS radius too small',
                    confidence=0.85
                ))
        
        if any("prominence" in e for e in errors):
            suggestions.append(TuningSuggestion(
                parameter='peak_min_prominence_m',
                current_value=None,
                suggested_value=5.0,
                direction='decrease',
                reason='Peak detected but prominence below threshold',
                confidence=0.8
            ))
    
    # Ridge case diagnostics
    if case_name == "t_ridge":
        if any("ridge_count" in e for e in errors):
            if "expected 1, got 0" in str(errors):
                suggestions.append(TuningSuggestion(
                    parameter='curvature_epsilon',
                    current_value=None,
                    suggested_value=0.00005,
                    direction='decrease',
                    reason='Ridge not detected - curvature threshold may be too strict',
                    confidence=0.7
                ))
                suggestions.append(TuningSuggestion(
                    parameter='curvature_epsilon_h_factor',
                    current_value=None,
                    suggested_value=0.20,
                    direction='decrease',
                    reason='Ridge not detected - mean curvature factor too high',
                    confidence=0.65
                ))
            elif "ridge_length" in str(e):
                suggestions.append(TuningSuggestion(
                    parameter='min_ridge_length_px',
                    current_value=None,
                    suggested_value=10,
                    direction='decrease',
                    reason='Ridge too short - min length threshold too high',
                    confidence=0.85
                ))
    
    # Saddle case diagnostics
    if case_name == "t_saddle":
        if any("saddle_count" in e for e in errors):
            suggestions.append(TuningSuggestion(
                parameter='saddle_confidence_threshold',
                current_value=None,
                suggested_value=0.25,
                direction='decrease',
                reason='Saddles missed - confidence threshold too high',
                confidence=0.75
            ))
            suggestions.append(TuningSuggestion(
                parameter='saddle_k_min_threshold',
                current_value=None,
                suggested_value=0.00015,
                direction='decrease',
                reason='Saddles missed - curvature threshold may be too high',
                confidence=0.7
            ))
    
    # Flat case diagnostics
    if case_name == "t_flat":
        if any("flat_zone_count" in e for e in errors):
            if "expected 1, got 0" in str(errors):
                suggestions.append(TuningSuggestion(
                    parameter='min_flat_zone_size_px',
                    current_value=None,
                    suggested_value=50,
                    direction='decrease',
                    reason='Flat zone not detected - minimum area too large',
                    confidence=0.9
                ))
        if any("slope_mean_deg" in e and ">" in str(e)):
            suggestions.append(TuningSuggestion(
                parameter='flat_zone_slope_threshold_deg',
                current_value=None,
                suggested_value=3.0,
                direction='decrease',
                reason='Slope too high for flat zone - threshold may be too permissive',
                confidence=0.8
            ))
        if any("elevation_std" in e and ">" in str(e)):
            suggestions.append(TuningSuggestion(
                parameter='noise_reduction_sigma',
                current_value=None,
                suggested_value=2.0,
                direction='increase',
                reason='Elevation variance too high - increase blur to flatten noise',
                confidence=0.7
            ))
    
    return suggestions

def make_peak_case(png_path: str = "t_peak.png") -> TestCase:
    """Single Gaussian peak at center, height=50m, radius=10m."""
    return TestCase(
        name="t_peak",
        png_path=png_path,
        config_overrides={'vertical_scale': 0.492, 'noise_reduction_sigma': 1.0},
        
        # Layer 0: elevation after calibration
        layer0={
            'elevation_center': MetricRange(24.0, 25.0),  # 50 grayscale × 0.492 ≈ 24.6m
            'elevation_min': MetricRange(0.0, 0.5),        # sea_level_offset=0.1
            'elevation_max': MetricRange(24.0, 25.5),
        },
        
        # Layer 1: slope statistics
        layer1={
            'slope_mean_deg': MetricRange(5.0, 15.0),
            'slope_max_deg': MetricRange(40.0, 80.0),      # Gaussian edges are steep
            'aspect_undefined_ratio': MetricRange(0.7, 0.95),  # Most of map is flat
        },
        
        # Layer 2: curvature statistics
        layer2={
            'curvature_convex_ratio': MetricRange(0.01, 0.05),  # Peak top is convex
            'curvature_flat_ratio': MetricRange(0.90, 0.98),    # Rest is flat
            'k_magnitude_peak': MetricRange(0.001, 0.01),       # At peak center
        },
        
        # Layer 3: feature detection
        layer3={
            'peak_count': MetricRange(1, 1),
            'ridge_count': MetricRange(0, 0),
            'saddle_count': MetricRange(0, 0),
        },
        
        # Ground truth for feature validation
        expected_features={
            'peaks': {
                'count': 1,
                'centroid_px': (64, 64),
                'centroid_tolerance_px': 3,
                'prominence_m': MetricRange(20.0, 30.0),  # After blur attenuation
            }
        }
    )


def make_ridge_case(png_path: str = "t_ridge.png") -> TestCase:
    """Horizontal ridge, 90% map width, height=40m."""
    return TestCase(
        name="t_ridge",
        png_path=png_path,
        config_overrides={'vertical_scale': 0.492},
        
        layer0={
            'elevation_ridge_crest': MetricRange(19.0, 20.5),  # 40 × 0.492 ≈ 19.7m
        },
        
        layer1={
            'slope_mean_deg': MetricRange(8.0, 20.0),
            'slope_along_ridge_std': MetricRange(0.0, 2.0),  # Should be uniform
        },
        
        layer2={
            'curvature_cylindrical_convex_ratio': MetricRange(0.02, 0.10),
        },
        
        layer3={
            'ridge_count': MetricRange(1, 2),  # NMS may split long ridges
            'ridge_length_px': MetricRange(100, 120),  # ~90% of 128px
        },
        
        expected_features={
            'ridges': {
                'count': 1,
                'orientation': 'horizontal',
                'length_px': MetricRange(100, 120),
            }
        }
    )


def make_saddle_case(png_path: str = "t_saddle.png") -> TestCase:
    """Central high peak + two lower side peaks = 2 saddles."""
    return TestCase(
        name="t_saddle",
        png_path=png_path,
        config_overrides={
            'vertical_scale': 0.492,
            'saddle_confidence_threshold': 0.50,  # Lower for synthetic
        },
        
        layer3={
            'peak_count': MetricRange(3, 3),
            'saddle_count': MetricRange(2, 2),    # One between each pair
        },
        
        expected_features={
            'peaks': {'count': 3},
            'saddles': {
                'count': 2,
                'locations_px': [(52, 64), (76, 64)],  # Approximate
                'location_tolerance_px': 5,
            }
        }
    )


def make_flat_case(png_path: str = "t_flat.png") -> TestCase:
    """Uniform flat zone - baseline test."""
    return TestCase(
        name="t_flat",
        png_path=png_path,
        config_overrides={},
        
        layer0={
            'elevation_std': MetricRange(0.0, 0.1),  # should be uniform
        },
        
        layer1={
            'slope_mean_deg': MetricRange(0.0, 0.5),
            'aspect_undefined_ratio': MetricRange(0.99, 1.0),
        },
        
        layer2={
            'curvature_flat_ratio': MetricRange(0.99, 1.0),
        },
        
        layer3={
            'peak_count': MetricRange(0, 0),
            'ridge_count': MetricRange(0, 0),
            'flat_zone_count': MetricRange(1, 1),
        },
        
        expected_features={
            'flat_zones': {
                'count': 1,
                'area_px': MetricRange(16000, 16384),  # ~128×128
            }
        }
    )


def get_all_cases() -> list:
    return [
        make_flat_case('./assets/t_flat.png'),
        make_peak_case('./assets/t_peak.png'),
        make_ridge_case('./assets/t_ridge.png'),
        make_saddle_case('./assets/t_saddle.png'),
    ]

#
# Metrics
#

def extract_layer0_metrics(heightmap: Heightmap) -> Dict[str, float]:
    """Metrics for Layer 0 output."""
    z = heightmap.data
    h, w = z.shape
    center = z[h//2, w//2]
    
    return {
        'elevation_min': float(np.min(z)),
        'elevation_max': float(np.max(z)),
        'elevation_mean': float(np.mean(z)),
        'elevation_std': float(np.std(z)),
        'elevation_center': float(center),
        'elevation_range': float(np.max(z) - np.min(z)),
    }


def extract_layer1_metrics(slope: np.ndarray, aspect: np.ndarray) -> Dict[str, float]:
    """Metrics for Layer 1 output."""
    flat_threshold = 0.5  # degrees
    aspect_undefined = np.sum(slope < flat_threshold) / slope.size
    
    return {
        'slope_mean_deg': float(np.mean(slope)),
        'slope_std_deg': float(np.std(slope)),
        'slope_max_deg': float(np.max(slope)),
        'aspect_undefined_ratio': float(aspect_undefined),
    }


def extract_layer2_metrics(curvature_type: np.ndarray, 
                          gaussian_curvature: np.ndarray) -> Dict[str, float]:
    """Metrics for Layer 2 output."""
    from core import CurvatureType
    
    total = curvature_type.size
    
    def count_type(ctype):
        return np.sum(curvature_type == ctype.name) / total
    
    return {
        'curvature_convex_ratio': count_type(CurvatureType.CONVEX),
        'curvature_concave_ratio': count_type(CurvatureType.CONCAVE),
        'curvature_saddle_ratio': count_type(CurvatureType.SADDLE),
        'curvature_flat_ratio': count_type(CurvatureType.FLAT),
        'curvature_cylindrical_convex_ratio': count_type(CurvatureType.CYLINDRICAL_CONVEX),
        'k_magnitude_mean': float(np.mean(np.abs(gaussian_curvature))),
        'k_magnitude_max': float(np.max(np.abs(gaussian_curvature))),
    }


def extract_layer3_metrics(features: list) -> Dict[str, Any]:
    """Metrics for Layer 3 output."""
    from core import PeakFeature, RidgeFeature, ValleyFeature, SaddleFeature, FlatZoneFeature
    
    # Calculate ridge lengths
    ridge_lengths = []
    for f in features:
        if isinstance(f, RidgeFeature) and hasattr(f, 'spine_points'):
            ridge_lengths.append(len(f.spine_points))
    
    # Calculate valley lengths
    valley_lengths = []
    for f in features:
        if isinstance(f, ValleyFeature) and hasattr(f, 'spine_points'):
            valley_lengths.append(len(f.spine_points))
    
    return {
        'peak_count': sum(1 for f in features if isinstance(f, PeakFeature)),
        'ridge_count': sum(1 for f in features if isinstance(f, RidgeFeature)),
        'valley_count': sum(1 for f in features if isinstance(f, ValleyFeature)),
        'saddle_count': sum(1 for f in features if isinstance(f, SaddleFeature)),
        'flat_zone_count': sum(1 for f in features if isinstance(f, FlatZoneFeature)),
        'ridge_length_px': np.mean(ridge_lengths) if ridge_lengths else 0.0,
        'valley_length_px': np.mean(valley_lengths) if valley_lengths else 0.0,
        'total_features': len(features),
    }

#
# Tool Class
#

class TestRunner:
    """
    Standalone test runner for synthetic terrain validation.
    
    Usage:
        runner = TestRunner(config)
        results = runner.run_all_cases()
        runner.report(results)
    """
    
    def __init__(self, base_config: PipelineConfig):
        self.base_config = base_config
        self.results: List[Dict] = []
    
    def _apply_overrides(self, overrides: Dict[str, Any]) -> PipelineConfig:
        """Create config with test-specific overrides."""
        import dataclasses
        config = dataclasses.replace(self.base_config, **overrides)
        return config
    
    def _load_test_image(self, png_path: str) -> np.ndarray:
        """Load grayscale test image."""
        path = Path(png_path)
        if not path.exists():
            # Try root folder fallback
            path = Path.cwd() / png_path
        if not path.exists():
            raise FileNotFoundError(f"Test image not found: {png_path}")
        
        return np.array(Image.open(path).convert('L'))
    
    def _run_layer0(self, raw: np.ndarray, config: PipelineConfig) -> Tuple[bool, Dict]:
        """Execute Layer 0 and validate metrics."""
        layer0 = Layer0_Calibration(config)
        heightmap = layer0.execute(RawImageInput(data=raw))
        
        metrics = extract_layer0_metrics(heightmap)
        return True, {'heightmap': heightmap, 'metrics': metrics}
    
    def _run_layer1(self, heightmap, config: PipelineConfig) -> Tuple[bool, Dict]:
        """Execute Layer 1 and validate metrics."""
        layer1 = Layer1_LocalGeometry(config)
        result = layer1.execute(heightmap)
        metrics = extract_layer1_metrics(result['slope'], result['aspect'])
        return True, {'slope': result['slope'], 'aspect': result['aspect'], 'metrics': metrics}
        
        
    def _run_layer2(self, heightmap, config: PipelineConfig) -> Tuple[bool, Dict]:
        """Execute Layer 2 and validate metrics."""
        layer2 = Layer2_RegionalGeometry(config)
        result = layer2.execute(heightmap)
        metrics = extract_layer2_metrics(result['curvature_type'], result['gaussian_curvature'])
        return True, {'metrics': metrics, 
                      'curvature': result['curvature'],
                      'curvature_type': result['curvature_type'],
                      'gaussian_curvature': result['gaussian_curvature']}
    
    def _run_layer3(self, bundle: dict, config: PipelineConfig) -> Tuple[bool, Dict]:
        """Execute Layer 3 and validate metrics."""
        layer3 = Layer3_TopologicalFeatures(config)
        features = layer3.execute(bundle)
        metrics = extract_layer3_metrics(features)
        return True, {'features': features, 'metrics': metrics}
    
    def _validate_metrics(self, actual: Dict[str, float], 
                         expected: Dict[str, MetricRange]) -> List[str]:
        """Compare actual metrics against expected ranges."""
        errors = []
        for name, metric_range in expected.items():
            if name not in actual:
                errors.append(f"Missing metric: {name}")
                continue
            ok, msg = metric_range.contains(actual[name])
            if not ok:
                errors.append(f"{name}: {msg}")
        return errors
    
    def _validate_features(self, features: list, 
                      expected: Dict[str, Dict]) -> List[str]:
        """Validate detected features against ground truth."""
        errors = []
        from core import PeakFeature, RidgeFeature, SaddleFeature, ValleyFeature, FlatZoneFeature
        
        for ftype, specs in expected.items():
            if ftype == 'peaks':
                detected = [f for f in features if isinstance(f, PeakFeature)]
                if 'count' in specs:
                    if len(detected) != specs['count']:
                        errors.append(f"Peak count: expected {specs['count']}, got {len(detected)}")
                if 'centroid_px' in specs and detected:
                    expected_cx, expected_cy = specs['centroid_px']
                    tol = specs.get('centroid_tolerance_px', 3)
                    actual_cx, actual_cy = detected[0].centroid
                    if abs(actual_cx - expected_cx) > tol or abs(actual_cy - expected_cy) > tol:
                        errors.append(f"Peak centroid: expected ({expected_cx},{expected_cy}), got ({actual_cx},{actual_cy})")
            
            elif ftype == 'ridges':
                detected = [f for f in features if isinstance(f, RidgeFeature)]
                if 'count' in specs:
                    if len(detected) != specs['count']:
                        errors.append(f"Ridge count: expected {specs['count']}, got {len(detected)}")
                if 'length_px' in specs and detected:
                    actual_length = len(detected[0].spine_points) if detected[0].spine_points else 0
                    ok, msg = specs['length_px'].contains(actual_length)
                    if not ok:
                        errors.append(f"Ridge length: {msg}")
            
            elif ftype == 'saddles':
                detected = [f for f in features if isinstance(f, SaddleFeature)]
                if 'count' in specs:
                    if len(detected) != specs['count']:
                        errors.append(f"Saddle count: expected {specs['count']}, got {len(detected)}")
                if 'locations_px' in specs and detected:
                    tol = specs.get('location_tolerance_px', 5)
                    for expected_xy in specs['locations_px']:
                        found = False
                        for saddle in detected:
                            ex, ey = expected_xy
                            ax, ay = saddle.centroid
                            if abs(ax - ex) <= tol and abs(ay - ey) <= tol:
                                found = True
                                break
                        if not found:
                            errors.append(f"Saddle location: expected near {expected_xy}, not found within {tol}px")
            
            elif ftype == 'flat_zones':
                detected = [f for f in features if isinstance(f, FlatZoneFeature)]
                if 'count' in specs:
                    if len(detected) != specs['count']:
                        errors.append(f"Flat zone count: expected {specs['count']}, got {len(detected)}")
                if 'area_px' in specs and detected:
                    ok, msg = specs['area_px'].contains(detected[0].area_pixels)
                    if not ok:
                        errors.append(f"Flat zone area: {msg}")
        
        return errors
    
    def run_case(self, case: TestCase) -> Dict:
        """Run a single test case through all layers."""
        result = {
            'name': case.name,
            'passed': True,
            'errors': [],
            'layer_results': {}
        }
        
        try:
            # Apply config overrides
            config = self._apply_overrides(case.config_overrides)
            
            # Load test image
            raw = self._load_test_image(case.png_path)
            
            # === Layer 0 ===
            ok, l0_out = self._run_layer0(raw, config)
            if ok and case.layer0:
                errors = self._validate_metrics(l0_out['metrics'], case.layer0)
                result['errors'].extend([f"L0/{e}" for e in errors])
            result['layer_results']['layer0'] = l0_out['metrics'] if ok else {}
            
            # === Layer 1 ===
            if ok and 'heightmap' in l0_out:
                ok, l1_out = self._run_layer1(l0_out['heightmap'], config)
                if ok and case.layer1:
                    errors = self._validate_metrics(l1_out['metrics'], case.layer1)
                    result['errors'].extend([f"L1/{e}" for e in errors])
                result['layer_results']['layer1'] = l1_out['metrics'] if ok else {}
            
            # === Layer 2 ===
            if ok and 'heightmap' in l0_out:
                ok, l2_out = self._run_layer2(l0_out['heightmap'], config)
                if ok and case.layer2:
                    errors = self._validate_metrics(l2_out['metrics'], case.layer2)
                    result['errors'].extend([f"L2/{e}" for e in errors])
                result['layer_results']['layer2'] = l2_out['metrics'] if ok else {}
            
            # === Layer 3 ===
            if ok and 'heightmap' in l0_out:
                bundle = {
                    'heightmap': l0_out['heightmap'],
                    'slope': l1_out.get('slope'),
                    'aspect': l1_out.get('aspect'),
                    'curvature': l2_out.get('curvature'),
                    'gaussian_curvature': l2_out.get('gaussian_curvature'),
                    'curvature_type': l2_out.get('curvature_type'),
                }
                ok, l3_out = self._run_layer3(bundle, config)
                if ok:
                    # Validate metrics
                    if case.layer3:
                        errors = self._validate_metrics(l3_out['metrics'], case.layer3)
                        result['errors'].extend([f"L3/{e}" for e in errors])
                    # Validate features
                    if case.expected_features:
                        errors = self._validate_features(l3_out['features'], case.expected_features)
                        result['errors'].extend([f"L3-features/{e}" for e in errors])
                result['layer_results']['layer3'] = {
                    'metrics': l3_out['metrics'] if ok else {},
                    'feature_count': len(l3_out['features']) if ok else 0
                }
            
            result['passed'] = len(result['errors']) == 0
            
        except Exception as e:
            result['passed'] = False
            result['errors'].append(f"Exception: {type(e).__name__}: {e}")
        
        return result
    
    def run_all_cases(self, case_list: List[TestCase] = None) -> List[Dict]:
        """Run all test cases."""
        if case_list is None:
            case_list = get_all_cases()
        
        self.results = []
        for case in case_list:
            print(f"Running {case.name}...")
            result = self.run_case(case)
            self.results.append(result)
            status = "PASS" if result['passed'] else "FAIL"
            print(f"  {status} - {len(result['errors'])} errors")
            for err in result['errors']:
                print(f"    - {err}")
        
        return self.results
    
    def report(self, results: List[Dict] = None) -> None:
        """Print summary report."""
        if results is None:
            results = self.results
        
        print("TEST SUMMARY")
        
        passed = sum(1 for r in results if r['passed'])
        total = len(results)
        
        print(f"Passed: {passed}/{total}")
        
        for r in results:
            status = "OK" if r['passed'] else "FAILED"
            print(f"{status} {r['name']}: {len(r['errors'])} errors")
            if r['errors']:
                for err in r['errors'][:3]:  # Show first 3 errors
                    print(f"    - {err}")
                if len(r['errors']) > 3:
                    print(f"    ... and {len(r['errors'])-3} more")
        
#
# Entry Point
#

if __name__ == '__main__':
    # run tests
    config = PipelineConfig()
    config.verbose = False
    runner = TestRunner(config)
    results = runner.run_all_cases()
    # report
    runner.report()
    # dump
    with open('calibration_log.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)