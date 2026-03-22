"""
Layer 2: Regional Geometry — Invariant Tests + Calibration

Imports from the real pipeline modules (core.py, rgeometry.py).
No local model redefinitions.

Output: [L2|test].......[GOOD/FAIL]
        On FAIL: newline + tab shows expected | obtained | why
Usage: python calibrate_layer2.py
"""

import numpy as np
from typing import Tuple

from core import Heightmap, NormalizationConfig, PipelineConfig
from rgeometry import Layer2_RegionalGeometry


# ─── Synthetic Surface Helpers ───────────────────────────────────────────────
#
# Amplitudes are chosen so that the resulting K sits well above the adaptive
# k_epsilon_min (3.7e-5 by default).  Paraboloid/hyperboloid with a=0.01
# gives K = 4a² = 4e-4, which is ~10× above the minimum.

def _make_heightmap(Z: np.ndarray, h_scale: float = 2.0) -> Heightmap:
    cfg = NormalizationConfig(horizontal_scale=h_scale, vertical_scale=0.1)
    return Heightmap(
        data=Z.astype(np.float32),
        config=cfg,
        pixel_to_world=lambda p: (p[0] * h_scale, p[1] * h_scale),
    )


def _grid(shape: Tuple[int, int], h_scale: float):
    H, W = shape
    x = np.arange(W) * h_scale - W * h_scale / 2
    y = np.arange(H) * h_scale - H * h_scale / 2
    return np.meshgrid(x, y)


def make_paraboloid(shape=(64, 64), h_scale=2.0, a=0.01):
    """z = a(x²+y²)  →  H = 2a, K = 4a²  (convex everywhere)"""
    X, Y = _grid(shape, h_scale)
    return _make_heightmap(a * (X**2 + Y**2), h_scale)


def make_hyperboloid(shape=(64, 64), h_scale=2.0, a=0.01):
    """z = a(x²-y²)  →  H = 0, K = -4a²  (saddle everywhere)"""
    X, Y = _grid(shape, h_scale)
    return _make_heightmap(a * (X**2 - Y**2), h_scale)


def make_sphere_cap(shape=(64, 64), h_scale=2.0, R=100.0):
    """z = √(R²-x²-y²)  →  H = 1/R, K = 1/R²"""
    H, W = shape
    x = np.arange(W) * h_scale - W * h_scale / 2
    y = np.arange(H) * h_scale - H * h_scale / 2
    X, Y = np.meshgrid(x, y)
    mask = X**2 + Y**2 < R**2
    Z = np.zeros_like(X)
    Z[mask] = np.sqrt(R**2 - X[mask]**2 - Y[mask]**2)
    return _make_heightmap(Z, h_scale)


def make_cylinder(shape=(64, 64), h_scale=2.0, a=0.01):
    """z = a·x²  →  K = 0  (curved in x, flat in y — should classify FLAT)"""
    X, Y = _grid(shape, h_scale)
    return _make_heightmap(a * X**2, h_scale)


def make_flat(shape=(64, 64), h_scale=2.0):
    """z = 0  →  H = 0, K = 0"""
    return _make_heightmap(np.zeros(shape), h_scale)


# ─── Tests ───────────────────────────────────────────────────────────────────

def t_analytical_h() -> Tuple[bool, str, str, str]:
    """Mean curvature H on paraboloid matches analytical value."""
    h_scale, a = 2.0, 0.01
    layer = Layer2_RegionalGeometry(PipelineConfig())
    result = layer.execute(make_paraboloid(h_scale=h_scale, a=a))

    analytical_H = 2 * a  # = 0.02
    measured_H = float(np.nanmean(np.abs(result["curvature"][10:-10, 10:-10])))
    err_pct = abs(measured_H - analytical_H) / analytical_H * 100

    ok = err_pct < 5.0  # finite differences should be <1% on a smooth paraboloid
    return (
        ok,
        f"H ≈ {analytical_H:.4f} (±5%)",
        f"H = {measured_H:.6f} (err={err_pct:.1f}%)",
        "Mean curvature diverges from analytical" if not ok else "",
    )


def t_analytical_k() -> Tuple[bool, str, str, str]:
    """Gaussian curvature K on hyperboloid has correct sign and magnitude."""
    h_scale, a = 2.0, 0.01
    layer = Layer2_RegionalGeometry(PipelineConfig())
    result = layer.execute(make_hyperboloid(h_scale=h_scale, a=a))

    analytical_K = -4 * a**2  # = -4e-4
    measured_K = float(np.nanmean(result["gaussian_curvature"][10:-10, 10:-10]))
    sign_ok = np.sign(measured_K) == np.sign(analytical_K)
    mag_err = abs(abs(measured_K) - abs(analytical_K)) / abs(analytical_K) * 100

    ok = sign_ok and mag_err < 10.0
    return (
        ok,
        f"K ≈ {analytical_K:.6f} (sign correct, ±10%)",
        f"K = {measured_K:.8f} (err={mag_err:.1f}%)",
        "Gaussian curvature sign or magnitude wrong" if not ok else "",
    )


def t_analytical_k_sphere() -> Tuple[bool, str, str, str]:
    """Gaussian curvature K on sphere cap is positive and near 1/R²."""
    h_scale, R = 2.0, 100.0
    layer = Layer2_RegionalGeometry(PipelineConfig())
    result = layer.execute(make_sphere_cap(h_scale=h_scale, R=R))

    analytical_K = 1.0 / R**2  # = 1e-4
    interior = result["gaussian_curvature"][10:-10, 10:-10]
    measured_K = float(np.nanmean(interior[~np.isnan(interior)]))
    sign_ok = measured_K > 0
    mag_err = abs(measured_K - analytical_K) / analytical_K * 100 if analytical_K > 0 else 0

    ok = sign_ok and mag_err < 50.0
    return (
        ok,
        f"K > 0, ≈ {analytical_K:.6f} (±50%)",
        f"K = {measured_K:.8f} (err={mag_err:.1f}%)",
        "Sphere cap curvature wrong" if not ok else "",
    )


def t_classification_rules() -> Tuple[bool, str, str, str]:
    """Each canonical surface classifies correctly on reliable interior pixels."""
    h_scale = 2.0
    layer = Layer2_RegionalGeometry(PipelineConfig())

    test_cases = [
        (make_paraboloid(h_scale=h_scale, a=0.01), "CONVEX",  "paraboloid"),
        (make_hyperboloid(h_scale=h_scale, a=0.01), "SADDLE", "hyperboloid"),
        (make_cylinder(h_scale=h_scale, a=0.01),    "FLAT",   "cylinder (K=0)"),
        (make_flat(h_scale=h_scale),                 "FLAT",   "flat plane"),
    ]

    results = []
    for hm, expected_label, desc in test_cases:
        result = layer.execute(hm)
        interior = result["curvature_type"][10:-10, 10:-10].flatten()
        pct = float(np.mean(interior == expected_label) * 100)
        results.append((desc, pct, pct > 70.0))

    all_ok = all(r[2] for r in results)
    summary = "; ".join(f"{r[0]}={r[1]:.0f}%" for r in results)
    return (
        all_ok,
        "each surface >70% correct label (interior pixels)",
        summary,
        "Classification rules broken" if not all_ok else "",
    )


def t_epsilon_scales_with_terrain() -> Tuple[bool, str, str, str]:
    """Adaptive epsilon is larger for steeper terrain."""
    h_scale = 2.0
    layer_steep  = Layer2_RegionalGeometry(PipelineConfig())
    layer_gentle = Layer2_RegionalGeometry(PipelineConfig())

    layer_steep.execute(make_paraboloid(h_scale=h_scale, a=0.05))
    layer_gentle.execute(make_paraboloid(h_scale=h_scale, a=0.001))

    steep_h_eps  = layer_steep._epsilon_used[0]
    gentle_h_eps = layer_gentle._epsilon_used[0]
    ok = steep_h_eps > gentle_h_eps

    return (
        ok,
        "steep terrain → larger h_epsilon than gentle terrain",
        f"steep={steep_h_eps:.6f}, gentle={gentle_h_eps:.6f}",
        "Adaptive epsilon not scaling with terrain amplitude" if not ok else "",
    )


def t_flat_surface_all_flat() -> Tuple[bool, str, str, str]:
    """Zero elevation map classifies entirely as FLAT."""
    layer = Layer2_RegionalGeometry(PipelineConfig())
    result = layer.execute(make_flat())
    flat_pct = float(np.mean(result["curvature_type"] == "FLAT") * 100)
    ok = flat_pct == 100.0
    return (
        ok,
        "100% FLAT on zero-elevation map",
        f"FLAT={flat_pct:.1f}%",
        "Non-FLAT pixels on zero surface" if not ok else "",
    )


def t_output_schema() -> Tuple[bool, str, str, str]:
    """Output dict contains all required keys with correct array shapes."""
    layer = Layer2_RegionalGeometry(PipelineConfig())
    hm = make_paraboloid()
    result = layer.execute(hm)

    required = {"curvature", "gaussian_curvature", "curvature_type"}
    missing = required - set(result.keys())
    shape_ok = (
        result["curvature"].shape == hm.data.shape
        and result["gaussian_curvature"].shape == hm.data.shape
        and result["curvature_type"].shape == hm.data.shape
    )
    ok = not missing and shape_ok
    obtained = (
        f"keys={set(result.keys())}, shapes match={shape_ok}"
        if not missing
        else f"missing keys: {missing}"
    )
    return (
        ok,
        f"keys={required}, shapes match input",
        obtained,
        "Output schema broken" if not ok else "",
    )


def t_epsilon_bounds() -> Tuple[bool, str, str, str]:
    """Adaptive epsilon respects configured minimums."""
    cfg = PipelineConfig()
    layer = Layer2_RegionalGeometry(cfg)
    layer.execute(make_flat())  # flat → anchor = 0 → falls back to minimums

    h_eps, k_eps = layer._epsilon_used
    h_ok = h_eps >= cfg.curvature_epsilon_h_min
    k_ok = k_eps >= cfg.curvature_epsilon_k_min
    ok = h_ok and k_ok
    return (
        ok,
        f"h_eps ≥ {cfg.curvature_epsilon_h_min}, k_eps ≥ {cfg.curvature_epsilon_k_min}",
        f"h_eps={h_eps:.6f}, k_eps={k_eps:.6f}",
        "Epsilon fell below configured minimum" if not ok else "",
    )


# ─── Calibration Output ──────────────────────────────────────────────────────

def _calibration_report(layer: Layer2_RegionalGeometry, hm: Heightmap) -> None:
    """Print epsilon summary with terrain context."""
    if layer._epsilon_used is None:
        print("  (no execute() call recorded)")
        return
    
    h_eps, k_eps = layer._epsilon_used
    
    # Characterize the terrain
    z = hm.data
    elev_range = z.max() - z.min()
    avg_slope = np.mean(np.gradient(z, hm.config.horizontal_scale)[0]**2 + 
                        np.gradient(z, hm.config.horizontal_scale)[1]**2)**0.5
    avg_slope_deg = np.degrees(np.arctan(avg_slope))
    
    # Contextual interpretation
    if elev_range < 10:
        terrain_type = "low-relief (hills, plains)"
    elif elev_range < 50:
        terrain_type = "moderate-relief (rolling hills, valleys)"
    else:
        terrain_type = "high-relief (mountains, deep valleys)"
    
    if avg_slope_deg < 10:
        slope_desc = "gentle"
    elif avg_slope_deg < 25:
        slope_desc = "moderate"
    else:
        slope_desc = "steep"
    
    print(f"\n  Terrain context:")
    print(f"    Type: {terrain_type} (elev range={elev_range:.1f}m)")
    print(f"    Slope: {slope_desc} (avg={avg_slope_deg:.1f}°)")
    print(f"\n  Adaptive epsilon thresholds:")
    print(f"    h_epsilon = {h_eps:.6f} (1/m) — min mean curvature to count as CONVEX/CONCAVE")
    print(f"    k_epsilon = {k_eps:.8f} (1/m²) — min Gaussian curvature to count as SADDLE")
    print(f"\n  Interpretation:")
    if h_eps < 0.005:
        print(f"    • Low h_epsilon → even gentle convex/concave features will be detected")
    elif h_eps < 0.02:
        print(f"    • Moderate h_epsilon → only distinct ridges/valleys will be detected")
    else:
        print(f"    • High h_epsilon → only sharp, well-defined features will be detected")
    
    if k_eps < 1e-4:
        print(f"    • Low k_epsilon → subtle saddles (potential chokepoints) will be detected")
    else:
        print(f"    • High k_epsilon → only pronounced saddles (mountain passes) will be detected")
    
    print(f"\n  Suggested PipelineConfig overrides for this terrain:")
    print(f"    curvature_epsilon_h_min = {h_eps * 0.5:.2e}")
    print(f"    curvature_epsilon_k_min = {k_eps * 0.5:.2e}")


# ─── Runner ──────────────────────────────────────────────────────────────────

def run():
    print("\n[L2] Regional Geometry Tests + Calibration")
    print("-" * 60)

    tests = [
        ("analytical_h",          t_analytical_h),
        ("analytical_k",          t_analytical_k),
        ("analytical_k_sphere",   t_analytical_k_sphere),
        ("classification",        t_classification_rules),
        ("epsilon_scaling",       t_epsilon_scales_with_terrain),
        ("flat_all_flat",         t_flat_surface_all_flat),
        ("output_schema",         t_output_schema),
        ("epsilon_bounds",        t_epsilon_bounds),
    ]

    all_ok = True
    calibration_layer = None
    calibration_hm = None  # ← ADD: store the heightmap used for calibration

    for name, fn in tests:
        ok, exp, got, why = fn()
        status = "GOOD" if ok else "FAIL"
        print(f"[L2|{name:24}].......[{status}]", end="")
        if not ok:
            all_ok = False
            print(f"\n\texpected: {exp}\n\tobtained: {got}\n\twhy:      {why}")
        else:
            print()
        
        if name == "classification":
            h_scale = 2.0
            hm = make_paraboloid(h_scale=h_scale, a=0.01)
            temp_layer = Layer2_RegionalGeometry(PipelineConfig())
            temp_layer.execute(hm)
            calibration_layer = temp_layer
            calibration_hm = hm

    print("\n" + "-" * 60)
    print("[L2|CALIBRATION OUTPUT]")
    print("-" * 60)
    if calibration_layer is not None and calibration_hm is not None:
        _calibration_report(calibration_layer, calibration_hm)
    else:
        print("  (No calibration data available — run classification test first)")
    print("-" * 60)
    print(f"[SUMMARY] {'PASS' if all_ok else 'FAILED'}\n")
    return all_ok


if __name__ == "__main__":
    run()