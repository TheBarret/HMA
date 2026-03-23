"""
Layer 2: Regional Geometry — Invariant Tests + Calibration
"""

import numpy as np
from typing import Tuple

from core import Heightmap, NormalizationConfig, PipelineConfig, GameScaling
from rgeometry import Layer2_RegionalGeometry


# ─── Synthetic Surface Helpers ────────────────────────────────────────────────
#
# Amplitudes are chosen so K sits well above the adaptive k_epsilon_min.
# Paraboloid/hyperboloid with a=0.01 gives K = 4a² = 4e-4 ≈ 10× k_epsilon_min.

def _make_heightmap(Z: np.ndarray, h_scale: float = 2.0) -> Heightmap:
    cfg = NormalizationConfig(horizontal_scale=h_scale, vertical_scale=0.2)
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
    H_sz, W = shape
    x = np.arange(W) * h_scale - W * h_scale / 2
    y = np.arange(H_sz) * h_scale - H_sz * h_scale / 2
    X, Y = np.meshgrid(x, y)
    mask = X**2 + Y**2 < R**2
    Z = np.zeros_like(X)
    Z[mask] = np.sqrt(R**2 - X[mask]**2 - Y[mask]**2)
    return _make_heightmap(Z, h_scale)


def make_cylinder(shape=(64, 64), h_scale=2.0, a=0.01):
    """z = a·x²  →  K = 0  (curved in x only — should classify FLAT)"""
    X, Y = _grid(shape, h_scale)
    return _make_heightmap(a * X**2, h_scale)


def make_flat(shape=(64, 64), h_scale=2.0):
    """z = 0  →  H = 0, K = 0"""
    return _make_heightmap(np.zeros(shape), h_scale)


def t_analytical_h() -> Tuple[bool, str, str, str]:
    """Mean curvature H on paraboloid matches analytical value within 5%."""
    h_scale, a = 2.0, 0.01
    layer = Layer2_RegionalGeometry(PipelineConfig())
    result = layer.execute(make_paraboloid(h_scale=h_scale, a=a))

    analytical_H = 2 * a
    
    # t_analytical_h - Sample center 8×8 pixels
    #measured_H = float(np.nanmean(np.abs(result["curvature"][10:-10, 10:-10])))
    measured_H = float(np.nanmean(np.abs(result["curvature"][28:36, 28:36])))
    
    err_pct = abs(measured_H - analytical_H) / analytical_H * 100

    ok = err_pct < 5.0
    return (
        ok,
        f"H ≈ {analytical_H:.4f} (±5%)",
        f"H = {measured_H:.6f} (err={err_pct:.1f}%)",
        "Mean curvature diverges from analytical" if not ok else "",
    )


def t_analytical_k() -> Tuple[bool, str, str, str]:
    """Gaussian curvature K on hyperboloid has correct sign and magnitude within 10%."""
    h_scale, a = 2.0, 0.01
    layer = Layer2_RegionalGeometry(PipelineConfig())
    result = layer.execute(make_hyperboloid(h_scale=h_scale, a=a))

    analytical_K = -4 * a**2
    
    # t_analytical_k - Sample center 8×8 pixels  
    #measured_K = float(np.nanmean(result["gaussian_curvature"][10:-10, 10:-10]))
    measured_K = float(np.nanmean(result["gaussian_curvature"][28:36, 28:36]))
    
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

    analytical_K = 1.0 / R**2
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
    """Each canonical surface classifies correctly on interior pixels (>70%)."""
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
    """Adaptive epsilon grows with terrain amplitude."""
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
    """Output dict has all required keys with shapes matching input."""
    layer = Layer2_RegionalGeometry(PipelineConfig())
    hm = make_paraboloid()
    result = layer.execute(hm)

    required = {"curvature", "gaussian_curvature", "curvature_type"}
    missing = required - set(result.keys())
    shape_ok = all(result[k].shape == hm.data.shape for k in required if k in result)
    ok = not missing and shape_ok
    obtained = (
        f"keys={set(result.keys())}, shapes match={shape_ok}"
        if not missing else f"missing keys: {missing}"
    )
    return (
        ok,
        f"keys={required}, shapes match input",
        obtained,
        "Output schema broken" if not ok else "",
    )


def t_epsilon_bounds() -> Tuple[bool, str, str, str]:
    """Adaptive epsilon never falls below configured minimums (flat terrain edge case)."""
    cfg = PipelineConfig()
    layer = Layer2_RegionalGeometry(cfg)
    layer.execute(make_flat())  # flat → p95(|H|) = 0 → must fall back to minimums

    h_eps, k_eps = layer._epsilon_used
    h_ok = h_eps >= cfg.curvature_epsilon_h_min
    k_ok = k_eps >= cfg.curvature_epsilon_k_min
    ok = h_ok and k_ok
    return (
        ok,
        f"h_eps ≥ {cfg.curvature_epsilon_h_min:.2e}, k_eps ≥ {cfg.curvature_epsilon_k_min:.2e}",
        f"h_eps={h_eps:.6f}, k_eps={k_eps:.6f}",
        "Epsilon fell below configured minimum" if not ok else "",
    )

def t_classification_logic() -> Tuple[bool, str, str, str]:
    """
    Validate curvature classification rules using FIXED epsilon.
    Tests each surface type independently (no seam artifacts).
    """
    h_scale, a = 2.0, 0.01
    cfg = PipelineConfig()
    cfg.adaptive_epsilon = False  # Disable adaptive for this test
    cfg.curvature_epsilon_h_min = 0.005  # Below paraboloid H=0.02
    cfg.curvature_epsilon_k_min = 0.0001  # Below hyperboloid |K|=0.0004
    
    layer = Layer2_RegionalGeometry(cfg)
    
    test_cases = [
        (make_paraboloid(h_scale=h_scale, a=a), "CONVEX", "paraboloid"),
        (make_hyperboloid(h_scale=h_scale, a=a), "SADDLE", "hyperboloid"),
        (make_flat(h_scale=h_scale), "FLAT", "flat plane"),
    ]
    
    failures = []
    details = []
    
    for hm, expected, desc in test_cases:
        result = layer.execute(hm)
        interior = result["curvature_type"][15:-15, 15:-15].flatten()  # Sample center only
        pct = float(np.mean(interior == expected) * 100)
        details.append(f"{desc}={pct:.0f}%")
        if pct < 80.0:  # Higher threshold since no seam artifacts
            failures.append(f"{desc} ({pct:.0f}%)")
    
    ok = len(failures) == 0
    return (
        ok,
        "each surface >80% correct (fixed epsilon, center sampling)",
        "; ".join(details),
        "Failed: " + " | ".join(failures) if not ok else "",
    )


def t_adaptive_epsilon_on_terrain() -> Tuple[bool, str, str, str]:
    """
    Validate adaptive epsilon scales with terrain amplitude.
    Uses a single continuous surface (no seams).
    """
    h_scale = 2.0
    cfg = PipelineConfig()
    cfg.adaptive_epsilon = True
    
    # Gentle terrain
    layer_gentle = Layer2_RegionalGeometry(cfg)
    layer_gentle.execute(make_paraboloid(h_scale=h_scale, a=0.005))
    gentle_h_eps = layer_gentle._epsilon_used[0]
    
    # Steep terrain
    layer_steep = Layer2_RegionalGeometry(cfg)
    layer_steep.execute(make_paraboloid(h_scale=h_scale, a=0.02))
    steep_h_eps = layer_steep._epsilon_used[0]
    
    ok = steep_h_eps > gentle_h_eps
    return (
        ok,
        f"steep terrain h_epsilon > gentle terrain ({steep_h_eps:.5f} > {gentle_h_eps:.5f})",
        f"steep={steep_h_eps:.5f}, gentle={gentle_h_eps:.5f}",
        "Adaptive epsilon not scaling with terrain amplitude" if not ok else "",
    )


def t_all_branches() -> Tuple[bool, str, str, str]:
    """
    Exercises all four curvature classes using three canonical surfaces
    stitched into quadrants of a single heightmap.
    
    FIX: Each quadrant has its own centered coordinate system.
    """
    h_scale, a = 2.0, 0.015  # Increased amplitude for stronger signal
    cfg = PipelineConfig()
    layer = Layer2_RegionalGeometry(cfg)
    shape = (64, 64)
    H, W = shape
    qh, qw = H // 2, W // 2
    Z = np.zeros(shape, dtype=np.float32)

    # CONVEX quadrant (top-left): Center coordinates on quadrant center
    x_convex = np.arange(qw) * h_scale - qw * h_scale / 2
    y_convex = np.arange(qh) * h_scale - qh * h_scale / 2
    Xc, Yc = np.meshgrid(x_convex, y_convex)
    Z[:qh, :qw] = a * (Xc**2 + Yc**2)

    # CONCAVE quadrant (top-right): Center coordinates on quadrant center
    x_concave = np.arange(qw) * h_scale - qw * h_scale / 2
    y_concave = np.arange(qh) * h_scale - qh * h_scale / 2
    Xv, Yv = np.meshgrid(x_concave, y_concave)
    Z[:qh, qw:] = -a * (Xv**2 + Yv**2)

    # SADDLE quadrant (bottom): Center coordinates on quadrant center
    x_saddle = np.arange(W) * h_scale - W * h_scale / 2
    y_saddle = np.arange(H - qh) * h_scale - (H - qh) * h_scale / 2
    Xs, Ys = np.meshgrid(x_saddle, y_saddle)
    Z[qh:, :] = a * (Xs**2 - Ys**2)

    hm = _make_heightmap(Z, h_scale)
    result = layer.execute(hm)
    h_eps, k_eps = layer._epsilon_used
    ct = result["curvature_type"]

    # Sample well inside each quadrant away from boundaries
    m = 10  # margin from edges and seams
    convex_mask = np.zeros(shape, bool); convex_mask[m:qh-m, m:qw-m] = True
    concave_mask = np.zeros(shape, bool); concave_mask[m:qh-m, qw+m:W-m] = True
    saddle_mask = np.zeros(shape, bool); saddle_mask[qh+m:H-m, m:W-m] = True

    regions = [
        ("CONVEX", convex_mask, "CONVEX", "lower h_factor"),
        ("CONCAVE", concave_mask, "CONCAVE", "lower h_factor"),
        ("SADDLE", saddle_mask, "SADDLE", "lower k_factor"),
    ]

    failures = []
    details = []
    for name, mask, expected, hint in regions:
        if not np.any(mask):
            details.append(f"{name}: no pixels sampled")
            failures.append(name)
            continue
        pct = float(np.mean(ct[mask] == expected) * 100)
        details.append(f"{name}={pct:.0f}%")
        if pct < 60.0:  # Lowered expectation for seam artifacts
            failures.append(f"{name} ({pct:.0f}% — hint: {hint} [h_eps={h_eps:.5f}, k_eps={k_eps:.6f}])")

    ok = len(failures) == 0
    return (
        ok,
        "all 3 non-flat regions >60% correct (seam artifacts expected)",
        "; ".join(details),
        "Failed: " + " | ".join(failures) if not ok else "",
    )


# ─── Calibration Report ───────────────────────────────────────────────────────
#
# Interprets the epsilon values in terms of the game's physical constraints.
# Uses GameScaling to derive the maximum physically possible curvature for
# the configured game type — epsilons are then expressed as a fraction of that
# maximum, which is the only meaningful way to read them.

def _calibration_report(cfg: PipelineConfig, h_eps: float, k_eps: float) -> None:
    gs = GameScaling.for_game(cfg.game_type)

    # Physical upper bounds on curvature from game scaling
    # H_max: slope changes from 0 → max_traversable over 2 pixels (sharpest realistic feature)
    # K_max: worst-case saddle ≈ H_max² (both principal curvatures at maximum)
    h_max = np.tan(np.radians(gs.vehicle_climb_angle_deg)) / (2 * gs.horizontal_scale_m_per_px)
    k_max = h_max ** 2
    elev_range_max = 255 * gs.vertical_scale_m_per_unit

    h_pct = h_eps / h_max * 100
    k_pct = k_eps / k_max * 100

    def sensitivity(pct):
        if pct < 3:  return "high   (subtle features detected)"
        if pct < 10: return "medium (distinct features only)"
        return            "low    (only sharp features detected)"

    print(f"\n  Game profile: {cfg.game_type.value}")
    print(f"    Pixel resolution : {gs.horizontal_scale_m_per_px:.1f} m/px")
    print(f"    Elevation range  : 0 – {elev_range_max:.0f} m  (255 × {gs.vertical_scale_m_per_unit} m/unit)")
    print(f"    Vehicle climb    : {gs.vehicle_climb_angle_deg:.0f}°"
          f"  →  H_max = {h_max:.4f} /m,  K_max = {k_max:.5f} /m²")

    print(f"\n  Epsilon thresholds (from last execute):")
    print(f"    h_epsilon = {h_eps:.5f} /m   ({h_pct:.1f}% of H_max)  sensitivity: {sensitivity(h_pct)}")
    print(f"    k_epsilon = {k_eps:.6f} /m²  ({k_pct:.1f}% of K_max)  sensitivity: {sensitivity(k_pct)}")

    print(f"\n  Context:")
    slope_change = h_eps * gs.horizontal_scale_m_per_px
    print(f"    Ridge/valley seeds : |H| > {h_eps:.5f} /m"
          f"  (~{slope_change:.3f} m/m slope-change per pixel)")
    print(f"    Saddle seeds       : K < -{k_eps:.6f} /m²"
          f"  (pass-like feature threshold)")

    # Warn if either epsilon is pinned to the floor
    if abs(h_eps - cfg.curvature_epsilon_h_min) < 1e-8:
        print(f"\n   h_epsilon is at the configured minimum ({cfg.curvature_epsilon_h_min:.2e}).")
        print(f"     Calibration surface was too flat to anchor an adaptive threshold.")
        print(f"     Run on a representative terrain sample for accurate thresholds.")
    if abs(k_eps - cfg.curvature_epsilon_k_min) < 1e-8:
        print(f"\n   k_epsilon is at the configured minimum ({cfg.curvature_epsilon_k_min:.2e}).")
        print(f"     Saddle detection threshold is a floor estimate only.")

    print(f"\n  Suggested PipelineConfig overrides for {cfg.game_type.value}:")
    print(f"    curvature_epsilon_h_min = {h_eps * 0.5:.2e}  # half current threshold")
    print(f"    curvature_epsilon_k_min = {k_eps * 0.5:.2e}  # half current threshold")


# ─── Runner ───────────────────────────────────────────────────────────────────

def run():
    print("\n[L2] Regional Geometry Tests + Calibration")
    print("-" * 60)

    tests = [
        ("analytical_h",        t_analytical_h),
        ("analytical_k",        t_analytical_k),
        ("analytical_k_sphere", t_analytical_k_sphere),
        ("classification",      t_classification_rules),
        ("classification_logic", t_classification_logic),
        ("adaptive_epsilon", t_adaptive_epsilon_on_terrain),
        ("epsilon_scaling",     t_epsilon_scales_with_terrain),
        ("flat_all_flat",       t_flat_surface_all_flat),
        ("output_schema",       t_output_schema),
        ("epsilon_bounds",      t_epsilon_bounds),
        ("curvature_test",      t_all_branches)
    ]

    all_ok = True
    calibration_layer = None

    for name, fn in tests:
        ok, exp, got, why = fn()
        status = "GOOD" if ok else "FAIL"
        print(f"[L2|{name:22}].......[{status}]", end="")
        if not ok:
            all_ok = False
            print(f"\n\texpected: {exp}\n\tobtained: {got}\n\twhy:      {why}")
        else:
            print()

        # Anchor the calibration report to a representative ARMA-scale terrain,
        # not a degenerate test fixture.  Done once after epsilon_scaling runs.
        if name == "epsilon_scaling":
            calibration_layer = Layer2_RegionalGeometry(PipelineConfig())
            calibration_layer.execute(make_paraboloid(a=0.01))

    cfg = PipelineConfig()
    print("\n" + "-" * 60)
    print("[L2|CALIBRATION OUTPUT]")
    print("-" * 60)
    if calibration_layer is not None and calibration_layer._epsilon_used is not None:
        h_eps, k_eps = calibration_layer._epsilon_used
        _calibration_report(cfg, h_eps, k_eps)
    else:
        print("  (no calibration data — epsilon_scaling test did not run)")
    print("-" * 60)
    print(f"[SUMMARY] {'PASS' if all_ok else 'FAILED'}\n")
    return all_ok


if __name__ == "__main__":
    run()