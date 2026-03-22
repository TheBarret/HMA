"""
Layer 1: Local Geometry — Invariant Tests
"""

import numpy as np
from typing import Tuple

from core import Heightmap, NormalizationConfig, PipelineConfig
from lgeometry import Layer1_LocalGeometry


# ─── Synthetic Surface Helpers (Test Fixtures) ───────────────────────────────

def _make_heightmap(Z: np.ndarray, h_scale: float = 2.0) -> Heightmap:
    """Create a Heightmap from elevation array."""
    cfg = NormalizationConfig(horizontal_scale=h_scale, vertical_scale=0.2, sea_level_offset=0.0)
    return Heightmap(
        data=Z.astype(np.float32),
        config=cfg,
        pixel_to_world=lambda p: (p[0] * h_scale, p[1] * h_scale),
    )


def make_plane(shape=(64, 64), h_scale=2.0, slope_deg=15.0, aspect_deg=45.0) -> Heightmap:
    """
    Planar surface with known slope and aspect.
    
    z = ax + by where:
      slope = arctan(√(a²+b²))
      aspect = atan2(a, b)  (0° = North, clockwise)
    """
    H, W = shape
    slope_rad = np.radians(slope_deg)
    aspect_rad = np.radians(aspect_deg)
    
    # Gradient components: dz/dx (East), dz/dy (North)
    dz_dx = np.tan(slope_rad) * np.sin(aspect_rad)
    dz_dy = np.tan(slope_rad) * np.cos(aspect_rad)
    
    x = np.arange(W) * h_scale - W * h_scale / 2
    y = np.arange(H) * h_scale - H * h_scale / 2
    X, Y = np.meshgrid(x, y)
    Z = dz_dx * X + dz_dy * Y
    
    return _make_heightmap(Z, h_scale)


def make_cone(shape=(64, 64), h_scale=2.0) -> Heightmap:
    """
    Cone (peak at center) for aspect circularity test.
    Downhill flows outward radially after gradient negation.
    """
    H, W = shape
    cx, cy = W // 2, H // 2
    x = np.arange(W) * h_scale - cx * h_scale
    y = np.arange(H) * h_scale - cy * h_scale
    X, Y = np.meshgrid(x, y)

    R = np.sqrt(X**2 + Y**2)
    Z = R  # peak at center: gradient points outward (uphill), negation gives downhill

    return _make_heightmap(Z, h_scale)


def make_flat(shape=(64, 64), h_scale=2.0, elevation=0.0) -> Heightmap:
    """Constant elevation surface."""
    H, W = shape
    Z = np.full((H, W), elevation, dtype=np.float32)
    return _make_heightmap(Z, h_scale)


# ─── Invariant Tests ─────────────────────────────────────────────────────────

def t_analytical_slope() -> Tuple[bool, str, str, str]:
    """Slope magnitude on plane matches analytical value."""
    h_scale = 2.0
    target_slope = 15.0
    target_aspect = 45.0
    
    hm = make_plane(slope_deg=target_slope, aspect_deg=target_aspect, h_scale=h_scale)
    layer = Layer1_LocalGeometry(PipelineConfig(horizontal_scale=h_scale))
    result = layer.execute(hm)
    
    interior = result["slope"][10:-10, 10:-10]
    mean_slope = float(np.mean(interior))
    err = abs(mean_slope - target_slope)
    
    ok = err < 1.0
    return ok, f"slope={target_slope}° ±1°", f"slope={mean_slope:.2f}° (err={err:.2f}°)", \
           "Gradient calculation or cell_size wrong" if not ok else ""


def t_aspect_circularity() -> Tuple[bool, str, str, str]:
    """Aspect on cone increases monotonically with angle."""
    h_scale = 2.0
    hm = make_cone(h_scale=h_scale)
    layer = Layer1_LocalGeometry(PipelineConfig(horizontal_scale=h_scale))
    result = layer.execute(hm)

    H, W = hm.data.shape
    cx, cy = W // 2, H // 2
    radius = 20

    angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]

    errors = []
    for angle_deg in angles_deg:
        rad = np.radians(angle_deg)
        x = int(cx + radius * np.sin(rad))
        # GIS North = up = negative row direction in image space
        y = int(cy - radius * np.cos(rad))
        if 0 <= y < H and 0 <= x < W:
            actual = result["aspect"][y, x]
            err = min(abs(actual - rad), 2 * np.pi - abs(actual - rad))
            errors.append(np.degrees(err))

    max_err = max(errors) if errors else 999
    ok = max_err < 10.0
    return ok, "aspect error <10° at 8 compass points", f"max error={max_err:.1f}°", \
           "Aspect direction or normalization broken" if not ok else ""


def t_zero_slope() -> Tuple[bool, str, str, str]:
    """Flat terrain → slope=0°, aspect=0° (North default)."""
    h_scale = 2.0
    hm = make_flat(elevation=50.0, h_scale=h_scale)
    layer = Layer1_LocalGeometry(PipelineConfig(horizontal_scale=h_scale))
    result = layer.execute(hm)
    
    max_slope = float(np.max(result["slope"]))
    aspect_unique = np.unique(result["aspect"])
    
    slope_ok = max_slope < 0.01
    aspect_ok = len(aspect_unique) == 1 and aspect_unique[0] == 0.0
    no_nan = not (np.any(np.isnan(result["slope"])) or np.any(np.isnan(result["aspect"])))
    
    ok = slope_ok and aspect_ok and no_nan
    return ok, "slope=0°, aspect=0°, no NaN", \
           f"max_slope={max_slope:.4f}°, aspect_unique={len(aspect_unique)}", \
           "Flat area handling broken" if not ok else ""


def t_edge_boundary() -> Tuple[bool, str, str, str]:
    """Edge slope matches interior within tolerance."""
    h_scale = 2.0
    hm = make_plane(slope_deg=10.0, aspect_deg=90.0, h_scale=h_scale)
    layer = Layer1_LocalGeometry(PipelineConfig(horizontal_scale=h_scale))
    result = layer.execute(hm)
    
    interior = result["slope"][10:-10, 10:-10]
    interior_mean = np.mean(interior)
    
    edge_top = result["slope"][0:5, 10:-10]
    edge_bottom = result["slope"][-5:, 10:-10]
    edge_left = result["slope"][10:-10, 0:5]
    edge_right = result["slope"][10:-10, -5:]
    
    edge_mean = np.mean(np.concatenate([
        edge_top.flatten(), edge_bottom.flatten(),
        edge_left.flatten(), edge_right.flatten()
    ]))
    
    drift = abs(edge_mean - interior_mean)
    ok = drift < 1.0
    return ok, "edge slope within 1° of interior", \
           f"interior={interior_mean:.2f}°, edge={edge_mean:.2f}° (drift={drift:.2f}°)", \
           "Boundary mode='reflect' not working" if not ok else ""


def t_slope_range() -> Tuple[bool, str, str, str]:
    """Slope clamped to [0, 90] degrees."""
    h_scale = 2.0
    hm = make_plane(slope_deg=85.0, aspect_deg=0.0, h_scale=h_scale)
    layer = Layer1_LocalGeometry(PipelineConfig(horizontal_scale=h_scale))
    result = layer.execute(hm)
    
    min_s = float(np.min(result["slope"]))
    max_s = float(np.max(result["slope"]))
    
    ok = min_s >= 0 and max_s <= 90
    return ok, "slope ∈ [0, 90]°", f"range=[{min_s:.1f}, {max_s:.1f}]°", \
           "Slope range validation broken" if not ok else ""


def t_aspect_range() -> Tuple[bool, str, str, str]:
    """Aspect normalized to [0, 2π] radians."""
    h_scale = 2.0
    hm = make_plane(slope_deg=10.0, aspect_deg=270.0, h_scale=h_scale)
    layer = Layer1_LocalGeometry(PipelineConfig(horizontal_scale=h_scale))
    result = layer.execute(hm)
    
    min_a = float(np.min(result["aspect"]))
    max_a = float(np.max(result["aspect"]))
    two_pi = 2 * np.pi
    
    ok = min_a >= 0 and max_a <= two_pi
    return ok, "aspect ∈ [0, 2π] rad", f"range=[{min_a:.3f}, {max_a:.3f}] rad", \
           "Aspect normalization broken" if not ok else ""


# ─── Runner ──────────────────────────────────────────────────────────────────

def run():
    print("\n[L1] Local Geometry Invariant Tests")
    print("-" * 50)
    
    tests = [
        ("analytical_slope", t_analytical_slope),
        ("aspect_circularity", t_aspect_circularity),
        ("zero_slope", t_zero_slope),
        ("edge_boundary", t_edge_boundary),
        ("slope_range", t_slope_range),
        ("aspect_range", t_aspect_range),
    ]
    
    all_ok = True
    for name, fn in tests:
        ok, exp, got, why = fn()
        status = "GOOD" if ok else "FAIL"
        print(f"[L1|{name:16}].......[{status}]", end="")
        if not ok:
            all_ok = False
            print(f"\n\texpected: {exp}\n\tobtained: {got}\n\twhy:      {why}")
        else:
            print()
    
    print("-" * 50 + f"\n[SUMMARY] {'PASS' if all_ok else 'FAILED'}\n")
    return all_ok


if __name__ == "__main__":
    run()