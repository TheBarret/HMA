"""
Layer 0 Calibration Test — Invariant Tests + Calibration

Imports from real pipeline modules (core.py, calibration.py).
No local model redefinitions.

Output: [L0|test].......[GOOD/FAIL]
        On FAIL: newline + tab shows expected | obtained | why
Usage: python calibrate_layer0.py
"""

import numpy as np
from typing import Tuple

from core import Heightmap, NormalizationConfig, PipelineConfig, RawImageInput
from calibration import Layer0_Calibration


# ─── Synthetic Surface Helpers (Test Fixtures) ───────────────────────────────

def _make_heightmap(Z: np.ndarray, h_scale: float = 2.0, offset: float = 0.0) -> Heightmap:
    """Create a Heightmap from elevation array (bypassing Layer 0)."""
    cfg = NormalizationConfig(horizontal_scale=h_scale, vertical_scale=0.2, sea_level_offset=offset)
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


def make_plane(shape=(64, 64), h_scale=2.0, slope_deg=10.0, aspect_deg=90.0, offset=10.0) -> Heightmap:
    """Planar surface with known slope and aspect."""
    H, W = shape
    slope_rad = np.radians(slope_deg)
    aspect_rad = np.radians(aspect_deg)
    
    dz_dx = np.tan(slope_rad) * np.sin(aspect_rad)
    dz_dy = np.tan(slope_rad) * np.cos(aspect_rad)
    
    x = np.arange(W) * h_scale - W * h_scale / 2
    y = np.arange(H) * h_scale - H * h_scale / 2
    X, Y = np.meshgrid(x, y)
    Z = offset + dz_dx * X + dz_dy * Y
    
    return _make_heightmap(Z, h_scale, offset)


def make_peak(shape=(64, 64), h_scale=2.0, height=25.0, sigma=8.0, offset=10.0) -> Heightmap:
    """Gaussian peak for feature preservation testing."""
    X, Y = _grid(shape, h_scale)
    Z = offset + height * np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    return _make_heightmap(Z, h_scale, offset)


def make_flat(shape=(64, 64), h_scale=2.0, elevation=10.0) -> Heightmap:
    """Flat surface for zero slope testing."""
    H, W = shape
    Z = np.full((H, W), elevation, dtype=np.float32)
    return _make_heightmap(Z, h_scale, elevation)


# ─── Test Helpers ─────────────────────────────────────────────────────────────

def raw_from_heightmap(hm: Heightmap) -> np.ndarray:
    """Convert Heightmap to raw uint8 grayscale (simulating sensor input)."""
    v_scale = hm.config.vertical_scale
    offset = hm.config.sea_level_offset
    raw = (hm.data - offset) / v_scale
    return np.clip(raw, 0, 255).astype(np.uint8)


# ─── Invariant Tests ─────────────────────────────────────────────────────────

def t_linearity() -> Tuple[bool, str, str, str]:
    """Linear scaling preserved after quantization+blur."""
    h_scale, v_scale, offset = 2.0, 0.2, 10.0
    cfg = PipelineConfig(horizontal_scale=h_scale, vertical_scale=v_scale,
                         sea_level_offset=offset, noise_reduction_sigma=1.2)
    
    # Create plane with known gradient
    hm = make_plane(slope_deg=15.0, aspect_deg=45.0, offset=offset)
    raw = raw_from_heightmap(hm)
    
    # Run calibration
    cal = Layer0_Calibration(cfg)
    calibrated = cal.execute(
        RawImageInput(data=raw),
        NormalizationConfig(h_scale, v_scale, offset)
    )
    
    # Reconstruct and compare
    reconstructed = (calibrated.data - offset) / v_scale
    expected_raw = raw.astype(float)
    corr = np.corrcoef(expected_raw.flatten(), reconstructed.flatten())[0, 1]
    mae = np.mean(np.abs(expected_raw - reconstructed))
    
    ok = corr > 0.95 and mae < 2.5
    return ok, f"corr>0.95, MAE<2.5", f"corr={corr:.3f}, MAE={mae:.2f}", \
           "Quantization/blur error" if not ok else ""


def t_bounds() -> Tuple[bool, str, str, str]:
    """No NaN/Inf values after calibration."""
    h_scale, v_scale, offset = 2.0, 0.2, 10.0
    cfg = PipelineConfig(horizontal_scale=h_scale, vertical_scale=v_scale,
                         sea_level_offset=offset, noise_reduction_sigma=1.2)
    
    hm = make_plane(offset=offset)
    raw = raw_from_heightmap(hm)
    
    cal = Layer0_Calibration(cfg)
    calibrated = cal.execute(
        RawImageInput(data=raw),
        NormalizationConfig(h_scale, v_scale, offset)
    )
    
    has_nan = np.any(np.isnan(calibrated.data))
    has_inf = np.any(np.isinf(calibrated.data))
    ok = not (has_nan or has_inf)
    return ok, "no NaN/Inf", f"NaN={has_nan}, Inf={has_inf}", \
           "Numerical instability" if not ok else ""


def t_coords() -> Tuple[bool, str, str, str]:
    """Pixel-to-world transform correct."""
    h_scale = 2.0
    cfg = PipelineConfig(horizontal_scale=h_scale)
    hm = make_flat(h_scale=h_scale)
    raw = raw_from_heightmap(hm)
    
    cal = Layer0_Calibration(cfg)
    calibrated = cal.execute(
        RawImageInput(data=raw),
        NormalizationConfig(h_scale, 0.2, 0.0)
    )
    
    o = calibrated.pixel_to_world((0, 0))
    s = calibrated.pixel_to_world((1, 0))
    ok = abs(o[0]) < 1e-6 and abs(s[0] - h_scale) < 1e-6
    return ok, f"origin=(0,0), step=({h_scale},0)", f"origin={o}, step={s}", \
           "Transform bug" if not ok else ""


def t_affine() -> Tuple[bool, str, str, str]:
    """Affine georeferencing path works."""
    from calibration import Layer0_Calibration
    from core import PipelineConfig, RawImageInput, NormalizationConfig
    
    cfg = PipelineConfig(horizontal_scale=2.0, vertical_scale=0.2,
                         sea_level_offset=10.0, noise_reduction_sigma=0)
    cal = Layer0_Calibration(cfg)
    georef = {'affine': [2.0, 0.0, 100.0, 0.0, 2.0, 200.0]}
    hm = cal.execute(
        RawImageInput(data=np.zeros((10, 10), dtype=np.uint8)),
        NormalizationConfig(2.0, 0.2, 10.0),
        georef
    )
    o = hm.pixel_to_world((0, 0))
    s = hm.pixel_to_world((1, 0))
    ok = abs(o[0] - 100.0) < 1e-6 and abs(s[0] - 102.0) < 1e-6
    return ok, "affine origin=(100,200), step→(102,200)", f"origin={o}, step={s}", \
           "Affine path broken" if not ok else ""


def t_features() -> Tuple[bool, str, str, str]:
    """Peak preservation within sigma-based tolerances."""
    h_scale, v_scale, offset = 2.0, 0.2, 10.0
    sigma = 1.2
    cfg = PipelineConfig(horizontal_scale=h_scale, vertical_scale=v_scale,
                         sea_level_offset=offset, noise_reduction_sigma=sigma)
    
    # Create peak at center
    hm = make_peak(height=25.0, sigma=8.0, offset=offset)
    raw = raw_from_heightmap(hm)
    
    cal = Layer0_Calibration(cfg)
    calibrated = cal.execute(
        RawImageInput(data=raw),
        NormalizationConfig(h_scale, v_scale, offset)
    )
    
    # Find peak in truth and calibrated
    truth_data = hm.data
    cal_data = calibrated.data
    
    truth_peak_y, truth_peak_x = np.unravel_index(np.argmax(truth_data), truth_data.shape)
    cal_peak_y, cal_peak_x = np.unravel_index(np.argmax(cal_data), cal_data.shape)
    truth_h = truth_data[truth_peak_y, truth_peak_x]
    cal_h = cal_data[cal_peak_y, cal_peak_x]
    
    pos_err = np.hypot(truth_peak_x - cal_peak_x, truth_peak_y - cal_peak_y)
    h_err = abs(truth_h - cal_h) / truth_h * 100
    
    max_pos_err = 2.5 * sigma  # ~3px
    max_h_err = 10 + 5 * sigma  # ~16%
    ok = pos_err < max_pos_err and h_err < max_h_err
    
    return ok, f"pos<{max_pos_err:.0f}px, height<{max_h_err:.0f}%", \
           f"pos={pos_err:.1f}px, height={h_err:.1f}%", \
           "Blur shifted/attenuated peak beyond limits" if not ok else ""


def t_monotonicity() -> Tuple[bool, str, str, str]:
    """Elevation order preserved (≥90% of pairs)."""
    h_scale, v_scale, offset = 2.0, 0.2, 10.0
    cfg = PipelineConfig(horizontal_scale=h_scale, vertical_scale=v_scale,
                         sea_level_offset=offset, noise_reduction_sigma=1.2)
    
    hm = make_peak(height=25.0, offset=offset)
    raw = raw_from_heightmap(hm)
    
    cal = Layer0_Calibration(cfg)
    calibrated = cal.execute(
        RawImageInput(data=raw),
        NormalizationConfig(h_scale, v_scale, offset)
    )
    
    H, W = hm.data.shape
    step = max(1, min(H, W) // 10)
    pts = [(y, x) for y in range(0, H, step) for x in range(0, W, step)]
    
    viol, total = 0, 0
    for i, (y1, x1) in enumerate(pts):
        for y2, x2 in pts[i + 1:]:
            od = hm.data[y1, x1] - hm.data[y2, x2]
            cd = calibrated.data[y1, x1] - calibrated.data[y2, x2]
            if abs(od) > 1.0:
                total += 1
                if od * cd < 0:
                    viol += 1
    
    rate = viol / total if total else 0
    ok = rate < 0.10
    return ok, "order preserved ≥90%", f"violation rate={rate*100:.1f}%", \
           "Non-monotonic blur or clipping" if not ok else ""


def t_edge() -> Tuple[bool, str, str, str]:
    """Boundary pixels don't exhibit artifacts."""
    h_scale, v_scale, offset = 2.0, 0.2, 10.0
    cfg = PipelineConfig(horizontal_scale=h_scale, vertical_scale=v_scale,
                         sea_level_offset=offset, noise_reduction_sigma=1.2)
    
    hm = make_plane(offset=offset)
    raw = raw_from_heightmap(hm)
    
    cal = Layer0_Calibration(cfg)
    calibrated = cal.execute(
        RawImageInput(data=raw),
        NormalizationConfig(h_scale, v_scale, offset)
    )
    
    d = calibrated.data
    h, w = d.shape
    edge = np.concatenate([d[0, :], d[-1, :], d[:, 0], d[:, -1]])
    interior = d[2:-2, 2:-2].flatten() if h > 4 and w > 4 else d.flatten()
    ratio = np.std(edge) / np.std(interior) if np.std(interior) > 0 else 0
    ok = ratio < 2.0
    return ok, "edge_std/interior_std < 2.0", f"ratio={ratio:.2f}", \
           "Boundary artifact from blur mode" if not ok else ""


def t_zero_input() -> Tuple[bool, str, str, str]:
    """Zero input produces valid heightmap with no NaN/Inf."""
    cfg = PipelineConfig(horizontal_scale=1.0, vertical_scale=0.1,
                         sea_level_offset=0.0, noise_reduction_sigma=1.2)
    cal = Layer0_Calibration(cfg)
    raw = np.zeros((32, 32), dtype=np.uint8)
    hm = cal.execute(
        RawImageInput(data=raw),
        NormalizationConfig(1.0, 0.1, 0.0)
    )
    
    ok = not (np.any(np.isnan(hm.data)) or np.any(np.isinf(hm.data))) and abs(hm.data.min()) < 1e-6
    return ok, "min=0, no NaN/Inf", f"min={hm.data.min():.3f}, NaN={np.any(np.isnan(hm.data))}", \
           "Zero input handling broken" if not ok else ""


def t_snapshot() -> Tuple[bool, str, str, str]:
    """Regression snapshot — catches accidental algorithm changes."""
    cfg = PipelineConfig(horizontal_scale=1.0, vertical_scale=0.1,
                         sea_level_offset=0.0, noise_reduction_sigma=1.2)
    cal = Layer0_Calibration(cfg)
    raw = np.full((32, 32), 128, dtype=np.uint8)
    raw[16, 16] = 255
    hm = cal.execute(
        RawImageInput(data=raw),
        NormalizationConfig(1.0, 0.1, 0.0)
    )
    
    metrics = {
        'center': float(hm.data[16, 16]),
        'nbhd': float(np.mean(hm.data[14:18, 14:18])),
        'std': float(np.std(hm.data))
    }
    
    # Expected values from reference run
    exp = {'center': 14.2, 'nbhd': 13.4, 'std': 0.09}
    tolerance = {'center': 0.5, 'nbhd': 0.5, 'std': 0.02}
    ok = all(abs(metrics[k] - exp[k]) < tolerance[k] for k in exp)
    
    return ok, f"center≈{exp['center']}, nbhd≈{exp['nbhd']}, std≈{exp['std']}", \
           f"center={metrics['center']:.1f}, nbhd={metrics['nbhd']:.1f}, std={metrics['std']:.3f}", \
           "Algorithm regression" if not ok else ""


# ─── Runner ──────────────────────────────────────────────────────────────────

def run():
    print("\n[L0] Calibration Invariant Tests")
    print("-" * 50)
    
    tests = [
        ("linearity",    t_linearity),
        ("bounds",       t_bounds),
        ("coords",       t_coords),
        ("affine",       t_affine),
        ("features",     t_features),
        ("monotonicity", t_monotonicity),
        ("edge",         t_edge),
        ("zero_input",   t_zero_input),
        ("snapshot",     t_snapshot),
    ]
    
    all_ok = True
    for name, fn in tests:
        ok, exp, got, why = fn()
        status = "GOOD" if ok else "FAIL"
        print(f"[L0|{name:12}].......[{status}]", end="")
        if not ok:
            all_ok = False
            print(f"\n\texpected: {exp}\n\tobtained: {got}\n\twhy:      {why}")
        else:
            print()
    
    print("-" * 50 + f"\n[SUMMARY] {'PASS' if all_ok else 'FAILED'}\n")
    return all_ok


if __name__ == "__main__":
    run()