"""
Layer 2: Regional Geometry Module

Computes second derivatives (curvature) from calibrated heightmap.
Critical for identifying ridges, valleys, and saddles.
"""

import numpy as np
from scipy import ndimage
from typing import Dict
import warnings

from core import Heightmap, ScalarField, PipelineConfig, PipelineLayer

## HELPERS

def compute_adaptive_epsilon(curvature: ScalarField, method='percentile') -> float:
    """
    Automatically determine optimal curvature epsilon based on data distribution.
    
    Args:
        curvature: Mean curvature field
        method: 'percentile', 'std', or 'mad' (median absolute deviation)
    
    Returns:
        Adaptive epsilon value
    """
    # Flatten and remove outliers for robust statistics
    c_flat = curvature.flatten()
    c_flat = c_flat[~np.isnan(c_flat)]
    
    if method == 'percentile':
        # Use 10th percentile of absolute curvature values
        # This captures meaningful curvature while ignoring noise
        epsilon = np.percentile(np.abs(c_flat), 10)
        
    elif method == 'std':
        # Use 0.5 * standard deviation (common in geomorphometry)
        epsilon = 0.5 * np.std(c_flat)
        
    elif method == 'mad':
        # Median Absolute Deviation - robust to outliers
        median = np.median(c_flat)
        mad = np.median(np.abs(c_flat - median))
        epsilon = 0.6745 * mad  # Convert to standard deviation equivalent
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Ensure epsilon is reasonable
    epsilon = max(epsilon, 1e-6)  # Never go below 1e-6
    
    return epsilon


def find_curvature_elbow(curvature: ScalarField) -> float:
    """
    Find the elbow point in curvature histogram for automatic threshold.
    Uses the Kneedle algorithm concept.
    """
    c_flat = np.abs(curvature.flatten())
    c_flat = c_flat[~np.isnan(c_flat)]
    
    # Create histogram
    hist, bin_edges = np.histogram(c_flat, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Normalize
    hist = hist / hist.max()
    
    # Find elbow (point of maximum curvature in the cumulative distribution)
    # This separates noise from signal
    cumulative = np.cumsum(hist) / np.sum(hist)
    
    # Find where cumulative distribution deviates from linear
    x = np.linspace(0, 1, len(cumulative))
    diff = cumulative - x
    
    # Elbow is where difference is maximum
    elbow_idx = np.argmax(diff)
    epsilon = bin_centers[elbow_idx]
    
    return epsilon

def compute_multiscale_epsilon(heightmap: Heightmap, config: PipelineConfig) -> float:
    """
    Compute epsilon by analyzing curvature at multiple scales.
    Real features persist across scales, noise doesn't.
    """
    z = heightmap.data
    cell_size = heightmap.config.horizontal_scale
    
    epsilons = []
    
    for scale_name, radius in config.analysis_scales.items():
        # Blur at this scale
        z_scaled = ndimage.gaussian_filter(z, sigma=radius)
        
        # Compute curvature
        dz_dy, dz_dx = np.gradient(z_scaled, cell_size)
        d2z_dy2, d2z_dxdy_1 = np.gradient(dz_dy, cell_size)
        d2z_dxdy_2, d2z_dx2 = np.gradient(dz_dx, cell_size)
        H = (d2z_dx2 + d2z_dy2) / 2
        
        # Adaptive epsilon at this scale
        eps = compute_adaptive_epsilon(H)
        epsilons.append(eps)
    
    # Use median across scales (robust to scale-dependent noise)
    epsilon = np.median(epsilons)
    
    return epsilon

def compute_snr_based_epsilon(curvature: ScalarField, noise_estimate: float) -> float:
    """
    Use signal-to-noise ratio to determine epsilon.
    
    Args:
        curvature: Mean curvature field
        noise_estimate: From Layer 0 (0.034m in your star peak example)
    
    Returns:
        Epsilon = noise_estimate * (curvature_scale_factor)
    """
    # Curvature scale factor: for 5m/pixel, 1m elevation change over 10m = 0.1 1/m
    # But noise at 0.034m elevation produces curvature ~0.0003 1/m
    # So we need epsilon above noise floor
    
    # Estimate curvature noise floor
    c_flat = curvature.flatten()
    noise_floor = np.percentile(np.abs(c_flat), 5)  # 5th percentile as noise floor
    
    # Set epsilon 2-3x above noise floor
    epsilon = noise_floor * 2.5
    
    return epsilon
    
# Version 2

class Layer2_Adaptive(Layer2_RegionalGeometry):
    """Layer 2 with automatic epsilon detection."""
    
    def execute(self, input_data: Heightmap) -> Dict[str, ScalarField]:
        """Compute curvature with automatic epsilon."""
        
        z = input_data.data
        cell_size = input_data.config.horizontal_scale
        
        # Compute curvature (same as before)
        dz_dy, dz_dx = np.gradient(z, cell_size)
        d2z_dy2, d2z_dxdy_1 = np.gradient(dz_dy, cell_size)
        d2z_dxdy_2, d2z_dx2 = np.gradient(dz_dx, cell_size)
        d2z_dxdy = (d2z_dxdy_1 + d2z_dxdy_2) / 2
        mean_curvature = (d2z_dx2 + d2z_dy2) / 2
        gaussian_curvature = d2z_dx2 * d2z_dy2 - d2z_dxdy**2
        
        # AUTO-DETECT EPSILON
        epsilon = self._auto_detect_epsilon(mean_curvature)
        
        print(f"Auto-detected curvature epsilon: {epsilon:.6f} 1/m")
        
        # Classify with detected epsilon
        curvature_type = self._classify_with_epsilon(
            mean_curvature, 
            gaussian_curvature, 
            epsilon
        )
        
        return {
            "curvature": mean_curvature.astype(np.float32),
            "curvature_type": curvature_type,
            "epsilon_used": epsilon  # Optional: include in output
        }
    
    def _auto_detect_epsilon(self, curvature: ScalarField) -> float:
        """
        Automatically detect optimal epsilon using multiple methods.
        """
        # Method 1: Statistical threshold (10th percentile)
        c_flat = np.abs(curvature.flatten())
        c_flat = c_flat[~np.isnan(c_flat)]
        epsilon_percentile = np.percentile(c_flat, 10)
        
        # Method 2: Standard deviation method
        epsilon_std = 0.5 * np.std(c_flat)
        
        # Method 3: MAD method (robust)
        median = np.median(c_flat)
        mad = np.median(np.abs(c_flat - median))
        epsilon_mad = 0.6745 * mad  # Convert to equivalent std
        
        # Method 4: Noise floor method
        noise_floor = np.percentile(c_flat, 5)
        epsilon_noise = noise_floor * 2.5
        
        # Combine methods (median of all for robustness)
        epsilons = [epsilon_percentile, epsilon_std, epsilon_mad, epsilon_noise]
        
        # Filter out unreasonable values (too small or too large)
        epsilons = [e for e in epsilons if 1e-8 < e < 1e-2]
        
        # Use median as final epsilon
        epsilon = np.median(epsilons)
        
        # Ensure minimum threshold
        epsilon = max(epsilon, 1e-6)
        
        return epsilon
    
    def _classify_with_epsilon(self, H, K, epsilon):
        """Classification with given epsilon."""
        result = np.full(H.shape, "FLAT", dtype='<U10')
        
        saddle_mask = K < -epsilon
        result[saddle_mask] = "SADDLE"
        
        convex_mask = (H > epsilon) & (K > epsilon) & ~saddle_mask
        result[convex_mask] = "CONVEX"
        
        concave_mask = (H < -epsilon) & (K > epsilon) & ~saddle_mask
        result[concave_mask] = "CONCAVE"
        
        return result

# Version 1
class Layer2_RegionalGeometry(PipelineLayer[Dict[str, ScalarField]]):
    """
    Compute second derivatives: curvature classification.
    
    Takes calibrated Heightmap directly (not slope/aspect) to avoid
    compounding numerical errors from two finite-difference steps.
    """
    
    # String labels for curvature types (compatible with Layer 3)
    CURVATURE_LABELS = {
        0: "FLAT",
        1: "CONVEX", 
        2: "CONCAVE",
        3: "SADDLE"
    }
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self._extra_smoothing = config.noise_reduction_sigma < 1.5
        
    def execute(self, input_data: Heightmap) -> Dict[str, ScalarField]:
        """
        Compute curvature fields from heightmap.
        
        Args:
            input_data: Heightmap from Layer 0
            
        Returns:
            dict with 'curvature' and 'curvature_type' fields
        """
        # Validate input
        if not isinstance(input_data, Heightmap):
            raise TypeError(f"Expected Heightmap, got {type(input_data)}")
        
        z = input_data.data
        cell_size = input_data.config.horizontal_scale
        
        # Optional: Extra smoothing for second derivatives
        # Only if Layer 0 smoothing was minimal
        if self._extra_smoothing:
            z = ndimage.gaussian_filter(z, sigma=0.5)
            #warnings.warn("Applied extra σ=0.5 smoothing for second derivatives", UserWarning)
            print("Applied extra σ=0.5 smoothing for second derivatives")
        
        # Compute all second derivatives correctly
        mean_curvature, gaussian_curvature = self._compute_curvature(z, cell_size)
        
        # Classify curvature type
        curvature_type = self._classify_curvature(mean_curvature, gaussian_curvature)
        
        return {
            "curvature": mean_curvature.astype(np.float32),
            "curvature_type": curvature_type
        }
    
    def _compute_curvature(self, z: ScalarField, cell_size: float) -> tuple[ScalarField, ScalarField]:
        """
        Compute mean and Gaussian curvature from heightfield.
        
        Uses central differences with symmetric mixed partial calculation.
        
        Returns:
            tuple: (mean_curvature, gaussian_curvature) in 1/meters
        """
        # First derivatives
        dz_dy, dz_dx = np.gradient(z, cell_size)
        
        # Second derivatives with symmetric mixed partial
        # Method 1: From dz_dy
        d2z_dy2, d2z_dxdy_1 = np.gradient(dz_dy, cell_size)
        
        # Method 2: From dz_dx  
        d2z_dxdy_2, d2z_dx2 = np.gradient(dz_dx, cell_size)
        
        # Average mixed partial for symmetry (ensures smoothness)
        d2z_dxdy = (d2z_dxdy_1 + d2z_dxdy_2) / 2
        
        # Mean curvature: H = (z_xx + z_yy) / 2
        # Units: 1/meters
        mean_curvature = (d2z_dx2 + d2z_dy2) / 2
        
        # Gaussian curvature: K = z_xx * z_yy - (z_xy)²
        # Units: 1/meters²
        gaussian_curvature = d2z_dx2 * d2z_dy2 - d2z_dxdy**2
        
        return mean_curvature, gaussian_curvature
    
    def _classify_curvature(self, H: ScalarField, K: ScalarField) -> np.ndarray:
        """
        Classify each pixel into curvature type based on mean (H) and Gaussian (K) curvature.
        
        Classification logic:
        - FLAT: |H| < ε and |K| < ε (near-zero curvature)
        - CONVEX: H > ε and K > ε (bowl upward, like ridge/peak)
        - CONCAVE: H < -ε and K > ε (bowl downward, like valley)
        - SADDLE: K < -ε (principal curvatures have opposite signs, like pass)
        
        Returns:
            String array with curvature labels
        """
        epsilon = self.config.curvature_epsilon
        
        # Initialize with FLAT
        result = np.full(H.shape, "FLAT", dtype='<U10')
        
        # Saddle points (negative Gaussian curvature) - HIGHEST PRIORITY
        # These are passes, cols, and other mixed curvature features
        saddle_mask = K < -epsilon
        result[saddle_mask] = "SADDLE"
        
        # Convex (positive mean curvature, positive Gaussian)
        # These are ridges, peaks, and other locally high features
        convex_mask = (H > epsilon) & (K > epsilon) & ~saddle_mask
        result[convex_mask] = "CONVEX"
        
        # Concave (negative mean curvature, positive Gaussian)
        # These are valleys, pits, and other locally low features
        concave_mask = (H < -epsilon) & (K > epsilon) & ~saddle_mask
        result[concave_mask] = "CONCAVE"
        
        # Log classification statistics
        self._log_classification_stats(result)
        
        return result
    
    def _log_classification_stats(self, curvature_type: np.ndarray) -> None:
        """Log curvature classification statistics."""
        total = curvature_type.size
        stats = {}
        for label in ["FLAT", "CONVEX", "CONCAVE", "SADDLE"]:
            count = np.sum(curvature_type == label)
            pct = 100 * count / total
            stats[label] = (count, pct)
        
        # Only log if significant non-flat areas
        if stats["CONVEX"][0] + stats["CONCAVE"][0] + stats["SADDLE"][0] > 1000:
            warnings.warn(
                f"Curvature classification: "
                f"CONVEX={stats['CONVEX'][0]:.0f} ({stats['CONVEX'][1]:.1f}%), "
                f"CONCAVE={stats['CONCAVE'][0]:.0f} ({stats['CONCAVE'][1]:.1f}%), "
                f"SADDLE={stats['SADDLE'][0]:.0f} ({stats['SADDLE'][1]:.1f}%), "
                f"FLAT={stats['FLAT'][0]:.0f} ({stats['FLAT'][1]:.1f}%)",
                UserWarning
            )
    
    @property
    def output_schema(self) -> dict:
        return {
            "curvature": {
                "type": "ScalarField", 
                "unit": "1/meters",
                "description": "Mean curvature (H)"
            },
            "curvature_type": {
                "type": "CategoricalField", 
                "values": ["CONVEX", "CONCAVE", "FLAT", "SADDLE"],
                "description": "Curvature classification for each pixel"
            }
        }


# Optional: Multi-scale curvature for your hairline ridge
class Layer2_MultiScale(Layer2_RegionalGeometry):
    """
    Extended version that computes curvature at multiple scales.
    
    Critical for your terrain where features vary widely in scale:
    - Hairline ridge: 1-3px (micro)
    - Mountain pass: 10-20px (meso)
    - Coastal features: 50+px (macro)
    """
    
    def execute(self, input_data: Heightmap) -> Dict[str, Dict]:
        """
        Compute curvature at multiple scales.
        
        Returns:
            dict with scale_name -> {"curvature": ..., "curvature_type": ...}
        """
        z = input_data.data
        cell_size = input_data.config.horizontal_scale
        
        results = {}
        
        for scale_name, radius in self.config.analysis_scales.items():
            # Apply Gaussian blur at this scale
            z_scaled = ndimage.gaussian_filter(z, sigma=radius)
            
            # Compute curvature on blurred surface
            mean_curvature, gaussian_curvature = self._compute_curvature(z_scaled, cell_size)
            curvature_type = self._classify_curvature(mean_curvature, gaussian_curvature)
            
            results[scale_name] = {
                "curvature": mean_curvature.astype(np.float32),
                "curvature_type": curvature_type
            }
        
        return results