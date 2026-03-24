"""
Layer 2: Regional Geometry Module

"""

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import warnings

from core import Heightmap, ScalarField, PipelineConfig, PipelineLayer

class Layer2_RegionalGeometry(PipelineLayer[Dict[str, ScalarField]]):
    """
    Compute second derivatives: curvature classification.
    """
    
    CURVATURE_LABELS = ["FLAT", "SADDLE", "CONVEX", "CONCAVE"]
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.adaptive_epsilon = config.adaptive_epsilon  # Read from config
        self._epsilon_used: Optional[tuple[float, float]] = None
    
    def _debug(self, msg: str) -> None:
        """Helper for verbose logging"""
        if self.config.verbose:
            print(f"[Regional] {msg}")
        
    def execute(self, input_data: Heightmap) -> Dict[str, ScalarField]:
        """
        Compute curvature fields from heightmap.
        
        Returns:
            dict with 'curvature' (mean curvature), 'gaussian_curvature', and 'curvature_type'
        """
        self._debug(f"Computing curvature fields | shape={input_data.shape}, cell_size={input_data.config.horizontal_scale:.4f}m")
        
        if not isinstance(input_data, Heightmap):
            raise TypeError(f"Expected Heightmap, got {type(input_data)}")
        
        z = input_data.data
        cell_size = input_data.config.horizontal_scale
        
        # Compute curvature
        mean_curvature, gaussian_curvature = self._compute_curvature(z, cell_size)
        
        # Validate and clean outputs
        self._debug("Validating curvature fields")
        mean_curvature, gaussian_curvature = self._validate_curvature(
            mean_curvature, gaussian_curvature
        )
        
        # Determine epsilon — adaptive uses both H and K independently
        self._debug("Determining epsilon thresholds")
        h_epsilon, k_epsilon = self._get_epsilon(mean_curvature, gaussian_curvature)
        self._epsilon_used = (h_epsilon, k_epsilon)

        # Classify curvature type
        self._debug(f"Classifying curvature | H_eps={h_epsilon:.6f}, K_eps={k_epsilon:.6f}")
        curvature_type = self._classify_curvature(
            mean_curvature, gaussian_curvature, h_epsilon, k_epsilon
        )

        # Log statistics
        self._log_classification_stats(curvature_type, h_epsilon, k_epsilon)
        
        return {
            "curvature": mean_curvature.astype(np.float32),
            "gaussian_curvature": gaussian_curvature.astype(np.float32),
            "curvature_type": curvature_type
        }
    
    def _compute_curvature(self, z: ScalarField, cell_size: float) -> Tuple[ScalarField, ScalarField]:
        """Compute mean and Gaussian curvature using full differential geometry formulas."""
        self._debug("Computing gradients and second derivatives")
        
        # First derivatives
        dz_dy, dz_dx = np.gradient(z, cell_size)
        
        # Second derivatives with symmetric mixed partial
        d2z_dy2, d2z_dxdy_1 = np.gradient(dz_dy, cell_size)
        d2z_dxdy_2, d2z_dx2 = np.gradient(dz_dx, cell_size)
        d2z_dxdy = (d2z_dxdy_1 + d2z_dxdy_2) / 2.0
        
        # Common denominator terms (account for slope magnitude)
        denom = 1 + dz_dx**2 + dz_dy**2
        denom_sqrt = np.sqrt(denom)
        
        # FULL Mean Curvature formula (valid for all slopes)
        # H = [(1+fy²)fxx - 2fxfyfxy + (1+fx²)fyy] / [2(1+fx²+fy²)^(3/2)]
        mean_curvature = (
            (1 + dz_dy**2) * d2z_dx2 
            - 2 * dz_dx * dz_dy * d2z_dxdy 
            + (1 + dz_dx**2) * d2z_dy2
        ) / (2 * denom_sqrt**3)
        
        # FULL Gaussian Curvature formula
        # K = (fxx*fyy - fxy²) / (1+fx²+fy²)²
        gaussian_curvature = (
            d2z_dx2 * d2z_dy2 - d2z_dxdy**2
        ) / (denom**2)
        
        return mean_curvature, gaussian_curvature
    
    def _validate_curvature(self, H: ScalarField, K: ScalarField) -> Tuple[ScalarField, ScalarField]:
        """Handle NaN, Inf, and extreme values. H and K are validated independently."""
        H = np.asarray(H, dtype=np.float32)
        K = np.asarray(K, dtype=np.float32)
        
        # Handle NaN — if either field has NaN, zero both at those positions
        nan_mask = np.isnan(H) | np.isnan(K)
        if np.any(nan_mask):
            nan_count = int(np.sum(nan_mask))
            self._debug(f"NaN values detected: {nan_count} positions, filling with 0")
            H[nan_mask] = 0.0
            K[nan_mask] = 0.0
        
        # Handle Inf — clip each field to its own reasonable range independently
        # H units: 1/m,  K units: 1/m² — different magnitudes, different clip bounds
        if np.any(np.isinf(H)):
            inf_count = int(np.sum(np.isinf(H)))
            self._debug(f"Mean curvature (H) contains {inf_count} Inf values, clipping to [-1.0, 1.0]")
            H = np.clip(H, -1.0, 1.0)        # 1/m: reasonable bound for terrain
        if np.any(np.isinf(K)):
            inf_count = int(np.sum(np.isinf(K)))
            self._debug(f"Gaussian curvature (K) contains {inf_count} Inf values, clipping to [-1.0, 1.0]")
            K = np.clip(K, -1.0, 1.0)        # 1/m²: same order of magnitude for terrain
        
        return H, K
    
    def _get_epsilon(self, H: ScalarField, K: ScalarField) -> tuple[float, float]:
        """
        Return (h_epsilon, k_epsilon) — independent thresholds for H and K.
        """
        if self.adaptive_epsilon:
            self._debug("Using adaptive epsilon (75th percentile)")
            return self._compute_adaptive_epsilon(H, K)
        else:
            eps = float(self.config.curvature_epsilon)
            self._debug(f"Using fixed epsilon: {eps:.6f}")
            return eps, eps

    def _compute_adaptive_epsilon_old(self, H: ScalarField, K: ScalarField) -> tuple[float, float]:
        """Compute adaptive thresholds using 75th percentile to avoid seam artifact inflation."""
        H_valid = H[~np.isnan(H)]
        K_valid = K[~np.isnan(K)]

        # Use 75th percentile instead of 95th.
        # 95th sits in seam artifacts and extreme peaks, inflating thresholds.
        # 75th sits in the "feature body" where most terrain signals live.
        h_anchor = float(np.percentile(np.abs(H_valid), 75))
        k_anchor = float(np.percentile(np.abs(K_valid), 75))

        h_epsilon = max(self.config.curvature_epsilon_h_factor * h_anchor,
                        self.config.curvature_epsilon_h_min)
        k_epsilon = max(self.config.curvature_epsilon_k_factor * k_anchor,
                        self.config.curvature_epsilon_k_min)

        self._debug(f"Adaptive epsilon | H_anchor={h_anchor:.6f} → H_eps={h_epsilon:.6f}, K_anchor={k_anchor:.6f} → K_eps={k_epsilon:.6f}")
        return h_epsilon, k_epsilon
        
    def _compute_adaptive_epsilon(self, H: ScalarField, K: ScalarField) -> tuple[float, float]:
        """Compute adaptive thresholds using 75th percentile of NON-FLAT regions only."""
        
        # First, identify potentially non-flat regions using slope or elevation variance
        # Simple approach: use a low-pass filtered version to find regions with variation
        from scipy import ndimage
        
        H_valid = H[~np.isnan(H)]
        K_valid = K[~np.isnan(K)]
        
        # Find regions with significant variation (ignore absolute flat)
        H_abs = np.abs(H)
        H_threshold = np.percentile(H_abs, 50)  # 50th percentile as baseline
        non_flat_mask = H_abs > H_threshold
        
        if np.sum(non_flat_mask) < 100:
            # Too few non-flat pixels, use global but with fallback
            h_anchor = np.percentile(np.abs(H_valid), 75)
            k_anchor = np.percentile(np.abs(K_valid), 75)
        else:
            # Use only non-flat regions for percentile calculation
            h_anchor = np.percentile(H_abs[non_flat_mask], 75)
            k_anchor = np.percentile(np.abs(K)[non_flat_mask], 75)
        
        h_epsilon = max(self.config.curvature_epsilon_h_factor * h_anchor,
                        self.config.curvature_epsilon_h_min)
        k_epsilon = max(self.config.curvature_epsilon_k_factor * k_anchor,
                        self.config.curvature_epsilon_k_min)
        
        self._debug(f"Adaptive epsilon (non-flat regions) | H_anchor={h_anchor:.6f} → H_eps={h_epsilon:.6f}, "
                    f"K_anchor={k_anchor:.6f} → K_eps={k_epsilon:.6f}")
        return h_epsilon, k_epsilon
        
    def _classify_curvature_old(self, H: ScalarField, K: ScalarField, h_epsilon: float, k_epsilon: float) -> np.ndarray:
        """Classify using independent H and K thresholds."""
        H = np.asarray(H)
        K = np.asarray(K)
        
        result = np.full(H.shape, "FLAT", dtype='<U10')
        
        # Saddle: K must be negative AND exceed K threshold
        saddle_mask = K < -k_epsilon
        result[saddle_mask] = "SADDLE"
        
        # Convex: H > h_epsilon AND K > k_epsilon
        convex_mask = (H > h_epsilon) & (K > k_epsilon) & ~saddle_mask
        result[convex_mask] = "CONVEX"
        
        # Concave: H < -h_epsilon AND K > k_epsilon
        concave_mask = (H < -h_epsilon) & (K > k_epsilon) & ~saddle_mask
        result[concave_mask] = "CONCAVE"
        
        return result
    
    # Change: 
    # * K > -k_epsilon instead of K > k_epsilon.
    #   This includes cylindrical surfaces (K≈0) in convex/concave classification.
    # * Cylindrical variants
    
    def _classify_curvature(self, H, K, h_epsilon, k_epsilon):
        result = np.full(H.shape, "FLAT", dtype='<U20')
        
        # Saddle
        saddle_mask = K < -k_epsilon
        result[saddle_mask] = "SADDLE"
        
        # Cylindrical convex (ridge-like)
        cyl_convex_mask = (H > h_epsilon) & (np.abs(K) < k_epsilon) & ~saddle_mask
        result[cyl_convex_mask] = "CYLINDRICAL_CONVEX"
        
        # Cylindrical concave (valley-like)
        cyl_concave_mask = (H < -h_epsilon) & (np.abs(K) < k_epsilon) & ~saddle_mask
        result[cyl_concave_mask] = "CYLINDRICAL_CONCAVE"
        
        # True convex (dome)
        convex_mask = (H > h_epsilon) & (K > k_epsilon) & ~saddle_mask
        result[convex_mask] = "CONVEX"
        
        # True concave (bowl)
        concave_mask = (H < -h_epsilon) & (K > k_epsilon) & ~saddle_mask
        result[concave_mask] = "CONCAVE"
        
        return result
    
    def _log_classification_stats(self, curvature_type: np.ndarray, h_epsilon: float, k_epsilon: float) -> None:
        """Log classification statistics."""
        total = curvature_type.size
        flat_count = np.sum(curvature_type == "FLAT")
        convex_count = np.sum(curvature_type == "CONVEX")
        concave_count = np.sum(curvature_type == "CONCAVE")
        saddle_count = np.sum(curvature_type == "SADDLE")
        
        self._debug(f"Classification results: CONVEX={convex_count} ({100*convex_count/total:.1f}%), CONCAVE={concave_count} ({100*concave_count/total:.1f}%), SADDLE={saddle_count} ({100*saddle_count/total:.1f}%), FLAT={flat_count} ({100*flat_count/total:.1f}%)")
    
    @property
    def output_schema(self) -> dict:
        """Schema for pipeline validation."""
        return {
            "curvature": {
                "type": "ScalarField", 
                "unit": "1/meters",
                "description": "Mean curvature (H)"
            },
            "gaussian_curvature": {
                "type": "ScalarField",
                "unit": "1/meters²",
                "description": "Gaussian curvature (K) - product of principal curvatures"
            },
            "curvature_type": {
                "type": "CategoricalField", 
                "values": self.CURVATURE_LABELS,
                "description": "Curvature classification based on H and K"
            }
        }
    
    @property
    def epsilon_used(self) -> Optional[tuple[float, float]]:
        """Get (h_epsilon, k_epsilon) used in the last classification."""
        return self._epsilon_used
        
class MultiScaleCurvatureAnalyzer:
    """
    Diagnostic tool for multi-scale curvature analysis.
    
    This is NOT a pipeline layer. It's for visualization and debugging
    to help determine the appropriate scale for feature extraction.
    
    Analyzes curvature at micro, meso, and macro scales to identify
    which scale best reveals terrain features.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Args:
            config: Pipeline configuration with analysis_scales dictionary
        """
        self.config = config
    
    def _log(self, msg: str) -> None:
        """Helper for verbose logging"""
        if self.config.verbose:
            print(f"[MultiScaleAnalyzer] {msg}")
    
    def analyze(self, heightmap: Heightmap) -> Dict[str, Dict[str, ScalarField]]:
        """
        Compute curvature at multiple scales for diagnostic purposes.
        
        Args:
            heightmap: Calibrated Heightmap from Layer 0
            
        Returns:
            dict with scale_name -> {
                "curvature": mean_curvature array,
                "gaussian_curvature": gaussian_curvature array,
                "curvature_type": classification array
            }
        """
        self._debug(f"Analyzing multi-scale curvature | scales={list(self.config.analysis_scales.keys())}")
        
        z = heightmap.data
        cell_size = heightmap.config.horizontal_scale
        results = {}
        
        for scale_name, radius in self.config.analysis_scales.items():
            self._debug(f"Processing scale: {scale_name} | sigma={radius}px")
            
            # Apply Gaussian blur at this scale
            z_scaled = ndimage.gaussian_filter(z, sigma=radius)
            
            # Compute curvature
            mean_curvature, gaussian_curvature = self._compute_curvature(z_scaled, cell_size)
            
            # Validate
            mean_curvature, gaussian_curvature = self._validate_curvature(
                mean_curvature, gaussian_curvature
            )
            
            # Classify with fixed epsilon from config (same value for H and K thresholds)
            eps = self.config.curvature_epsilon
            
            curvature_type = self._classify_curvature(
                mean_curvature, 
                gaussian_curvature, 
                h_epsilon=eps,
                k_epsilon=eps
            )
            
            results[scale_name] = {
                "curvature": mean_curvature.astype(np.float32),
                "gaussian_curvature": gaussian_curvature.astype(np.float32),
                "curvature_type": curvature_type
            }
        
        self._debug("Multi-scale analysis complete")
        return results
    
    def _compute_curvature(self, z: ScalarField, cell_size: float) -> Tuple[ScalarField, ScalarField]:
        """Compute mean and Gaussian curvature."""
        # First derivatives
        dz_dy, dz_dx = np.gradient(z, cell_size)
        
        # Second derivatives with symmetric mixed partial
        d2z_dy2, d2z_dxdy_1 = np.gradient(dz_dy, cell_size)
        d2z_dxdy_2, d2z_dx2 = np.gradient(dz_dx, cell_size)
        
        # Average mixed partial for symmetry
        d2z_dxdy = (d2z_dxdy_1 + d2z_dxdy_2) / 2.0
        
        # Mean curvature: H = (z_xx + z_yy) / 2
        mean_curvature = (d2z_dx2 + d2z_dy2) / 2.0
        
        # Gaussian curvature: K = z_xx * z_yy - (z_xy)²
        gaussian_curvature = d2z_dx2 * d2z_dy2 - d2z_dxdy * d2z_dxdy
        
        return mean_curvature, gaussian_curvature
    
    def _validate_curvature(self, H: ScalarField, K: ScalarField) -> Tuple[ScalarField, ScalarField]:
        """Handle NaN, Inf, and extreme values. H and K are validated independently."""
        H = np.asarray(H, dtype=np.float32)
        K = np.asarray(K, dtype=np.float32)
        
        # Handle NaN — if either field has NaN, zero both at those positions
        nan_mask = np.isnan(H) | np.isnan(K)
        if np.any(nan_mask):
            nan_count = int(np.sum(nan_mask))
            self._debug(f"NaN values detected: {nan_count} positions, filling with 0")
            H[nan_mask] = 0.0
            K[nan_mask] = 0.0
        
        # Handle Inf — clip bounds derived from units:
        #   H units: 1/m   → clip to [-1.0, 1.0]   (slope changes by 1m/m per meter — vertical wall)
        #   K units: 1/m²  → clip to [-1.0, 1.0]²  = [-1.0, 1.0] numerically, but K = κ₁·κ₂
        #                     worst case K = H_max² = 1.0 /m², so same numeric bound holds —
        #                     but semantically K << H for gentle terrain, so use H_clip² = 1.0
        H_CLIP = 1.0          # 1/m
        K_CLIP = H_CLIP ** 2  # 1/m² — principal curvatures both at H_max simultaneously
        
        if np.any(np.isinf(H)):
            inf_count = int(np.sum(np.isinf(H)))
            self._debug(f"Mean curvature (H) contains {inf_count} Inf values, clipping to [{-H_CLIP}, {H_CLIP}]")
            H = np.clip(H, -H_CLIP, H_CLIP)
        if np.any(np.isinf(K)):
            inf_count = int(np.sum(np.isinf(K)))
            self._debug(f"Gaussian curvature (K) contains {inf_count} Inf values, clipping to [{-K_CLIP}, {K_CLIP}]")
            K = np.clip(K, -K_CLIP, K_CLIP)
        
        return H, K
    
    def _classify_curvature(self, H: ScalarField, K: ScalarField, h_epsilon: float, k_epsilon: float) -> np.ndarray:
        """Classify each pixel into curvature type."""
        H = np.asarray(H)
        K = np.asarray(K)
        
        result = np.full(H.shape, "FLAT", dtype='<U10')
        
        saddle_mask  = K < -k_epsilon
        convex_mask  = (H >  h_epsilon) & (K >  k_epsilon) & ~saddle_mask
        concave_mask = (H < -h_epsilon) & (K >  k_epsilon) & ~saddle_mask
        
        result[saddle_mask]  = "SADDLE"
        result[convex_mask]  = "CONVEX"
        result[concave_mask] = "CONCAVE"
        
        return result
    
    def suggest_optimal_scale(self, heightmap: Heightmap, metric: str = 'variance') -> str:
        """
        Analyze curvature at all scales and suggest the optimal one.
        
        Args:
            heightmap: Calibrated Heightmap
            metric: 'variance', 'saddle_count', or 'non_flat_ratio'
            
        Returns:
            Name of optimal scale ('micro', 'meso', or 'macro')
        """
        self._debug(f"Suggesting optimal scale | metric={metric}")
        
        results = self.analyze(heightmap)
        scales = list(results.keys())
        
        if not scales:
            self._debug("No scales found, defaulting to 'meso'")
            return 'meso'
        
        scores = {}
        
        for scale_name, data in results.items():
            curvature = data['curvature']
            curvature_type = data['curvature_type']
            
            if metric == 'variance':
                # Higher variance = more feature detail
                scores[scale_name] = np.var(curvature)
                self._debug(f"Scale {scale_name}: variance={scores[scale_name]:.6f}")
                
            elif metric == 'saddle_count':
                # More saddles = more topological complexity
                scores[scale_name] = np.sum(curvature_type == "SADDLE")
                self._debug(f"Scale {scale_name}: saddle_count={scores[scale_name]}")
                
            elif metric == 'non_flat_ratio':
                # Ratio of non-flat pixels
                non_flat = np.sum(curvature_type != "FLAT")
                scores[scale_name] = non_flat / curvature_type.size
                self._debug(f"Scale {scale_name}: non_flat_ratio={scores[scale_name]:.4f}")
                
            else:
                # Default: combine variance and saddle count
                scores[scale_name] = np.var(curvature) * (1 + np.sum(curvature_type == "SADDLE") / curvature_type.size)
                self._debug(f"Scale {scale_name}: combined_score={scores[scale_name]:.6f}")
        
        # Return scale with highest score
        optimal = max(scores, key=scores.get)
        self._debug(f"Optimal scale: {optimal} (score={scores[optimal]:.6f})")
        
        return optimal
    
    def get_scale_statistics(self, heightmap: Heightmap) -> Dict[str, Dict]:
        """
        Get detailed statistics for each scale.
        
        Returns:
            dict with scale_name -> {
                'curvature_stats': {'min', 'max', 'mean', 'std'},
                'classification_counts': {label: count},
                'saddle_density': float
            }
        """
        self._debug("Computing scale statistics")
        
        results = self.analyze(heightmap)
        statistics = {}
        
        for scale_name, data in results.items():
            curvature = data['curvature']
            curvature_type = data['curvature_type']
            
            unique, counts = np.unique(curvature_type, return_counts=True)
            classification_counts = dict(zip(unique, counts))
            
            statistics[scale_name] = {
                'curvature_stats': {
                    'min': float(np.min(curvature)),
                    'max': float(np.max(curvature)),
                    'mean': float(np.mean(curvature)),
                    'std': float(np.std(curvature))
                },
                'classification_counts': classification_counts,
                'saddle_density': classification_counts.get('SADDLE', 0) / curvature_type.size
            }
            
            self._debug(f"Scale {scale_name}: saddle_density={statistics[scale_name]['saddle_density']:.4f}, curvature_std={statistics[scale_name]['curvature_stats']['std']:.6f}")
        
        return statistics
    
    def visualize_scales(self, heightmap: Heightmap, save_path: str = None):
        """
        Create a visualization comparing curvature at different scales.
        
        Args:
            heightmap: Calibrated Heightmap
            save_path: Optional path to save the figure
        """
        self._debug(f"Generating multi-scale visualization | save_path={save_path}")
        
        results = self.analyze(heightmap)
        scales = list(results.keys())
        
        fig, axes = plt.subplots(len(scales), 3, figsize=(15, 5 * len(scales)))
        
        # Handle single scale case
        if len(scales) == 1:
            axes = axes.reshape(1, -1)
        
        for idx, scale_name in enumerate(scales):
            data = results[scale_name]
            curvature = data['curvature']
            curvature_type = data['curvature_type']
            
            # Curvature map
            ax = axes[idx, 0]
            vmax = np.percentile(np.abs(curvature), 95)
            im = ax.imshow(curvature, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.set_title(f'{scale_name.title()} Scale: Mean Curvature\nσ={self.config.analysis_scales[scale_name]}px')
            ax.axis('off')
            plt.colorbar(im, ax=ax, shrink=0.8)
            
            # Curvature type
            ax = axes[idx, 1]
            type_to_int = {"FLAT": 0, "CONVEX": 1, "CONCAVE": 2, "SADDLE": 3}
            type_numeric = np.vectorize(type_to_int.get)(curvature_type)
            im = ax.imshow(type_numeric, cmap='Set3', vmin=0, vmax=3)
            ax.set_title(f'Classification')
            ax.axis('off')
            cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], shrink=0.8)
            cbar.ax.set_yticklabels(['FLAT', 'CONVEX', 'CONCAVE', 'SADDLE'], fontsize=8)
            
            # Histogram
            ax = axes[idx, 2]
            ax.hist(curvature.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
            ax.set_title(f'Curvature Distribution')
            ax.set_xlabel('Curvature (1/m)')
            ax.set_ylabel('Frequency')
            ax.axvline(0, color='red', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Multi-Scale Curvature Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self._debug(f"Visualization saved to: {save_path}")
        else:
            self._debug("Visualization displayed (not saved)")