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
    
    CURVATURE_LABELS = ["FLAT", "CONVEX", "CONCAVE", "SADDLE"]
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.adaptive_epsilon = config.adaptive_epsilon  # Read from config
        self._epsilon_used: Optional[tuple[float, float]] = None
        
    def execute(self, input_data: Heightmap) -> Dict[str, ScalarField]:
        """
        Compute curvature fields from heightmap.
        
        Returns:
            dict with 'curvature' (mean curvature), 'gaussian_curvature', and 'curvature_type'
        """
        if not isinstance(input_data, Heightmap):
            raise TypeError(f"Expected Heightmap, got {type(input_data)}")
        
        z = input_data.data
        cell_size = input_data.config.horizontal_scale
        
        # Compute curvature
        mean_curvature, gaussian_curvature = self._compute_curvature(z, cell_size)
        
        # Validate and clean outputs
        mean_curvature, gaussian_curvature = self._validate_curvature(
            mean_curvature, gaussian_curvature
        )
        
        # Determine epsilon — adaptive uses both H and K independently
        h_epsilon, k_epsilon = self._get_epsilon(mean_curvature, gaussian_curvature)
        self._epsilon_used = (h_epsilon, k_epsilon)

        # Classify curvature type
        curvature_type = self._classify_curvature(
            mean_curvature, gaussian_curvature, h_epsilon, k_epsilon
        )

        # Log statistics
        self._log_classification_stats(curvature_type, h_epsilon, k_epsilon)
        
        return {
            "curvature": mean_curvature.astype(np.float32),
            "gaussian_curvature": gaussian_curvature.astype(np.float32),  # ← ADDED
            "curvature_type": curvature_type
        }
    
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
            print(f"Curvature contains {nan_count} NaN values - filling with 0")
            H[nan_mask] = 0.0
            K[nan_mask] = 0.0
        
        # Handle Inf — clip each field to its own reasonable range independently
        # H units: 1/m,  K units: 1/m² — different magnitudes, different clip bounds
        if np.any(np.isinf(H)):
            inf_count = int(np.sum(np.isinf(H)))
            print(f"Mean curvature (H) contains {inf_count} Inf values - clipping")
            H = np.clip(H, -1.0, 1.0)        # 1/m: reasonable bound for terrain
        if np.any(np.isinf(K)):
            inf_count = int(np.sum(np.isinf(K)))
            print(f"Gaussian curvature (K) contains {inf_count} Inf values - clipping")
            K = np.clip(K, -1.0, 1.0)        # 1/m²: same order of magnitude for terrain
        
        return H, K
    
    def _get_epsilon(self, H: ScalarField, K: ScalarField) -> tuple[float, float]:
        """
        Return (h_epsilon, k_epsilon) — independent thresholds for H and K.
        """
        if self.adaptive_epsilon:
            return self._compute_adaptive_epsilon(H, K)
        else:
            eps = float(self.config.curvature_epsilon)
            return eps, eps  # same value, but returned as tuple for consistency

    def _compute_adaptive_epsilon(self, H: ScalarField, K: ScalarField) -> tuple[float, float]:
        H_valid = H[~np.isnan(H)]
        K_valid = K[~np.isnan(K)]

        # Use 95th percentile of |H| and |K| as the signal anchor.
        # std() is dominated by the ~95% near-zero flat pixels on game maps,
        # producing thresholds so small that noise gets classified as features.
        # The 95th percentile sits in the actual feature signal, not the noise floor.
        h_anchor = float(np.percentile(np.abs(H_valid), 95))
        k_anchor = float(np.percentile(np.abs(K_valid), 95))

        print(f"_compute_adaptive_epsilon(pre): H_p95={h_anchor:.6f}, K_p95={k_anchor:.6f}")

        h_epsilon = max(self.config.curvature_epsilon_h_factor * h_anchor,
                        self.config.curvature_epsilon_h_min)
        k_epsilon = max(self.config.curvature_epsilon_k_factor * k_anchor,
                        self.config.curvature_epsilon_k_min)

        print(f"_compute_adaptive_epsilon(post): h_epsilon={h_epsilon:.6f}, k_epsilon={k_epsilon:.6f}")
        return h_epsilon, k_epsilon
    
    def _classify_curvature(self, H: ScalarField, K: ScalarField, h_epsilon: float, k_epsilon: float) -> np.ndarray:
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
    
    def _log_classification_stats(self, curvature_type: np.ndarray, h_epsilon: float, k_epsilon: float) -> None:
        """Log classification statistics."""
        total        = curvature_type.size
        flat_count   = np.sum(curvature_type == "FLAT")
        convex_count = np.sum(curvature_type == "CONVEX")
        concave_count= np.sum(curvature_type == "CONCAVE")
        saddle_count = np.sum(curvature_type == "SADDLE")
        
        non_flat = convex_count + concave_count + saddle_count
        if non_flat > 1000:
            print(
                f"\n-> Curvature classification (H_ε={h_epsilon:.6f} 1/m, K_ε={k_epsilon:.6f} 1/m²): "
                f"\n        CONVEX={convex_count} ({100*convex_count/total:.1f}%), "
                f"\n        CONCAVE={concave_count} ({100*concave_count/total:.1f}%), "
                f"\n        SADDLE={saddle_count} ({100*saddle_count/total:.1f}%), "
                f"\n        FLAT={flat_count} ({100*flat_count/total:.1f}%)"
            )
    
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
        z = heightmap.data
        cell_size = heightmap.config.horizontal_scale
        results = {}
        
        for scale_name, radius in self.config.analysis_scales.items():
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
        """Handle NaN, Inf, and extreme values."""
        H = np.asarray(H, dtype=np.float32)
        K = np.asarray(K, dtype=np.float32)
        
        # Handle NaN
        if np.any(np.isnan(H)):
            H = np.nan_to_num(H, nan=0.0)
            K = np.nan_to_num(K, nan=0.0)
        
        # Handle Inf
        if np.any(np.isinf(H)):
            H = np.clip(H, -1.0, 1.0)
            K = np.clip(K, -1.0, 1.0)
        
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
        results = self.analyze(heightmap)
        scales = list(results.keys())
        
        if not scales:
            return 'meso'
        
        scores = {}
        
        for scale_name, data in results.items():
            curvature = data['curvature']
            curvature_type = data['curvature_type']
            
            if metric == 'variance':
                # Higher variance = more feature detail
                scores[scale_name] = np.var(curvature)
                
            elif metric == 'saddle_count':
                # More saddles = more topological complexity
                scores[scale_name] = np.sum(curvature_type == "SADDLE")
                
            elif metric == 'non_flat_ratio':
                # Ratio of non-flat pixels
                non_flat = np.sum(curvature_type != "FLAT")
                scores[scale_name] = non_flat / curvature_type.size
                
            else:
                # Default: combine variance and saddle count
                scores[scale_name] = np.var(curvature) * (1 + np.sum(curvature_type == "SADDLE") / curvature_type.size)
        
        # Return scale with highest score
        optimal = max(scores, key=scores.get)
        
        # Log the scores for debugging
        print(f"\nMulti-scale analysis scores ({metric}):")
        for scale, score in scores.items():
            print(f"  {scale}: {score:.6f}")
        print(f"Optimal scale: {optimal}")
        
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
        
        return statistics
    
    def visualize_scales(self, heightmap: Heightmap, save_path: str = None):
        """
        Create a visualization comparing curvature at different scales.
        
        Args:
            heightmap: Calibrated Heightmap
            save_path: Optional path to save the figure
        """
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
            print(f"Saved multi-scale visualization to: {save_path}")