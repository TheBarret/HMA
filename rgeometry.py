"""
Layer 2: Regional Geometry Module
"""

import numpy as np
import hashlib
import json
from scipy import ndimage
from typing import Dict, Tuple, Optional

from core import Heightmap, ScalarField, PipelineConfig, PipelineLayer, Datacache

class Layer2_RegionalGeometry(PipelineLayer[Dict[str, ScalarField]]):
    """
    Compute second derivatives: curvature classification.
    """
    
    CURVATURE_LABELS = ["FLAT", "SADDLE", "CONVEX", "CONCAVE"]
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.cache = Datacache(
            layer_name="layer2",
            cache_dir=config.cache_dir
        )
        self.adaptive_epsilon = config.adaptive_epsilon  # Read from config
        self._epsilon_used: Optional[tuple[float, float]] = None
    
    def _log(self, msg: str) -> None:
        """Helper for verbose logging"""
        if self.config.verbose:
            print(f"[Regional] {msg}")
        
    def execute(self, input_data: Heightmap) -> Dict[str, ScalarField]:
        """
        Compute curvature fields from heightmap.
        
        Returns:
            dict with 'curvature' (mean curvature), 'gaussian_curvature', and 'curvature_type'
        """
        self._log(f"Computing curvature fields, shape={input_data.shape}, cell_size={input_data.config.horizontal_scale:.4f}m")
        
        if not isinstance(input_data, Heightmap):
            raise TypeError(f"Expected Heightmap, got {type(input_data)}")
        
        cache_id = self._get_cache_id(input_data)
        
        # Check cache
        if (self.cache.exists(cache_id, "config", "json") and 
            self.cache.exists(cache_id, "mean", "npy") and 
            self.cache.exists(cache_id, "gauss", "npy") and 
            self.cache.exists(cache_id, "type", "npy")):
            return self._load_from_cache(cache_id)
       
        # get hash
        result = self._compute_hash(input_data)
        
        z = input_data.data
        cell_size = input_data.config.horizontal_scale
        
        # Compute curvature
        mean_curvature, gaussian_curvature = self._compute_curvature(z, cell_size)
        
        # Validate and clean outputs
        self._log("Validating curvature fields")
        mean_curvature, gaussian_curvature = self._validate_curvature(
            mean_curvature, gaussian_curvature
        )
        
        # Determine epsilon — adaptive uses both H and K independently
        self._log("Determining epsilon thresholds")
        h_epsilon, k_epsilon = self._get_epsilon(mean_curvature, gaussian_curvature)
        self._epsilon_used = (h_epsilon, k_epsilon)

        # Classify curvature type
        self._log(f"Classifying curvature | H_eps={h_epsilon:.6f}, K_eps={k_epsilon:.6f}")
        curvature_type = self._classify_curvature(
            mean_curvature, gaussian_curvature, h_epsilon, k_epsilon
        )

        # Log statistics
        self._log_classification_stats(curvature_type, h_epsilon, k_epsilon)
        
        # prepare
        result = {
            "curvature": mean_curvature.astype(np.float32),
            "gaussian_curvature": gaussian_curvature.astype(np.float32),
            "curvature_type": curvature_type
        }
        
        # post snapshot
        self._save_to_cache(cache_id, result)
        
        # commit
        return result
    
    def _compute_curvature(self, z: ScalarField, cell_size: float) -> Tuple[ScalarField, ScalarField]:
        """Compute mean and Gaussian curvature using full differential geometry formulas."""
        self._log("Computing gradients and second derivatives")
        
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
            self._log(f"NaN values detected: {nan_count} positions, filling with 0")
            H[nan_mask] = 0.0
            K[nan_mask] = 0.0
        
        # Handle Inf — clip each field to its own reasonable range independently
        # H units: 1/m,  K units: 1/m² — different magnitudes, different clip bounds
        if np.any(np.isinf(H)):
            inf_count = int(np.sum(np.isinf(H)))
            self._log(f"Mean curvature (H) contains {inf_count} Inf values, clipping to [-1.0, 1.0]")
            H = np.clip(H, -1.0, 1.0)        # 1/m: reasonable bound for terrain
        if np.any(np.isinf(K)):
            inf_count = int(np.sum(np.isinf(K)))
            self._log(f"Gaussian curvature (K) contains {inf_count} Inf values, clipping to [-1.0, 1.0]")
            K = np.clip(K, -1.0, 1.0)        # 1/m²: same order of magnitude for terrain
        
        return H, K
    
    def _get_epsilon(self, H: ScalarField, K: ScalarField) -> tuple[float, float]:
        """
        Return (h_epsilon, k_epsilon) — independent thresholds for H and K.
        """
        
        if self.adaptive_epsilon:
            self._log(f"Using adaptive epsilon [{self.config.adaptive_percentile} percentile]")
            return self._compute_adaptive_epsilon(H, K)
        else:
            eps = float(self.config.curvature_epsilon)
            self._log(f"Using fixed epsilon: {eps:.6f}")
            return eps, eps

    def _compute_adaptive_epsilon(self, H: ScalarField, K: ScalarField) -> tuple[float, float]:
        """Compute adaptive thresholds with both floor AND ceiling."""
        
        # added: over-inflation clamps on K and H
        
        H_valid = H[~np.isnan(H)]
        K_valid = K[~np.isnan(K)]
        
        # Find non-flat regions
        H_abs = np.abs(H)
        H_threshold = np.percentile(H_abs, 50)
        non_flat_mask = H_abs > H_threshold
        
        if np.sum(non_flat_mask) < 100:
            h_anchor = np.percentile(np.abs(H_valid), 75)
            k_anchor = np.percentile(np.abs(K_valid), 75)
        else:
            h_anchor = np.percentile(H_abs[non_flat_mask], 75)
            k_anchor = np.percentile(np.abs(K)[non_flat_mask], 75)
        
        # Apply factor
        h_raw = self.config.curvature_epsilon_h_factor * h_anchor
        k_raw = self.config.curvature_epsilon_k_factor * k_anchor
        
        # Clamp between min/max
        self._log(f"Epsilon clamp range MIN: H={self.config.curvature_epsilon_h_min}, K={self.config.curvature_epsilon_k_min}")
        self._log(f"Epsilon clamp range MAX: H={self.config.curvature_epsilon_h_max}, K={self.config.curvature_epsilon_k_max}")
        h_epsilon = np.clip(
            h_raw, 
            self.config.curvature_epsilon_h_min, 
            self.config.curvature_epsilon_h_max
        )
        k_epsilon = np.clip(
            k_raw, 
            self.config.curvature_epsilon_k_min, 
            self.config.curvature_epsilon_k_max
        )
        
        self._log(f"Adaptive epsilon is set to H={h_epsilon:.6f}, K={k_epsilon:.6f}")
        return h_epsilon, k_epsilon
       
       
    # Classification
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
        
        self._log(f"Classification results: CONVEX={convex_count} ({100*convex_count/total:.1f}%), CONCAVE={concave_count} ({100*concave_count/total:.1f}%), SADDLE={saddle_count} ({100*saddle_count/total:.1f}%), FLAT={flat_count} ({100*flat_count/total:.1f}%)")
    
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
    
    # ---------------------------------------------------------------------
    # Interfaces
    # ---------------------------------------------------------------------
    
    @property
    def epsilon_used(self) -> Optional[tuple[float, float]]:
        """Get (h_epsilon, k_epsilon) used in the last classification."""
        return self._epsilon_used
        
    # ---------------------------------------------------------------------
    # Caching mechanics
    # ---------------------------------------------------------------------
    
    def _compute_hash(self, heightmap: Heightmap) -> str:
        # Hash the elevation data
        data_hash = hashlib.sha256(heightmap.data.tobytes()).hexdigest()
        
        # Hash all config parameters that affect curvature computation
        config_params = {
            "adaptive_epsilon": self.config.adaptive_epsilon,
            "curvature_epsilon": self.config.curvature_epsilon,
            "curvature_epsilon_h_factor": self.config.curvature_epsilon_h_factor,
            "curvature_epsilon_k_factor": self.config.curvature_epsilon_k_factor,
            "adaptive_percentile": self.config.adaptive_percentile,
            "curvature_epsilon_h_min": self.config.curvature_epsilon_h_min,
            "curvature_epsilon_h_max": self.config.curvature_epsilon_h_max,
            "curvature_epsilon_k_min": self.config.curvature_epsilon_k_min,
            "curvature_epsilon_k_max": self.config.curvature_epsilon_k_max,
        }
        config_str = json.dumps(config_params, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()
        
        return self.cache.make_key(data_hash, config_hash)
    
    def _get_cache_id(self, heightmap: Heightmap) -> str:
        # Hash elevation data
        data_hash = hashlib.sha256(heightmap.data.tobytes()).hexdigest()
        
        # Hash all config parameters that affect curvature computation
        config_params = {
            "adaptive_epsilon": self.config.adaptive_epsilon,
            "curvature_epsilon": self.config.curvature_epsilon,
            "curvature_epsilon_h_factor": self.config.curvature_epsilon_h_factor,
            "curvature_epsilon_k_factor": self.config.curvature_epsilon_k_factor,
            "adaptive_percentile": self.config.adaptive_percentile,
            "curvature_epsilon_h_min": self.config.curvature_epsilon_h_min,
            "curvature_epsilon_h_max": self.config.curvature_epsilon_h_max,
            "curvature_epsilon_k_min": self.config.curvature_epsilon_k_min,
            "curvature_epsilon_k_max": self.config.curvature_epsilon_k_max,
        }
        config_str = json.dumps(config_params, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()
        
        return self.cache.make_key(data_hash, config_hash)
    
    def _save_to_cache(self, cache_id: str, result: Dict[str, ScalarField]) -> None:
        """Save curvature results to cache."""
        
        # self._epsilon_used = numpy float32
        h_eps, k_eps = self._epsilon_used
        _epsilon_used_primitive = (float(h_eps), float(k_eps))
        
        metadata = {
            "layer": 2,
            "adaptive_epsilon": self.config.adaptive_epsilon,
            "epsilon_used": _epsilon_used_primitive,
            "shape": result["curvature"].shape,
            "dtype": str(result["curvature"].dtype)
        }
        self.cache.save_json(cache_id, "config", metadata)
        self.cache.save_array(cache_id, "mean", result["curvature"])
        self.cache.save_array(cache_id, "gauss", result["gaussian_curvature"])
        self.cache.save_array(cache_id, "type", result["curvature_type"])
    
    def _load_from_cache(self, cache_id: str) -> Dict[str, ScalarField]:
        """Load curvature results from cache."""
        metadata = self.cache.load_json(cache_id, "config")
        
        # self._epsilon_used = numpy float32
        if "epsilon_used" in metadata:
            eps = metadata["epsilon_used"]
            self._epsilon_used = (float(eps[0]), float(eps[1]))
        
        # Load arrays
        mean = self.cache.load_array(cache_id, "mean")
        gauss = self.cache.load_array(cache_id, "gauss")
        ctype = self.cache.load_array(cache_id, "type")
        
        return {
            "curvature": mean,
            "gaussian_curvature": gauss,
            "curvature_type": ctype
        }
