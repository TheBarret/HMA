"""
Layer 1: Local Geometry Module

Computes first derivatives from calibrated heightmap:
- Slope magnitude (degrees)
- Aspect direction (radians, 0° = North, clockwise)
"""

import numpy as np
from typing import Dict

from core import Heightmap, ScalarField, PipelineConfig, PipelineLayer


class Layer1_LocalGeometry(PipelineLayer[Dict[str, ScalarField]]):
    """
    Compute first derivatives: slope magnitude and aspect direction.
    
    Uses central differences with proper spacing for real-world units.
    Aspect follows GIS convention: 0° = North, increasing clockwise.
    """
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
    
    def execute(self, input_data: Heightmap) -> Dict[str, ScalarField]:
        """
        Compute slope and aspect from calibrated heightmap.
        
        Args:
            input_data: Heightmap from Layer 0
            
        Returns:
            dict with 'slope' (degrees) and 'aspect' (radians) fields
        """
        if self.config.verbose:
            print(f"[Geometry] Computing slope/aspect | shape={input_data.shape}, cell_size={input_data.config.horizontal_scale:.4f}m")
        
        # Validate input
        if not isinstance(input_data, Heightmap):
            raise TypeError(f"Expected Heightmap, got {type(input_data)}")
        
        # Compute gradient with correct spacing
        slope_deg, aspect_rad = self._compute_gradient(input_data)
        
        # Validate and clean results
        if self.config.verbose:
            print("[Geometry] Validating derivative fields")
        slope_deg, aspect_rad = self._validate_derivatives(slope_deg, aspect_rad)
        
        # Convert to float32 for memory efficiency
        if self.config.verbose:
            print(f"[Geometry] Output: slope=[{np.min(slope_deg):.2f}°, {np.max(slope_deg):.2f}°]")
        
        return {
            "slope": slope_deg.astype(np.float32),
            "aspect": aspect_rad.astype(np.float32)
        }
    
    def _compute_gradient(self, heightmap: Heightmap) -> tuple[ScalarField, ScalarField]:
        """
        Compute gradient using central differences.

        Returns:
            tuple: (slope_degrees, aspect_radians)
        """
        if self.config.verbose:
            print("[Geometry] Computing central differences gradient")
        
        z = heightmap.data
        cell_size = heightmap.config.horizontal_scale
        
        dz_dy, dz_dx = np.gradient(z, cell_size)

        # Slope magnitude: arctan(sqrt((dz/dx)² + (dz/dy)²))
        # This is the angle of steepest ascent from horizontal
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = np.degrees(slope_rad)

        # Aspect: direction of steepest DESCENT (water flow direction)
        # GIS convention: 0° = North, increasing clockwise (90° = East)
        aspect_rad = np.arctan2(dz_dx, -dz_dy)
        aspect_rad = (aspect_rad + 2 * np.pi) % (2 * np.pi)

        return slope_deg, aspect_rad
    
    def _validate_derivatives(self, slope: ScalarField, aspect: ScalarField) -> tuple[ScalarField, ScalarField]:
        """
        Validate and clean derivative fields.

        Checks:
        - Slope within [0, 90] degrees
        - Aspect within [0, 2π] radians
        - Handle flat areas (slope ≈ 0)
        """
        # Check slope range
        if np.any(slope < 0):
            if self.config.verbose:
                print("[Geometry] Negative slope detected, clipping to 0")
            slope = np.clip(slope, 0, 90)

        if np.any(slope > 90):
            if self.config.verbose:
                print(f"[Geometry] Slope >90° detected (max={np.max(slope):.2f}°), clipping")
            slope = np.clip(slope, 0, 90)

        # Handle flat areas where aspect is undefined
        flat_mask = slope < self.config.flat_threshold_deg
        if np.any(flat_mask):
            if self.config.verbose:
                flat_pct = 100 * np.sum(flat_mask) / flat_mask.size
                print(f"[Geometry] Flat areas: {flat_pct:.1f}% of map, setting aspect=0")
            aspect = aspect.copy()  # avoid mutating input array
            aspect[flat_mask] = 0.0

        # Validate aspect range
        if np.any(aspect < 0) or np.any(aspect > 2 * np.pi):
            if self.config.verbose:
                print("[Geometry] Aspect outside [0,2π], normalizing")
            aspect = aspect % (2 * np.pi)

        # Check for NaN/Inf
        if np.any(np.isnan(slope)) or np.any(np.isnan(aspect)):
            raise ValueError("NaN values detected in derivatives")

        if np.any(np.isinf(slope)) or np.any(np.isinf(aspect)):
            raise ValueError("Infinite values detected in derivatives")

        return slope, aspect
    
    @property
    def output_schema(self) -> dict:
        return {
            "slope": {
                "type": "ScalarField", 
                "range": [0, 90], 
                "unit": "degrees",
                "description": "Steepest slope angle from horizontal"
            },
            "aspect": {
                "type": "ScalarField", 
                "range": [0, 6.28318], 
                "unit": "radians",
                "description": "Direction of steepest descent: 0=North, π/2=East, π=South, 3π/2=West"
            }
        }

# -----------------------------------------------------
# Sobel-based implementation for better noise immunity
# -----------------------------------------------------

class Layer1_LocalGeometry_Sobel(Layer1_LocalGeometry):
    """
    Alternative implementation using Sobel operators.
    
    Sobel provides built-in smoothing and is less sensitive to noise,
    but may slightly reduce spatial resolution.
    """
    
    def _compute_gradient(self, heightmap: Heightmap) -> tuple[ScalarField, ScalarField]:
        """Compute gradient using Sobel operators."""
        if self.config.verbose:
            print(f"[Geometry] Computing Sobel gradient | cell_size={heightmap.config.horizontal_scale:.4f}m")
        
        from scipy.ndimage import convolve

        z = heightmap.data
        cell_size = heightmap.config.horizontal_scale

        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / (8 * cell_size)
        # Flip sobel_y sign to match np.gradient axis-0 convention (North-up)
        sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / (8 * cell_size)

        dz_dx = convolve(z, sobel_x, mode='reflect')
        dz_dy = convolve(z, sobel_y, mode='reflect')

        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = np.degrees(slope_rad)

        # Arctan2: negate only dz_dx for downhill GIS convention
        aspect_rad = np.arctan2(dz_dx, -dz_dy)
        aspect_rad = (aspect_rad + 2 * np.pi) % (2 * np.pi)

        return slope_deg, aspect_rad