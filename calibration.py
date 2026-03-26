"""
Layer 0: Calibration Module
"""

import numpy as np
from scipy import ndimage
from typing import Optional, Callable, Tuple

from core import (
    Heightmap, 
    NormalizationConfig, 
    RawImageInput,
    PixelCoord, 
    WorldCoord,
    PipelineConfig,
    ScalarField
)

class Layer0_Calibration:
    """
    Stage 0: Convert raw imagery → calibrated mathematical surface.
    
    This stage is domain-agnostic and only concerned with establishing
    the measurement framework. No terrain analysis occurs here.
    
    Design principles:
    1. All transformations are invertible when possible
    2. Metadata drives calibration when available
    3. Noise reduction preserves structural features
    4. Output is pure geometric surface with no interpretation
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._pixel_transform: Optional[Callable[[PixelCoord], WorldCoord]] = None
    
    def execute(
            self, 
            input_data,  # 'RawImageInput', 'np.ndarray'
            normalization_config: Optional[NormalizationConfig] = None,
            georeferencing: Optional[dict] = None
        ) -> Heightmap:
        """
        Convert raw image to calibrated heightmap.
        
        Args:
            input_data: Raw grayscale image with optional metadata
            normalization_config: Explicit calibration parameters.
                                  If None, attempts to derive from metadata.
            georeferencing: Optional world coordinate mapping.
                            If None, assumes identity transform.
                            
        Returns:
            Heightmap: Clean, calibrated mathematical surface
        """
        # 1. Validate input — handle both RawImageInput and raw numpy arrays
        if isinstance(input_data, np.ndarray):
            # Auto-wrap raw array in RawImageInput
            input_data = RawImageInput(data=input_data)
        elif not hasattr(input_data, 'validate'):
            raise TypeError(f"Expected RawImageInput or np.ndarray, got {type(input_data)}")
        
        if self.config.verbose:
            print(f"[Calibration] Validating input: {type(input_data).__name__}")
        
        if not input_data.validate():
            raise ValueError("Invalid input data format")
        
        # 2. Derive calibration parameters
        if normalization_config is None:
            if self.config.verbose:
                print("[Calibration] Deriving calibration from metadata")
            normalization_config = self._derive_calibration(input_data)
        elif self.config.verbose:
            print(f"[Calibration] Using provided calibration: h_scale={normalization_config.horizontal_scale}, v_scale={normalization_config.vertical_scale}")
        
        # 3. Define coordinate transform
        if self.config.verbose:
            print("[Calibration] Building coordinate transform")
        self._pixel_transform = self._build_coordinate_transform(
            georeferencing, 
            normalization_config.horizontal_scale
        )
        
        # 4. Convert to elevation values
        if self.config.verbose:
            print(f"[Calibration] Converting to elevation: shape={input_data.data.shape}")
        elevation_data = self._normalize_elevation(
            input_data.data, 
            normalization_config
        )
        
        # 5. Apply noise reduction (Gaussian blur)
        if self.config.verbose and self.config.noise_reduction_sigma > 0:
            print(f"[Calibration] Applying Gaussian blur: sigma={self.config.noise_reduction_sigma}")
        clean_elevation = self._reduce_noise(elevation_data)
        
        # 6. Validate output quality
        if self.config.verbose:
            print("[Calibration] Validating output surface")
        self._validate_surface(clean_elevation)
        
        # 7. Create immutable Heightmap
        if self.config.verbose:
            print("[Calibration] Creating Heightmap object")
        return Heightmap(
            data=clean_elevation,
            config=normalization_config,
            pixel_to_world=self._pixel_transform,
            origin=self._get_origin(georeferencing)
        )
    
    def _derive_calibration(self, input_data: RawImageInput) -> NormalizationConfig:
        """
        Derive calibration parameters from available metadata.
        
        This is a heuristic approach when explicit metadata isn't provided.
        In production, this should be replaced with actual georeferencing data.
        """
        # Attempt to extract from metadata
        if input_data.metadata:
            if 'horizontal_scale' in input_data.metadata:
                h_scale = input_data.metadata['horizontal_scale']
                v_scale = input_data.metadata.get('vertical_scale', self.config.vertical_scale)
                offset  = input_data.metadata.get('sea_level_offset', self.config.sea_level_offset)
                return NormalizationConfig(h_scale, v_scale, offset)

        return NormalizationConfig(
                    horizontal_scale=self.config.horizontal_scale,
                    vertical_scale=self.config.vertical_scale,
                    sea_level_offset=self.config.sea_level_offset
                )
        
    def _normalize_elevation(
        self, 
        raw_data: np.ndarray, 
        config: NormalizationConfig
    ) -> ScalarField:
        """
        Convert uint8 grayscale values to real-world elevation in meters.
        
        The conversion is linear: z = (value * vertical_scale) + offset
        
        Returns:
            float32 array of elevations in meters
        """
        # Convert to float to avoid overflow
        elevations = raw_data.astype(np.float32)
        
        # Apply linear transformation
        elevations = elevations * config.vertical_scale + config.sea_level_offset
        
        return elevations
    
    def _reduce_noise(self, elevation_data: ScalarField) -> ScalarField:
        """
        Apply Gaussian blur for noise reduction.
        
        Gaussian blur is chosen because:
        1. It's a low-pass filter that preserves large-scale structure
        2. It's differentiable (important for subsequent derivative calculations)
        3. The kernel is separable for performance
        4. It provides scale-space representation naturally
        """
        if self.config.noise_reduction_sigma > 0:
            return ndimage.gaussian_filter(
                elevation_data, 
                sigma=self.config.noise_reduction_sigma,
                mode='reflect'  # Reflect boundary to avoid edge artifacts
            )
        return elevation_data
    
    def _build_coordinate_transform(
        self, 
        georeferencing: Optional[dict],
        horizontal_scale: float
    ) -> Callable[[PixelCoord], WorldCoord]:
        """
        Build transformation from pixel coordinates to world coordinates.
        
        Supports:
        - Identity transform (no georeferencing)
        - Affine transform (with rotation, scaling, translation)
        - Polynomial transform (future)
        """
        if georeferencing is None:
            # Identity: (x, y) → (x * scale, y * scale)
            if self.config.verbose:
                print("[Calibration] Using identity coordinate transform")
            def identity_transform(pixel: PixelCoord) -> WorldCoord:
                x, y = pixel
                return (x * horizontal_scale, y * horizontal_scale)
            return identity_transform
        
        # Affine transform: [a b tx; c d ty] * [x; y; 1]
        if 'affine' in georeferencing:
            if self.config.verbose:
                print("[Calibration] Using affine coordinate transform")
            affine = georeferencing['affine']
            def affine_transform(pixel: PixelCoord) -> WorldCoord:
                x, y = pixel
                world_x = affine[0] * x + affine[1] * y + affine[2]
                world_y = affine[3] * x + affine[4] * y + affine[5]
                return (world_x, world_y)
            return affine_transform
        
        # Fallback to simple scaling
        if self.config.verbose:
            print("[Calibration] Unrecognized georeferencing, falling back to simple scaling")
        return lambda pixel: (pixel[0] * horizontal_scale, pixel[1] * horizontal_scale)
    
    def _get_origin(self, georeferencing: Optional[dict]) -> WorldCoord:
        """Extract world coordinate origin from georeferencing."""
        if georeferencing and 'origin' in georeferencing:
            return tuple(georeferencing['origin'])
        return (0.0, 0.0)
    
    def _validate_surface(self, elevation_data: ScalarField) -> None:
        """
        Validate the calibrated surface for analysis readiness.
        
        Checks:
        1. No NaN or infinite values
        2. Reasonable elevation range
        3. Smoothness (optional)
        """
        if np.any(np.isnan(elevation_data)):
            raise ValueError("NaN values detected in calibrated surface")
        
        if np.any(np.isinf(elevation_data)):
            raise ValueError("Infinite values detected in calibrated surface")
        
        # warn about extreme elevation ranges
        elevation_range = elevation_data.max() - elevation_data.min()
        if elevation_range > self.config.max_elevation_range:  
            if self.config.verbose:
                print(f"[Calibration] Extreme elevation range: {elevation_range:.1f}m")
        
        # Check for artificial steps
        # This could indicate poor calibration or quantization artifacts
        hist, bin_edges = np.histogram(elevation_data, bins=self.config.histogram_bins)
        
        if np.max(hist) > 0.9 * len(elevation_data.flat):
            if elevation_range < 0.01:
                if self.config.verbose:
                    print("[Calibration] Highly quantized values - check vertical scale")
            else:
                if self.config.verbose:
                    print(f"[Calibration] Highly quantized values (range={elevation_range:.2f}m) - vertical scale may be too coarse")


class Layer0_Calibration_With_QualityMetrics(Layer0_Calibration):
    """
    Extended version that provides quality metrics alongside the heightmap.
    
    This allows for diagnostic feedback about the calibration process.
    """
    
    def execute_with_metrics(
        self, 
        input_data: RawImageInput,
        normalization_config: Optional[NormalizationConfig] = None,
        georeferencing: Optional[dict] = None
    ) -> Tuple[Heightmap, dict]:
        """
        Execute calibration and return quality metrics.
        
        Returns:
            Tuple[Heightmap, dict]: (calibrated surface, quality metrics)
        """
        if self.config.verbose:
            print("[Calibration] Executing with quality metrics")
        
        heightmap = self.execute(input_data, normalization_config, georeferencing)
        
        metrics = {
            'elevation_stats': {
                'min': float(np.min(heightmap.data)),
                'max': float(np.max(heightmap.data)),
                'mean': float(np.mean(heightmap.data)),
                'std': float(np.std(heightmap.data)),
                'range': float(np.ptp(heightmap.data))
            },
            'noise_estimate': self._estimate_noise_level(heightmap.data),
            'data_completeness': 1.0,  # No missing data in basic implementation
            'calibration': {
                'horizontal_scale': heightmap.config.horizontal_scale,
                'vertical_scale': heightmap.config.vertical_scale,
                'sea_level_offset': heightmap.config.sea_level_offset
            },
            'shape': heightmap.shape
        }
        
        if self.config.verbose:
            print(f"[Calibration] Metrics: range={metrics['elevation_stats']['range']:.2f}m, noise={metrics['noise_estimate']:.4f}")
        
        return heightmap, metrics
    
    def _estimate_noise_level(self, elevation_data: ScalarField) -> float:
        """
        Estimate noise level using median absolute deviation of high-pass filtered data.
        
        This is a simple estimator that works for natural terrain.
        """
        # High-pass filter using Laplacian
        high_pass = ndimage.laplace(elevation_data)
        
        # Robust estimate of scale (MAD)
        mad = np.median(np.abs(high_pass - np.median(high_pass)))
        
        # Convert to standard deviation assuming Gaussian noise
        noise_std = mad / self.config.std_gaussian
        
        return float(noise_std)