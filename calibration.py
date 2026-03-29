"""
Layer 0: Calibration Module
"""

import hashlib
import json
import numpy as np
from pathlib import Path
from scipy import ndimage
from typing import Optional, Callable

from core import (
    Heightmap,
    NormalizationConfig,
    RawImageInput,
    PixelCoord,
    WorldCoord,
    PipelineConfig,
    ScalarField,
    Datacache
)


class Layer0_Calibration:
    """
    Stage 0: Convert raw imagery → calibrated mathematical surface.
    
    Clean, deterministic calibration with no georeferencing complexity.
    Output is a Heightmap with simple pixel-to-world scaling.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.cache = Datacache(
            layer_name="layer0",
            cache_dir=config.cache_dir
        )
    
    def _log(self, msg: str) -> None:
        """Helper for verbose logging"""
        if self.config.verbose:
            print(f"[Calibration] {msg}")
    
    def execute(self, input_data: RawImageInput) -> Heightmap:
        """
        Convert raw image to calibrated heightmap.
        
        Args:
            input_data: Raw grayscale image with optional metadata
            
        Returns:
            Heightmap: Clean, calibrated mathematical surface
        """
        # Validate input
        if isinstance(input_data, np.ndarray):
            # Auto-wrap raw array in RawImageInput
            input_data = RawImageInput(data=input_data)
        elif not hasattr(input_data, 'validate'):
            raise TypeError(f"Expected RawImageInput or np.ndarray, got {type(input_data)}")
            
        self._log(f"Validating input: {type(input_data).__name__}")
        
        if not input_data.validate():
            raise ValueError("Invalid input data format")
                
        # Build cache key
        cache_id = self._get_cache_id(input_data)
        
        # Check cache
        if self.cache.exists(cache_id, "config", "json") and self.cache.exists(cache_id, "elevation" ,"npy"):
            self._log(f"cache exists: {cache_id}")
            return self._load_from_cache(cache_id)
        
        # Compute fresh
        self._log('generating new cache...')
        result = self._compute(input_data)
        
        # Save to cache
        self._save_to_cache(cache_id, result)
        
        return result
    
    def _compute(self, input_data: RawImageInput) -> Heightmap:
        """Compute heightmap from raw input."""
        self._log(f"Computing heightmap: shape={input_data.data.shape}")
        
        # Derive calibration parameters
        norm_config = self._get_normalization_config(input_data)
        
        # Convert to elevation values
        elevation = self._normalize_elevation(input_data.data, norm_config)
        
        # Apply noise reduction
        if self.config.noise_reduction_sigma > 0:
            elevation = self._reduce_noise(elevation)
        
        # Validate output
        self._validate_surface(elevation)
        
        # Create Heightmap with simple coordinate transform
        return Heightmap(
            data=elevation.astype(np.float32),
            config=norm_config,
            pixel_to_world=self._make_pixel_transform(),
            origin=(0.0, 0.0)
        )
    
    def _get_normalization_config(self, input_data: RawImageInput) -> NormalizationConfig:
        """Derive calibration parameters from input metadata or config."""
        if input_data.metadata:
            h_scale = input_data.metadata.get('horizontal_scale', self.config.horizontal_scale)
            v_scale = input_data.metadata.get('vertical_scale', self.config.vertical_scale)
            offset = input_data.metadata.get('sea_level_offset', self.config.sea_level_offset)
            return NormalizationConfig(h_scale, v_scale, offset)
        
        return NormalizationConfig(
            horizontal_scale=self.config.horizontal_scale,
            vertical_scale=self.config.vertical_scale,
            sea_level_offset=self.config.sea_level_offset
        )
    
    def _normalize_elevation(self, raw_data: np.ndarray, config: NormalizationConfig) -> ScalarField:
        """Convert uint8 grayscale to elevation in meters."""
        elevations = raw_data.astype(np.float32)
        elevations = elevations * config.vertical_scale + config.sea_level_offset
        return elevations
    
    def _reduce_noise(self, elevation: ScalarField) -> ScalarField:
        """Apply Gaussian blur for noise reduction."""
        return ndimage.gaussian_filter(
            elevation,
            sigma=self.config.noise_reduction_sigma,
            mode='reflect'
        )
    
    def _make_pixel_transform(self) -> Callable[[PixelCoord], WorldCoord]:
        """Simple pixel-to-world transform using horizontal scale."""
        scale = self.config.horizontal_scale
        
        def transform(pixel: PixelCoord) -> WorldCoord:
            x, y = pixel
            return (x * scale, y * scale)
        
        return transform
    
    def _validate_surface(self, elevation: ScalarField) -> None:
        """Validate the calibrated surface."""
        if np.any(np.isnan(elevation)):
            raise ValueError("NaN values detected in calibrated surface")
        
        if np.any(np.isinf(elevation)):
            raise ValueError("Infinite values detected in calibrated surface")
        
        if self.config.verbose:
            elevation_range = elevation.max() - elevation.min()
            if elevation_range > self.config.max_elevation_range:
                self._log(f"Warning: Extreme elevation range: {elevation_range:.1f}m")
    
    # ---------------------------------------------------------------------
    # Caching mechanics
    # ---------------------------------------------------------------------
    
    def _get_cache_id(self, input_data: RawImageInput) -> str:
        """Generate deterministic cache ID from input and config."""
        # Hash the raw pixel data
        data_hash = hashlib.sha256(input_data.data.tobytes()).hexdigest()
        
        # Hash metadata if present
        if input_data.metadata:
            meta_str = json.dumps(input_data.metadata, sort_keys=True)
            meta_hash = hashlib.sha256(meta_str.encode()).hexdigest()
        else:
            meta_hash = ""
        
        # Hash relevant config parameters
        config_params = {
            "horizontal_scale": self.config.horizontal_scale,
            "vertical_scale": self.config.vertical_scale,
            "sea_level_offset": self.config.sea_level_offset,
            "noise_reduction_sigma": self.config.noise_reduction_sigma,
        }
        config_str = json.dumps(config_params, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()
        
        # Combine all hashes
        return self.cache.make_key(data_hash, meta_hash, config_hash)
    
    def _save_to_cache(self, cache_id: str, heightmap: Heightmap) -> None:
        """Save heightmap to cache."""
        # Save metadata as JSON
        metadata = {
            "layer": 0,
            "config": {
                "horizontal_scale": heightmap.config.horizontal_scale,
                "vertical_scale": heightmap.config.vertical_scale,
                "sea_level_offset": heightmap.config.sea_level_offset,
            },
            "shape": heightmap.shape,
            "dtype": str(heightmap.data.dtype),
            "array_file": f"{cache_id}.layer0.elevation.npy"
        }
        self.cache.save_json(cache_id, "config", metadata)
        
        # Save elevation array as NPY
        self.cache.save_array(cache_id, "elevation", heightmap.data)
    
    def _load_from_cache(self, cache_id: str) -> Heightmap:
        """Load heightmap from cache."""
        # Load metadata
        metadata = self.cache.load_json(cache_id, 'config')
        
        # Load elevation array
        elevation = self.cache.load_array(cache_id, "elevation")
        
        # Reconstruct normalization config
        norm_config = NormalizationConfig(
            horizontal_scale=metadata["config"]["horizontal_scale"],
            vertical_scale=metadata["config"]["vertical_scale"],
            sea_level_offset=metadata["config"]["sea_level_offset"]
        )
        
        # Reconstruct heightmap
        return Heightmap(
            data=elevation,
            config=norm_config,
            pixel_to_world=self._make_pixel_transform(),
            origin=(0.0, 0.0)
        )