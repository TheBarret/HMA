# Layer 3: Topological Features

import numpy as np
from scipy import ndimage
from scipy.ndimage import label, find_objects, maximum_filter, minimum_filter
from skimage import measure, morphology
from skimage.morphology import skeletonize, remove_small_objects
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
from uuid import uuid4
import warnings

from core import (
    Heightmap, ScalarField, PipelineConfig, PipelineLayer,
    TerrainFeature, ClassifiedFeature, CurvatureType, Traversability,
    PeakFeature, RidgeFeature, ValleyFeature, SaddleFeature, FlatZoneFeature
)
from core import PixelCoord, WorldCoord


class Layer3_TopologicalFeatures(PipelineLayer[List[TerrainFeature]]):
    """
    Resolve continuous fields into discrete terrain structures.
    
    Now uses both mean curvature (H) and Gaussian curvature (K) for:
    - Feature confidence weighting
    - Ridge/valley width estimation
    - Saddle point verification
    - Feature hierarchy establishment
    """
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self._min_feature_size = 5          # Minimum pixels for a feature
        self._min_ridge_length = 10          # Minimum ridge length in pixels
        self._ridge_width = 3                # Expected ridge width in pixels
        self._saddle_confidence_threshold = 0.1  # Top 30% by K magnitude
        self._saddle_k_min_threshold = 5e-5  # Hard minimum |K| for saddle (noise floor)
        # Derived from diagnostic: |K| > 5e-5 retains ~20% of saddle pixels (real features)
        # |K| < 5e-5 is indistinguishable from flat-terrain numerical noise on gentle maps
        self._border_margin = 10             # Ignore features within N pixels of image edge
        # Edge pixels have extreme curvature artifacts from Gaussian blur reflect boundary
        
    def execute(self, input_data: Dict) -> List[TerrainFeature]:
        """
        Extract discrete terrain features from continuous fields.
        
        Args:
            input_data: LayerBundle containing:
                - heightmap: Heightmap from Layer 0
                - curvature: Mean curvature field (H)
                - gaussian_curvature: Gaussian curvature field (K)
                - curvature_type: Curvature classification
                - slope: Slope field (degrees)
                - aspect: Aspect field (radians)
                
        Returns:
            List of TerrainFeature objects
        """
        # Extract from bundle
        heightmap = input_data['heightmap']
        mean_curvature = input_data['curvature']
        gaussian_curvature = input_data['gaussian_curvature']
        curvature_type = input_data['curvature_type']
        slope = input_data.get('slope', None)
        aspect = input_data.get('aspect', None)
        
        # Validate inputs
        if slope is None or aspect is None:
            warnings.warn("Slope/aspect not provided, some feature refinement will be limited")
        
        # Calculate confidence weights from Gaussian curvature magnitude
        # Higher |K| = more confident feature detection
        k_confidence = self._calculate_confidence_weights(gaussian_curvature)
        
        features = []
        
        # 1. Extract peaks (CONVEX + local max + high K confidence)
        peaks = self._extract_peaks(heightmap, curvature_type, gaussian_curvature, 
                                    k_confidence, mean_curvature, slope)
        features.extend(peaks)
        
        # 2. Extract ridges (linear CONVEX features)
        ridges = self._extract_ridges(heightmap, curvature_type, gaussian_curvature,
                                      k_confidence, mean_curvature, slope, aspect)
        features.extend(ridges)
        
        # 3. Extract valleys (linear CONCAVE features)
        valleys = self._extract_valleys(heightmap, curvature_type, gaussian_curvature,
                                        k_confidence, mean_curvature, slope, aspect)
        features.extend(valleys)
        
        # 4. Extract saddles (SADDLE points with negative K)
        saddles = self._extract_saddles(heightmap, curvature_type, gaussian_curvature,
                                        k_confidence, mean_curvature, slope)
        features.extend(saddles)
        
        # 5. Extract flat zones (FLAT regions with low slope)
        flat_zones = self._extract_flat_zones(heightmap, curvature_type, slope)
        features.extend(flat_zones)
        
        # 6. Build feature hierarchy (peaks connect to ridges, ridges to saddles)
        self._build_feature_hierarchy(features)
        
        # Log statistics
        print(f"\n=== Layer 3: Feature Extraction ===")
        print(f"Peaks: {len(peaks)}")
        print(f"Ridges: {len(ridges)}")
        print(f"Valleys: {len(valleys)}")
        print(f"Saddles: {len(saddles)}")
        print(f"Flat zones: {len(flat_zones)}")
        print(f"Total features: {len(features)}")
        
        # Log confidence distribution
        if peaks:
            avg_confidence = np.mean([p.metadata.get('confidence', 0) for p in peaks])
            print(f"Average peak confidence: {avg_confidence:.3f}")
        
        return features
    
    def _calculate_confidence_weights(self, gaussian_curvature: ScalarField) -> np.ndarray:
        """Calculate confidence weights based on |K| magnitude."""
        k_abs = np.abs(gaussian_curvature)
        
        # Use 90th percentile as reference for strong features
        k_90 = np.percentile(k_abs, 90) if np.any(k_abs > 0) else 1.0
        
        # Confidence = normalized |K| (capped at 1.0)
        confidence = np.clip(k_abs / k_90, 0, 1.0)
        
        return confidence
    
    def _extract_peaks(self, heightmap: Heightmap, curvature_type: np.ndarray,
                       gaussian_curvature: np.ndarray, k_confidence: np.ndarray,
                       mean_curvature: np.ndarray, slope: Optional[np.ndarray]) -> List[PeakFeature]:
        """Extract peaks from convex regions with high K confidence."""
        z = heightmap.data
        peaks = []
        
        # Mask: convex regions with positive Gaussian curvature
        convex_mask = (curvature_type == "CONVEX")
        
        # Refine: only pixels with K > 0 (true convex)
        k_positive = gaussian_curvature > 0
        convex_mask = convex_mask & k_positive
        
        # Find local maxima (pixel higher than 8 neighbors)
        local_max = (z == maximum_filter(z, size=3))
        
        # Combine: convex region, positive K, AND local maximum
        peak_mask = convex_mask & local_max
        
        # Apply confidence threshold (only high-confidence peaks)
        high_confidence = k_confidence > self._saddle_confidence_threshold
        peak_mask = peak_mask & high_confidence
        
        # Label connected components
        labeled, num_features = label(peak_mask)
        
        for i in range(1, num_features + 1):
            mask = (labeled == i)
            size = np.sum(mask)
            if size < self._min_feature_size:
                continue
                
            # Find centroid (highest point)
            ys, xs = np.where(mask)
            elevations = z[mask]
            highest_idx = np.argmax(elevations)
            centroid_px = (xs[highest_idx], ys[highest_idx])
            
            # Get elevation stats
            elevation_range = (np.min(z[mask]), np.max(z[mask]))
            
            # Calculate prominence
            prominence = self._calculate_prominence(heightmap, centroid_px)
            
            # Calculate average slope
            avg_slope = np.mean(slope[mask]) if slope is not None else 15.0
            
            # Average confidence for this peak
            avg_confidence = np.mean(k_confidence[mask])
            
            # Estimate width from curvature magnitude
            h_peak = mean_curvature[centroid_px[1], centroid_px[0]]
            k_peak = gaussian_curvature[centroid_px[1], centroid_px[0]]
            
            peak = PeakFeature(
                centroid=centroid_px,
                elevation_range=elevation_range,
                prominence=prominence,
                _avg_slope=float(avg_slope), 
                avg_slope=avg_slope,
                metadata={
                    'size': size,
                    'elevation': float(z[centroid_px[1], centroid_px[0]]),
                    'confidence': float(avg_confidence),
                    'mean_curvature': float(h_peak),
                    'gaussian_curvature': float(k_peak)
                }
            )
            peaks.append(peak)
        
        return peaks
    
    def _extract_ridges(self, heightmap: Heightmap, curvature_type: np.ndarray,
                        gaussian_curvature: np.ndarray, k_confidence: np.ndarray,
                        mean_curvature: np.ndarray, slope: Optional[np.ndarray],
                        aspect: Optional[np.ndarray]) -> List[RidgeFeature]:
        """Extract ridge lines from convex regions with skeletonization."""
        z = heightmap.data
        ridges = []
        
        # Mask: convex regions with positive K
        convex_mask = (curvature_type == "CONVEX") & (gaussian_curvature > 0)
        
        # Clean small regions
        cleaned = remove_small_objects(convex_mask, min_size=self._min_feature_size)
        
        # Skeletonize to get ridge spine
        skeleton = skeletonize(cleaned)
        
        # Remove small skeletons
        skeleton = remove_small_objects(skeleton, min_size=self._min_ridge_length)
        
        # Label ridge segments
        labeled, num_features = label(skeleton)
        
        for i in range(1, num_features + 1):
            mask = (labeled == i)
            ys, xs = np.where(mask)
            
            if len(xs) < self._min_ridge_length:
                continue
                
            # Get spine points (ordered along ridge)
            spine_points = self._order_spine_points(xs, ys)
            
            # Calculate centroid (midpoint of spine)
            centroid_px = (int(np.mean(xs)), int(np.mean(ys)))
            
            # Get elevation stats along ridge
            elevations = z[ys, xs]
            elevation_range = (np.min(elevations), np.max(elevations))
            
            # Find connected peaks
            connected_peaks = self._find_connected_peaks(heightmap, spine_points)
            
            # Average slope and confidence along ridge
            avg_slope = np.mean(slope[ys, xs]) if slope is not None else 10.0
            avg_confidence = np.mean(k_confidence[ys, xs])
            
            # Estimate ridge width from curvature
            ridge_curvatures = mean_curvature[ys, xs]
            avg_curvature = np.mean(np.abs(ridge_curvatures))
            cell_size = heightmap.config.horizontal_scale
            width_meters = 1.0 / (avg_curvature * cell_size) if avg_curvature > 0 else self._ridge_width * cell_size
            
            ridge = RidgeFeature(
                centroid=centroid_px,
                elevation_range=elevation_range,
                spine_points=spine_points,
                connected_peaks=connected_peaks,
                metadata={
                    'length': len(spine_points),
                    'width_meters': float(width_meters),
                    'confidence': float(avg_confidence),
                    'avg_curvature': float(avg_curvature)
                }
            )
            ridges.append(ridge)
        
        return ridges
    
    def _extract_valleys(self, heightmap: Heightmap, curvature_type: np.ndarray,
                     gaussian_curvature: np.ndarray, k_confidence: np.ndarray,
                     mean_curvature: np.ndarray, slope: Optional[np.ndarray],
                     aspect: Optional[np.ndarray]) -> List[ValleyFeature]:
        """Extract valley lines from concave regions."""
        z = heightmap.data
        valleys = []
        
        # Mask: concave regions with positive K (bowl-shaped)
        concave_mask = (curvature_type == "CONCAVE") & (gaussian_curvature > 0)
        
        # Clean and skeletonize
        cleaned = remove_small_objects(concave_mask, min_size=self._min_feature_size)
        skeleton = skeletonize(cleaned)
        skeleton = remove_small_objects(skeleton, min_size=self._min_ridge_length)
        
        labeled, num_features = label(skeleton)
        
        for i in range(1, num_features + 1):
            mask = (labeled == i)
            ys, xs = np.where(mask)
            
            if len(xs) < self._min_ridge_length:
                continue
                
            spine_points = self._order_spine_points(xs, ys)
            centroid_px = (int(np.mean(xs)), int(np.mean(ys)))
            
            elevations = z[ys, xs]
            elevation_range = (np.min(elevations), np.max(elevations))
            
            avg_slope = np.mean(slope[ys, xs]) if slope is not None else 10.0
            avg_confidence = np.mean(k_confidence[ys, xs])
            
            # Calculate approximate drainage area (simplified)
            drainage_area = self._estimate_drainage_area(heightmap, spine_points, aspect)
            
            valley = ValleyFeature(
                centroid=centroid_px,
                elevation_range=elevation_range,
                spine_points=spine_points,
                drainage_area=drainage_area,
                metadata={
                    'length': len(spine_points),
                    'confidence': float(avg_confidence),
                    'avg_slope': float(avg_slope),
                    'avg_curvature': float(np.mean(mean_curvature[ys, xs]))
                }
            )
            valleys.append(valley)
        
        return valleys


    def _extract_saddles(self, heightmap: Heightmap, curvature_type: np.ndarray,
                         gaussian_curvature: np.ndarray, k_confidence: np.ndarray,
                         mean_curvature: np.ndarray, slope: Optional[np.ndarray]) -> List[SaddleFeature]:
        """
        Extract saddle points where K < 0.

        Two-stage filtering:
        1. Hard minimum |K| threshold — rejects numerical noise on flat terrain.
           The confidence weight alone is insufficient because on gentle maps the
           90th-percentile anchor is itself tiny, making 0.3 × anchor ≈ noise floor.
        2. Border margin — edge pixels always have extreme curvature artifacts from
           the Gaussian blur reflect boundary and must be excluded.
        """
        z = heightmap.data
        h, w = z.shape
        m = self._border_margin
        saddles = []

        # Mask: classified SADDLE with negative K AND above hard noise floor
        saddle_mask = (
            (curvature_type == "SADDLE") &
            (gaussian_curvature < 0) &
            (np.abs(gaussian_curvature) > self._saddle_k_min_threshold)
        )

        # Strip border pixels — reflect padding creates false curvature spikes at edges
        border = np.zeros_like(saddle_mask)
        border[m:h-m, m:w-m] = True
        saddle_mask = saddle_mask & border

        # Additional: confidence filter on top of hard threshold
        confident_saddle = k_confidence > self._saddle_confidence_threshold
        saddle_mask = saddle_mask & confident_saddle

        labeled, num_features = label(saddle_mask)
        
        for i in range(1, num_features + 1):
            mask = (labeled == i)
            if np.sum(mask) < 3:
                continue
                
            ys, xs = np.where(mask)
            centroid_px = (int(np.mean(xs)), int(np.mean(ys)))
            
            elevation = z[centroid_px[1], centroid_px[0]]
            k_value = gaussian_curvature[centroid_px[1], centroid_px[0]]
            h_value = mean_curvature[centroid_px[1], centroid_px[0]]
            
            # Find connecting features (will be populated in Layer 4)
            connections = self._find_saddle_connections(heightmap, centroid_px)
            
            saddle = SaddleFeature(
                centroid=centroid_px,
                elevation_range=(elevation, elevation),
                elevation=elevation,
                connecting_ridges=connections['ridges'],
                connecting_valleys=connections['valleys'],
                k_curvature=float(k_value),
                metadata={
                    'size': int(np.sum(mask)),
                    'mean_curvature': float(h_value),
                    'confidence': float(abs(k_value)),
                    'avg_slope': float(slope[centroid_px[1], centroid_px[0]]) if slope is not None else 10.0
                }
            )
            saddles.append(saddle)
        
        return saddles


    def _extract_flat_zones(self, heightmap: Heightmap, curvature_type: np.ndarray,
                            slope: Optional[np.ndarray]) -> List[FlatZoneFeature]:
        """Extract flat zones for traversability."""
        z = heightmap.data
        cell_size = heightmap.config.horizontal_scale
        flat_zones = []
        
        # Mask: flat curvature
        flat_mask = (curvature_type == "FLAT")
        
        # Refine: low slope (< 5°)
        if slope is not None:
            flat_mask = flat_mask & (slope < 5.0)
        
        labeled, num_features = label(flat_mask)
        
        for i in range(1, num_features + 1):
            mask = (labeled == i)
            size = np.sum(mask)
            if size < self._min_feature_size * 10:  # Minimum 50 pixels for flat zone
                continue
                
            ys, xs = np.where(mask)
            centroid_px = (int(np.mean(xs)), int(np.mean(ys)))
            
            elevations = z[ys, xs]
            elevation_range = (np.min(elevations), np.max(elevations))
            
            # Calculate bounds for contains_point
            x_min, x_max = np.min(xs), np.max(xs)
            y_min, y_max = np.min(ys), np.max(ys)
            
            # Slope statistics
            if slope is not None:
                slopes_in_zone = slope[ys, xs]
                max_slope = float(np.max(slopes_in_zone))
                min_slope = float(np.min(slopes_in_zone))
                avg_slope = float(np.mean(slopes_in_zone))
            else:
                max_slope = 0.0
                min_slope = 0.0
                avg_slope = 0.0
            
            flat_zone = FlatZoneFeature(
                centroid=centroid_px,
                elevation_range=elevation_range,
                area_pixels=size,
                max_slope=max_slope,
                min_slope=min_slope,
                metadata={
                    'size': size,
                    'area_m2': size * (cell_size ** 2),
                    'bounds': (x_min, x_max, y_min, y_max),
                    'avg_slope': avg_slope
                }
            )
            flat_zones.append(flat_zone)
        
        return flat_zones


    def _estimate_drainage_area(self, heightmap: Heightmap, spine_points: List[PixelCoord],
                                aspect: Optional[np.ndarray]) -> Optional[float]:
        """Estimate drainage area for valley (simplified)."""
        if aspect is None or not spine_points:
            return None
        
        # Simplified: area is proportional to valley length * average width
        # Will be properly calculated in Layer 4 with flow accumulation
        length = len(spine_points)
        cell_area = heightmap.config.horizontal_scale ** 2
        return length * cell_area * 10  # Rough estimate
    
    def _calculate_prominence(self, heightmap: Heightmap, peak_px: PixelCoord) -> float:
        """Calculate prominence of a peak."""
        z = heightmap.data
        x, y = peak_px
        peak_elev = z[y, x]
        
        # Find lowest saddle connecting to higher terrain
        radius = int(100 / heightmap.config.horizontal_scale)
        y_min = max(0, y - radius)
        y_max = min(heightmap.shape[1], y + radius)
        x_min = max(0, x - radius)
        x_max = min(heightmap.shape[0], x + radius)
        
        local_min = np.min(z[y_min:y_max, x_min:x_max])
        prominence = peak_elev - local_min
        
        return float(prominence)
    
    def _order_spine_points(self, xs: np.ndarray, ys: np.ndarray) -> List[PixelCoord]:
        """
        Order ridge/valley skeleton points as a continuous polyline.

        Angle-sort from centroid fails for linear features because it interleaves
        points from both sides of the line, producing a zigzag spine.

        Instead: nearest-neighbour chain starting from one endpoint.
        Endpoint = point with fewest neighbours within 2px (degree-1 node on skeleton).
        Falls back to the point furthest from centroid if no degree-1 node found.
        """
        if len(xs) < 2:
            return [(int(xs[0]), int(ys[0]))]

        points = np.column_stack([xs, ys]).astype(np.float32)
        n = len(points)

        # Find a good start point: prefer degree-1 skeleton node (an endpoint)
        start_idx = 0
        min_neighbours = n
        cx, cy = np.mean(xs), np.mean(ys)
        max_dist = -1

        for i in range(n):
            dists = np.sqrt(np.sum((points - points[i]) ** 2, axis=1))
            neighbours = np.sum((dists > 0) & (dists <= 2.0))
            if neighbours < min_neighbours:
                min_neighbours = neighbours
                start_idx = i
            dist_from_centroid = (xs[i] - cx) ** 2 + (ys[i] - cy) ** 2
            if dist_from_centroid > max_dist:
                max_dist = dist_from_centroid
                fallback_idx = i

        if min_neighbours > 2:
            start_idx = fallback_idx  # No clear endpoint, use furthest point

        # Nearest-neighbour chain: always move to closest unvisited point
        visited = np.zeros(n, dtype=bool)
        ordered_idx = [start_idx]
        visited[start_idx] = True

        for _ in range(n - 1):
            current = points[ordered_idx[-1]]
            dists = np.sqrt(np.sum((points - current) ** 2, axis=1))
            dists[visited] = np.inf
            nearest = np.argmin(dists)
            ordered_idx.append(nearest)
            visited[nearest] = True

        return [(int(xs[i]), int(ys[i])) for i in ordered_idx]
    
    def _find_connected_peaks(self, heightmap: Heightmap, spine_points: List[PixelCoord]) -> Set[str]:
        """Find peak IDs connected to this ridge."""
        # Simplified - will be populated when peaks have IDs and distance matching
        return set()
    
    def _find_saddle_connections(self, heightmap: Heightmap, saddle_px: PixelCoord) -> Dict:
        """Find ridges and valleys connected to saddle."""
        # Simplified - will be populated in Layer 4
        return {'ridges': set(), 'valleys': set()}
    
    def _build_feature_hierarchy(self, features: List[TerrainFeature]) -> None:
        """Build hierarchical relationships between features."""
        # Simplified - will be enhanced in Layer 4
        pass
    
    @property
    def output_schema(self) -> dict:
        return {
            "type": "List[TerrainFeature]",
            "feature_types": ["PeakFeature", "RidgeFeature", "ValleyFeature", "SaddleFeature", "FlatZoneFeature"],
            "geometry": "points and polylines in PixelCoord space"
        }