"""
Layer 3: Topological Features
"""
import numpy as np
from scipy import ndimage
from scipy.ndimage import label, find_objects, maximum_filter, minimum_filter
from skimage.morphology import skeletonize, remove_small_objects
from typing import List, Dict, Tuple, Set, Optional, Any
from dataclasses import field

from core import (
    Heightmap, ScalarField, PipelineConfig, PipelineLayer,
    TerrainFeature, ClassifiedFeature, CurvatureType, FeatureType, Traversability,
    PeakFeature, RidgeFeature, ValleyFeature, SaddleFeature, FlatZoneFeature,
    PixelCoord, WorldCoord
)

class Layer3_TopologicalFeatures(PipelineLayer[List[TerrainFeature]]):
    """
    Resolve continuous fields into discrete terrain structures.
    """

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self._min_feature_size = config.min_peak_size_px
        self._min_ridge_length = config.min_ridge_length_px
        self._ridge_width = config.min_ridge_length_px // 3
        self._saddle_confidence_threshold = config.saddle_confidence_threshold
        self._saddle_k_min_threshold = config.saddle_k_min_threshold
        self._border_margin = config.border_margin_px
        
        self._annular_inner_px = max(3, int(config.peak_annular_inner_m / config.horizontal_scale))
        self._annular_outer_px = max(7, int(config.peak_annular_outer_m / config.horizontal_scale))
    
    def _log(self, msg: str) -> None:
        """Helper for verbose logging"""
        if self.config.verbose:
            print(f"[Topological] {msg}")

    def execute(self, input_data: Dict[str, Any]) -> List[TerrainFeature]:
        """Extract discrete terrain features from continuous fields."""
        heightmap = input_data['heightmap']
        mean_curvature = input_data['curvature']
        gaussian_curvature = input_data['gaussian_curvature']
        curvature_type = input_data['curvature_type']
        slope = input_data.get('slope', None)
        aspect = input_data.get('aspect', None)

        if slope is None or aspect is None:
            self._log("Slope/aspect not provided, feature refinement limited")

        self._log(f"Starting feature extraction | shape={heightmap.shape}, cell_size={heightmap.config.horizontal_scale:.4f}m")
        
        k_confidence = self._calculate_confidence_weights(gaussian_curvature)
        features = []

        self._log("Extracting peaks")
        peaks = self._extract_peaks(heightmap, curvature_type, gaussian_curvature,
                                    k_confidence, mean_curvature, slope)
        features.extend(peaks)
        self._log(f"Peaks extracted: {len(peaks)}")

        self._log("Extracting ridges")
        ridges = self._extract_ridges(heightmap, curvature_type, gaussian_curvature,
                                      k_confidence, mean_curvature, slope, aspect)
        features.extend(ridges)
        self._log(f"Ridges extracted: {len(ridges)}")

        self._log("Extracting valleys")
        valleys = self._extract_valleys(heightmap, curvature_type, gaussian_curvature,
                                        k_confidence, mean_curvature, slope, aspect)
        features.extend(valleys)
        self._log(f"Valleys extracted: {len(valleys)}")

        self._log("Extracting saddles")
        saddles = self._extract_saddles(heightmap, curvature_type, gaussian_curvature,
                                        k_confidence, mean_curvature, slope)
        features.extend(saddles)
        self._log(f"Saddles extracted: {len(saddles)}")

        self._log("Extracting flat zones")
        flat_zones = self._extract_flat_zones(heightmap, curvature_type, slope)
        features.extend(flat_zones)
        self._log(f"Flat zones extracted: {len(flat_zones)}")

        # sea-level control filter
        original_count = len(features)
        features = self._purge_sea_domain(features)
        if (original_count - len(features)) == 0:
            self._log(f"No features detected, seal-level parameters might be too aggressive")
            
        if self.config.verbose and original_count != len(features):
            self._log(f"sea-level filter: {original_count} → {len(features)} features ({original_count - len(features)} removed)")
            
        # commit
        self._build_feature_hierarchy(features)

        self._log(f"Extraction complete | total={len(features)}, peaks={len(peaks)}, ridges={len(ridges)}, valleys={len(valleys)}, saddles={len(saddles)}, flat_zones={len(flat_zones)}")
        if peaks:
            avg_confidence = np.mean([p.metadata.get('confidence', 0) for p in peaks])
            self._log(f"Average peak confidence: {avg_confidence:.3f}")

        return features

    def _purge_sea_domain(self, features: List[TerrainFeature]) -> List[TerrainFeature]:
        """
        Filter out features below the elevation reference threshold.
        
        Uses feature's MAX elevation for peaks/ridges (crest matters),
        but MIN elevation for valleys (bottom matters for drainage).
        
        Returns:
            List of features that meet elevation criteria
        """
        if not self.config.exclude_below_reference:
            return features  # No filtering, return all
        
        ref = self.config.elevation_reference_m
        filtered = []
        filtered_count = 0
        
        for feature in features:
            min_z, max_z = feature.elevation_range
            
            # Type-specific elevation logic
            if isinstance(feature, (PeakFeature, RidgeFeature)):
                # For peaks/ridges, use MAX elevation (crest/summit)
                # A ridge that peaks above reference is valid even if base is low
                passes = max_z >= ref
            elif isinstance(feature, (ValleyFeature, FlatZoneFeature)):
                # For valleys/flats, use MIN elevation (bottom)
                # A valley entirely below reference may be "submerged"
                passes = min_z >= ref
            elif isinstance(feature, SaddleFeature):
                # Saddles are passes - use average elevation
                avg_z = (min_z + max_z) / 2.0
                passes = avg_z >= ref
            else:
                # Default: use max elevation (conservative)
                passes = max_z >= ref
            
            if passes:
                filtered.append(feature)
            else:
                filtered_count += 1
                # track filtered features for debugging
                feature.metadata['filtered_reason'] = f'below_reference_{ref}m'
        
        if self.config.verbose and filtered_count > 0:
            self._log(f"Purged {filtered_count} features below {ref}m reference")
        
        return filtered

    def _calculate_confidence_weights(self, gaussian_curvature: ScalarField) -> np.ndarray:
        """Calculate confidence weights based on |K| magnitude."""
        k_abs = np.abs(gaussian_curvature)
        non_zero = k_abs[k_abs > 0]
        if len(non_zero) > 0:
            k_90 = np.percentile(non_zero, 90)
            k_90 = max(k_90, 1e-8)
        else:
            k_90 = 1.0
        confidence = np.clip(k_abs / k_90, 0, 1.0)
        return confidence

    def _is_curvature(self, arr: np.ndarray, ctype: CurvatureType) -> np.ndarray:
        """Dtype-agnostic comparison: works for object arrays of enums AND int-encoded arrays."""
        return arr == ctype.name
    
    def _extract_peaks(self, heightmap, curvature_type, gaussian_curvature,
               k_confidence, mean_curvature, slope) -> List[PeakFeature]:
        """
        Extract peaks using regional maxima + curvature validation.
        
        For peak validation, we consider both CONVEX and CYLINDRICAL_CONVEX
        as "convex enough" because peak flanks are often cylindrical.
        """
        from skimage import morphology
        from scipy import ndimage

        z = heightmap.data
        cell_size = heightmap.config.horizontal_scale
        peaks = []

        z_smooth = ndimage.gaussian_filter(z, sigma=self.config.peak_smooth_sigma)
        regional_max_mask = morphology.local_maxima(z_smooth, connectivity=2)
        labeled_peaks, num_candidates = ndimage.label(regional_max_mask)

        self._log(f"Peak candidates: {num_candidates} regional maxima")
        
        if num_candidates == 0:
            self._log("No regional maxima found, using elevation fallback")
            return self._extract_peaks_fallback(heightmap, slope)

        shoulder_radius_inner = max(2, int(self.config.peak_annular_inner_m / cell_size)) 
        shoulder_radius_outer = max(10, int(self.config.peak_annular_outer_m / cell_size)) 

        # --- PATCH: hoist grid creation outside the candidate loop ---
        y_grid, x_grid = np.ogrid[:z.shape[0], :z.shape[1]]

        validated_count = 0
        for i in range(1, num_candidates + 1):
            peak_pixels = np.argwhere(labeled_peaks == i)
            if len(peak_pixels) < self._min_feature_size:
                continue

            centroid_y, centroid_x = peak_pixels.mean(axis=0).astype(int)

            dist = np.sqrt((x_grid - centroid_x)**2 + (y_grid - centroid_y)**2)
            shoulder_mask = (dist >= shoulder_radius_inner) & (dist <= shoulder_radius_outer)

            shoulder_types = curvature_type[shoulder_mask]
            if len(shoulder_types) < 10:
                continue

            convex_count = np.sum(
                self._is_curvature(shoulder_types, CurvatureType.CONVEX) |
                self._is_curvature(shoulder_types, CurvatureType.CYLINDRICAL_CONVEX)
            )
            convex_ratio = convex_count / len(shoulder_types)
                      
            if convex_ratio >= self.config.peak_shoulder_convex_ratio:
                prominence = self._calculate_prominence(heightmap, (centroid_x, centroid_y))
                if prominence > self.config.peak_min_prominence_m:
                    mask = (labeled_peaks == i)
                    peaks.append(self._create_peak_feature(
                        heightmap, mask, (centroid_x, centroid_y), prominence,
                        mean_curvature, gaussian_curvature, slope,
                        method="curvature_annular"
                    ))
                    validated_count += 1
        
        self._log(f"Peaks validated: {validated_count} (convex_ratio threshold={self.config.peak_shoulder_convex_ratio})")

        if len(peaks) == 0:
            self._log("No curvature-validated peaks detected, using elevation fallback")
            return self._extract_peaks_fallback(heightmap, slope)

        return peaks

    def _extract_peaks_fallback(self, heightmap, slope) -> List[PeakFeature]:
        """Elevation-based fallback for maps where curvature fails."""
        z = heightmap.data
        peaks = []

        local_max_fallback = (z == ndimage.maximum_filter(z, size=7))
        border_mask = np.zeros_like(z, dtype=bool)
        border_mask[self._border_margin:-self._border_margin,
                    self._border_margin:-self._border_margin] = True
        local_max_fallback = local_max_fallback & border_mask

        labeled, num_features = ndimage.label(local_max_fallback)
        self._log(f"Fallback candidates: {num_features} local maxima")
        
        validated_count = 0
        for i in range(1, num_features + 1):
            mask = (labeled == i)
            if np.sum(mask) < self._min_feature_size * 2:
                continue

            ys, xs = np.where(mask)
            centroid_px = (int(np.mean(xs)), int(np.mean(ys)))
            prominence = self._calculate_prominence(heightmap, centroid_px)

            if prominence > self.config.peak_min_prominence_m:
                peaks.append(self._create_peak_feature(
                    heightmap, mask, centroid_px, prominence,
                    None, None, slope,
                    method="elevation_fallback"
                ))
                validated_count += 1
        
        self._log(f"Fallback peaks validated: {validated_count}")
        return peaks

    def _create_peak_feature(self, heightmap, mask, centroid_px, prominence,
                             mean_curvature, gaussian_curvature, slope, method="curvature"):
        """Helper to create a PeakFeature with consistent metadata."""
        z = heightmap.data
        x, y = centroid_px
        elevation = float(z[y, x])

        avg_slope = np.mean(slope[mask]) if slope is not None else 10.0
        defensive_rating = min(1.0, (prominence / 15.0) * (1 - abs(avg_slope - 15) / 30))
        confidence_v = float(min(1.0, prominence / self.config.peak_confidence))
        
        peak = PeakFeature(
            centroid=centroid_px,
            elevation_range=(float(np.min(z[mask])), elevation),
            prominence=prominence,
            feature_type=FeatureType.PEAK,
            confidence=confidence_v,
            metadata={
                'size': int(np.sum(mask)),
                'elevation': elevation,
                'confidence': confidence_v,
                'defensive_rating': defensive_rating,
                'detection_method': method,
                'mean_curvature': float(mean_curvature[y, x]) if mean_curvature is not None else 0,
                'gaussian_curvature': float(gaussian_curvature[y, x]) if gaussian_curvature is not None else 0
            }
        )
        peak._avg_slope = float(avg_slope)
        return peak

    def _extract_ridges(self, heightmap: Heightmap, curvature_type: np.ndarray,
                    gaussian_curvature: np.ndarray, k_confidence: np.ndarray,
                    mean_curvature: np.ndarray, slope: Optional[np.ndarray],
                    aspect: Optional[np.ndarray]) -> List[RidgeFeature]:
        """
        Extract ridge lines from convex and cylindrical convex regions.
        
        With the new classification:
        - CONVEX: dome-like features (peak tops, rounded ridges)
        - CYLINDRICAL_CONVEX: true ridge lines (linear convex features)
        """
        
        z = heightmap.data
        ridges = []

        # CONVEX and CYLINDRICAL_CONVEX for ridge detection
        convex_mask = self._is_curvature(curvature_type, CurvatureType.CONVEX)
        cylindrical_convex_mask = self._is_curvature(curvature_type, CurvatureType.CYLINDRICAL_CONVEX)
        ridge_mask = convex_mask | cylindrical_convex_mask
        
        ridge_pixels = np.sum(ridge_mask)
        self._log(f"Ridge mask pixels: {ridge_pixels} ({100*ridge_pixels/ridge_mask.size:.1f}%)")
        
        cleaned = remove_small_objects(ridge_mask, min_size=self._min_feature_size)
        skeleton = skeletonize(cleaned)
        skeleton = remove_small_objects(skeleton, min_size=self._min_ridge_length)
        labeled, num_features = label(skeleton)
        
        self._log(f"Ridge candidates: {num_features} (min_length={self._min_ridge_length}px)")
        
        validated_count = 0
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
            
            ridge_curvatures = mean_curvature[ys, xs]
            avg_curvature = np.mean(np.abs(ridge_curvatures))
            cell_size = heightmap.config.horizontal_scale
            width_meters = 1.0 / (avg_curvature * cell_size) if avg_curvature > 0 else self._ridge_width * cell_size
            
            # Determine ridge type from curvature classification
            ridge_type = "cylindrical"
            if np.any(self._is_curvature(curvature_type[ys, xs], CurvatureType.CONVEX)):
                ridge_type = "dome_crest"
            
            ridge = RidgeFeature(
                centroid=centroid_px,
                elevation_range=elevation_range,
                spine_points=spine_points,
                feature_type=FeatureType.RIDGE,
                confidence=float(avg_confidence),
                metadata={
                    'length': len(spine_points),
                    'width_meters': float(width_meters),
                    'confidence': float(avg_confidence),
                    'avg_curvature': float(avg_curvature),
                    'ridge_type': ridge_type,
                    'detection_method': 'curvature_classification'
                }
            )
            ridge._avg_slope = float(avg_slope)
            ridges.append(ridge)
            validated_count += 1
        
        self._log(f"Ridges validated: {validated_count}")
        return ridges

    def _extract_valleys(self, heightmap: Heightmap, curvature_type: np.ndarray,
                     gaussian_curvature: np.ndarray, k_confidence: np.ndarray,
                     mean_curvature: np.ndarray, slope: Optional[np.ndarray],
                     aspect: Optional[np.ndarray]) -> List[ValleyFeature]:
        """
        Extract valley lines from concave and cylindrical concave regions.
        
        With the new classification:
        - CONCAVE: bowl-like features (depressions, valley bottoms)
        - CYLINDRICAL_CONCAVE: true valley lines (linear concave features)
        """
        
        z = heightmap.data
        valleys = []

        # CONCAVE and CYLINDRICAL_CONCAVE for valley detection
        concave_mask = self._is_curvature(curvature_type, CurvatureType.CONCAVE)
        cylindrical_concave_mask = self._is_curvature(curvature_type, CurvatureType.CYLINDRICAL_CONCAVE)
        valley_mask = concave_mask | cylindrical_concave_mask
        
        valley_pixels = np.sum(valley_mask)
        self._log(f"Valley mask pixels: {valley_pixels} ({100*valley_pixels/valley_mask.size:.1f}%)")
        
        cleaned = remove_small_objects(valley_mask, min_size=self._min_feature_size)
        skeleton = skeletonize(cleaned)
        skeleton = remove_small_objects(skeleton, min_size=self._min_ridge_length)
        labeled, num_features = label(skeleton)
        
        self._log(f"Valley candidates: {num_features} (min_length={self._min_ridge_length}px)")
        
        validated_count = 0
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
            drainage_area = self._estimate_drainage_area(heightmap, spine_points, aspect)
            
            # Determine valley type from curvature classification
            valley_type = "cylindrical"
            if np.any(self._is_curvature(curvature_type[ys, xs], CurvatureType.CONCAVE)):
                valley_type = "bowl_bottom"
            
            valley = ValleyFeature(
                centroid=centroid_px,
                elevation_range=elevation_range,
                spine_points=spine_points,
                drainage_area=drainage_area,
                feature_type=FeatureType.VALLEY,
                confidence=float(avg_confidence),
                metadata={
                    'length': len(spine_points),
                    'confidence': float(avg_confidence),
                    'avg_slope': float(avg_slope),
                    'avg_curvature': float(np.mean(mean_curvature[ys, xs])),
                    'valley_type': valley_type,
                    'detection_method': 'curvature_classification'
                }
            )
            valley._avg_slope = float(avg_slope)
            valleys.append(valley)
            validated_count += 1
        
        self._log(f"Valleys validated: {validated_count}")
        return valleys

    def _extract_saddles(self, heightmap, curvature_type, gaussian_curvature,
                     k_confidence, mean_curvature, slope) -> List[SaddleFeature]:
        """
        Extract saddles directly from curvature classification.
        
        Uses two filtering criteria:
        1. k_magnitude: Absolute Gaussian curvature magnitude (geometric strength)
        2. confidence: Normalized confidence score (0-1) relative to map's strongest saddle
        """
        from scipy.spatial import KDTree
        
        z = heightmap.data
        h, w = z.shape
        m = self._border_margin
        saddles = []
        
        # Use the curvature classification directly!
        saddle_mask = self._is_curvature(curvature_type, CurvatureType.SADDLE)
        
        # Apply border margin
        border_mask = np.zeros_like(saddle_mask)
        border_mask[m:h-m, m:w-m] = True
        saddle_mask = saddle_mask & border_mask
        
        # Elevation floor (avoid water/sea-level artifacts)
        elev_floor = heightmap.config.sea_level_offset
        saddle_mask = saddle_mask & (z > elev_floor + 1.0)
        
        candidate_ys, candidate_xs = np.where(saddle_mask)
        self._log(f"Saddle candidates: {len(candidate_xs)} from curvature classification")
        
        if len(candidate_xs) == 0:
            return saddles
        
        # Get K magnitudes and confidence scores
        k_magnitudes = np.abs(gaussian_curvature[candidate_ys, candidate_xs])
        confidences = k_confidence[candidate_ys, candidate_xs]  # Already normalized 0-1
        
        # NMS: group nearby candidates, keeping the strongest (highest K magnitude)
        coords = np.column_stack([candidate_xs, candidate_ys]).astype(np.float32)
        
        # Sort by K magnitude (strongest saddles first)
        sorted_idx = np.argsort(k_magnitudes)[::-1]
        
        tree = KDTree(coords)
        suppressed = np.zeros(len(candidate_xs), dtype=bool)
        radius_px = self.config.peak_nms_radius_px  # Reuse NMS radius
        
        for idx in sorted_idx:
            if suppressed[idx]:
                continue
            # Keep this saddle, suppress neighbors
            neighbors = tree.query_ball_point(coords[idx], r=radius_px)
            for nb in neighbors:
                if nb != idx:
                    suppressed[nb] = True
        
        final_xs = candidate_xs[~suppressed]
        final_ys = candidate_ys[~suppressed]
        final_k = k_magnitudes[~suppressed]
        final_conf = confidences[~suppressed]
        
        # Apply both thresholds
        k_threshold = self.config.saddle_k_min_threshold
        confidence_threshold = self.config.saddle_confidence_threshold
        self._log(f"Saddle parameters: k_threshold={k_threshold:.2e}, confidence_threshold={confidence_threshold:.2f}")
        
        if len(final_k) > 0:
            self._log(f"Saddle statistics: K min={np.min(final_k):.2e}, K max={np.max(final_k):.2e}, "
                      f"K threshold={k_threshold:.2e}, confidence threshold={confidence_threshold:.2f}")
        
        validated_count = 0
        for cx, cy, k_mag, conf in zip(final_xs, final_ys, final_k, final_conf):
            # Apply both filters: must meet OR exceed thresholds
            if k_mag < k_threshold:
                continue
            if conf < confidence_threshold:
                continue
            elevation = float(z[cy, cx])
            
            saddle = SaddleFeature(
                centroid=(int(cx), int(cy)),
                elevation_range=(elevation, elevation),
                saddle_elevation_m=elevation,
                #REMOVED: connecting_ridges=set(),
                #REMOVED: connecting_valleys=set(),
                k_curvature=float(gaussian_curvature[cy, cx]),
                feature_type=FeatureType.SADDLE,
                confidence=float(conf),
                metadata={
                    'mean_curvature': float(mean_curvature[cy, cx]),
                    'k_magnitude': float(k_mag),
                    'confidence': float(conf),
                    'avg_slope': float(slope[cy, cx]) if slope is not None else 10.0,
                }
            )
            saddles.append(saddle)
            validated_count += 1
        
        return saddles

    def _extract_flat_zones(self, heightmap: Heightmap, curvature_type: np.ndarray,
                            slope: Optional[np.ndarray]) -> List[FlatZoneFeature]:
        """Extract flat zones for traversability."""
        z = heightmap.data
        cell_size = heightmap.config.horizontal_scale
        flat_zones = []

        if slope is not None:
            flat_mask = (slope < self.config.flat_zone_slope_threshold_deg)
        else:
            flat_mask = self._is_curvature(curvature_type, CurvatureType.FLAT)

        elev_floor = heightmap.config.sea_level_offset
        flat_mask = flat_mask & (z >= elev_floor)
        
        flat_pixels = np.sum(flat_mask)
        self._log(f"Flat zone mask pixels: {flat_pixels} ({100*flat_pixels/flat_mask.size:.1f}%)")

        labeled, num_features = label(flat_mask)
        self._log(f"Flat zone candidates: {num_features} (min_size={self.config.min_flat_zone_size_px}px)")
        
        validated_count = 0
        for i in range(1, num_features + 1):
            mask = (labeled == i)
            size = np.sum(mask)
            if size < self.config.min_flat_zone_size_px:
                continue

            ys, xs = np.where(mask)
            centroid_px = (int(np.mean(xs)), int(np.mean(ys)))
            elevations = z[ys, xs]
            elevation_range = (np.min(elevations), np.max(elevations))

            x_min, x_max = np.min(xs), np.max(xs)
            y_min, y_max = np.min(ys), np.max(ys)

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
                feature_type=FeatureType.FLAT,
                confidence=1.0,  # TODO: GAUGE CONFIDENCE
                metadata={
                    'size': size,
                    'area_m2': size * (cell_size ** 2),
                    'bounds': (x_min, x_max, y_min, y_max),
                    'avg_slope': avg_slope
                }
            )
            flat_zones.append(flat_zone)
            validated_count += 1
        
        self._log(f"Flat zones validated: {validated_count}")
        return flat_zones

    def _estimate_drainage_area(self, heightmap: Heightmap, spine_points: List[PixelCoord],
                                 aspect: Optional[np.ndarray]) -> Optional[float]:
        """Estimate drainage area for valley (simplified)."""
        if aspect is None or not spine_points:
            return None
        length = len(spine_points)
        cell_area = heightmap.config.horizontal_scale ** 2
        return length * cell_area * 10

    def _calculate_prominence(self, heightmap: Heightmap, peak_px: PixelCoord) -> float:
        """Calculate prominence of a peak."""
        z = heightmap.data
        x, y = peak_px
        peak_elev = z[y, x]

        radius = int(self.config.prominence_search_radius_m / heightmap.config.horizontal_scale)
        y_min = max(0, y - radius)
        y_max = min(z.shape[0], y + radius)
        x_min = max(0, x - radius)
        x_max = min(z.shape[1], x + radius)

        local_min = np.min(z[y_min:y_max, x_min:x_max])
        prominence = peak_elev - local_min

        return float(prominence)

    def _order_spine_points(self, xs: np.ndarray, ys: np.ndarray) -> List[PixelCoord]:
        """Order ridge/valley skeleton points as a continuous polyline."""
        if len(xs) < 2:
            return [(int(xs[0]), int(ys[0]))]

        points = np.column_stack([xs, ys]).astype(np.float32)
        n = len(points)

        # use KDTree for all neighbour queries
        from scipy.spatial import KDTree
        tree = KDTree(points)

        # Find start point: prefer a skeleton endpoint (≤1 neighbour within 2px).
        # query k=3 gives [self, nearest, 2nd nearest] — neighbour count = hits within r=2 minus self.
        k_query = min(4, n)
        dists, _ = tree.query(points, k=k_query)
        # dists[:,0] is always 0.0 (self); count columns 1..k that are within 2.0
        neighbour_counts = np.sum(dists[:, 1:] <= 2.0, axis=1)

        min_neighbours = neighbour_counts.min()
        if min_neighbours <= 2:
            # True endpoint exists — pick the one with fewest neighbours
            candidates = np.where(neighbour_counts == min_neighbours)[0]
        else:
            # No clean endpoint (closed loop / dense cluster) — fall back to
            # the point furthest from the centroid
            cx, cy = np.mean(xs), np.mean(ys)
            dist_from_centroid = (xs - cx) ** 2 + (ys - cy) ** 2
            candidates = np.array([np.argmax(dist_from_centroid)])

        start_idx = int(candidates[0])

        # Greedy nearest-neighbour walk using KDTree
        visited = np.zeros(n, dtype=bool)
        ordered_idx = [start_idx]
        visited[start_idx] = True

        for _ in range(n - 1):
            current = points[ordered_idx[-1]]
            # Query enough neighbours to find the closest unvisited point.
            # k=min(n,8) covers dense skeletons; fall back to full search if needed.
            k_walk = min(n, 8)
            _, idxs = tree.query(current, k=k_walk)
            idxs = np.atleast_1d(idxs)
            chosen = None
            for idx in idxs:
                if not visited[idx]:
                    chosen = idx
                    break
            if chosen is None:
                # All close neighbours visited — find globally nearest unvisited
                unvisited = np.where(~visited)[0]
                dists_uv = np.sum((points[unvisited] - current) ** 2, axis=1)
                chosen = unvisited[np.argmin(dists_uv)]
            ordered_idx.append(chosen)
            visited[chosen] = True

        return [(int(xs[i]), int(ys[i])) for i in ordered_idx]

    def _build_feature_hierarchy(self, features: List[TerrainFeature]) -> None:
        """Build hierarchical relationships between features."""
        from scipy.spatial import KDTree

        self._log('Building feature hierarchy...')
        peaks   = [f for f in features if isinstance(f, PeakFeature)]
        ridges  = [f for f in features if isinstance(f, RidgeFeature)]
        valleys = [f for f in features if isinstance(f, ValleyFeature)]
        saddles = [f for f in features if isinstance(f, SaddleFeature)]

        # build spatial indices once over all spine points

        # Peak centroid index
        if peaks:
            peak_coords  = np.array([p.centroid for p in peaks], dtype=np.float32)
            peak_tree    = KDTree(peak_coords)
        else:
            peak_tree = None

        # Ridge spine-point index: each entry maps back to its ridge feature_id
        ridge_spine_coords = []
        ridge_spine_ids    = []
        for ridge in ridges:
            for pt in ridge.spine_points:
                ridge_spine_coords.append(pt)
                ridge_spine_ids.append(ridge.feature_id)

        if ridge_spine_coords:
            ridge_tree = KDTree(np.array(ridge_spine_coords, dtype=np.float32))
        else:
            ridge_tree = None

        # Valley spine-point index
        valley_spine_coords = []
        valley_spine_ids    = []
        for valley in valleys:
            for pt in valley.spine_points:
                valley_spine_coords.append(pt)
                valley_spine_ids.append(valley.feature_id)

        if valley_spine_coords:
            valley_tree = KDTree(np.array(valley_spine_coords, dtype=np.float32))
        else:
            valley_tree = None

        # Connect ridges to peaks
        tolerance = self.config.feature_connection_tolerance_px
        for ridge in ridges:
            ridge.connected_peaks = self._find_connected_peaks(ridge.spine_points, peak_tree, peaks, tolerance)

        # Connect saddles to ridges and valleys
        for saddle in saddles:
            conn = self._find_saddle_connections(
                saddle.centroid,
                ridge_tree, ridge_spine_ids,
                valley_tree, valley_spine_ids,
                tolerance
            )
            saddle.connecting_ridges   = conn['ridges']
            saddle.connecting_valleys  = conn['valleys']


    def _find_connected_peaks(self, spine_points: List[PixelCoord],
                               peak_tree, peaks: List[PeakFeature],
                               tolerance: float) -> Set[str]:
        """Find peak IDs connected to this ridge via endpoint proximity."""
        connected = set()

        if len(spine_points) < 2 or peak_tree is None:
            return connected

        endpoints = np.array([spine_points[0], spine_points[-1]], dtype=np.float32)
        # query_ball_point returns all peaks within the tolerance radius
        matches = peak_tree.query_ball_point(endpoints, r=tolerance)
        for hit_list in matches:
            for idx in hit_list:
                connected.add(peaks[idx].feature_id)

        if connected:
            self._log(f'found {len(connected)} peaks connected to ridge endpoint, tolerance={tolerance}')

        return connected


    def _find_saddle_connections(self, saddle_px: PixelCoord,
                                  ridge_tree, ridge_spine_ids: List[str],
                                  valley_tree, valley_spine_ids: List[str],
                                  tolerance: float) -> Dict:
        """Find ridges and valleys connected to saddle via spatial index."""
        connections = {'ridges': set(), 'valleys': set()}
        pt = np.array([saddle_px], dtype=np.float32)

        if ridge_tree is not None:
            for idx in ridge_tree.query_ball_point(pt[0], r=tolerance):
                connections['ridges'].add(ridge_spine_ids[idx])

        if valley_tree is not None:
            for idx in valley_tree.query_ball_point(pt[0], r=tolerance):
                connections['valleys'].add(valley_spine_ids[idx])

        #total = len(connections['ridges']) + len(connections['valleys'])
        #if total > 0:
        #    self._log(f'found {total} connections at PixelCoord{saddle_px}, tolerance={tolerance}')

        return connections
    
    @property
    def output_schema(self) -> dict:
        return {
            "type": "List[TerrainFeature]",
            "feature_types": ["PeakFeature", "RidgeFeature", "ValleyFeature", "SaddleFeature", "FlatZoneFeature"],
            "geometry": "points and polylines in PixelCoord space"
        }   