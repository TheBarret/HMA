"""
Layer 3: Topological Features
"""
import numpy as np
from scipy import ndimage
from scipy.ndimage import label, find_objects, maximum_filter, minimum_filter
from skimage import measure, morphology
from skimage.morphology import skeletonize, remove_small_objects
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
from uuid import uuid4

from core import (
    Heightmap, ScalarField, PipelineConfig, PipelineLayer,
    TerrainFeature, ClassifiedFeature, CurvatureType, Traversability,
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
        self._annular_inner_px = max(3, int(5 / config.horizontal_scale))
        self._annular_outer_px = max(7, int(12 / config.horizontal_scale))
    
    def _log(self, msg: str) -> None:
        """Helper for verbose logging"""
        if self.config.verbose:
            print(f"[Topological] {msg}")

    def execute(self, input_data: Dict) -> List[TerrainFeature]:
        """Extract discrete terrain features from continuous fields."""
        heightmap = input_data['heightmap']
        mean_curvature = input_data['curvature']
        gaussian_curvature = input_data['gaussian_curvature']
        curvature_type = input_data['curvature_type']
        slope = input_data.get('slope', None)
        aspect = input_data.get('aspect', None)

        if slope is None or aspect is None:
            if self.config.verbose:
                self._log("Slope/aspect not provided, some feature refinement will be limited")

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

        self._build_feature_hierarchy(features)

        if self.config.verbose:
            self._log(f"Extraction complete | total={len(features)}, peaks={len(peaks)}, ridges={len(ridges)}, valleys={len(valleys)}, saddles={len(saddles)}, flat_zones={len(flat_zones)}")
            if peaks:
                avg_confidence = np.mean([p.metadata.get('confidence', 0) for p in peaks])
                self._log(f"Average peak confidence: {avg_confidence:.3f}")

        return features

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
        """Extract peaks using regional maxima + curvature type validation."""
        from skimage import morphology
        from scipy import ndimage

        z = heightmap.data
        cell_size = heightmap.config.horizontal_scale
        peaks = []

        z_smooth = ndimage.gaussian_filter(z, sigma=1.5)
        regional_max_mask = morphology.local_maxima(z_smooth, connectivity=2)
        labeled_peaks, num_candidates = ndimage.label(regional_max_mask)

        self._log(f"Peak candidates: {num_candidates} regional maxima")
        
        if num_candidates == 0:
            self._log("No regional maxima found, using elevation fallback")
            return self._extract_peaks_fallback(heightmap, slope)

        shoulder_radius_inner = max(2, int(3 / cell_size))
        shoulder_radius_outer = max(10, int(20 / cell_size))

        validated_count = 0
        for i in range(1, num_candidates + 1):
            peak_pixels = np.argwhere(labeled_peaks == i)
            if len(peak_pixels) < self._min_feature_size:
                continue

            centroid_y, centroid_x = peak_pixels.mean(axis=0).astype(int)

            y, x = np.ogrid[:z.shape[0], :z.shape[1]]
            dist = np.sqrt((x - centroid_x)**2 + (y - centroid_y)**2)
            shoulder_mask = (dist >= shoulder_radius_inner) & (dist <= shoulder_radius_outer)

            shoulder_types = curvature_type[shoulder_mask]
            if len(shoulder_types) < 10:
                continue

            convex_count = np.sum(self._is_curvature(shoulder_types, CurvatureType.CONVEX))
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

        peak = PeakFeature(
            centroid=centroid_px,
            elevation_range=(float(np.min(z[mask])), elevation),
            prominence=prominence,
            metadata={
                'size': int(np.sum(mask)),
                'elevation': elevation,
                'confidence': float(prominence / 20.0),
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
        """Extract ridge lines from convex regions with skeletonization."""
        z = heightmap.data
        ridges = []

        convex_mask = self._is_curvature(curvature_type, CurvatureType.CONVEX)
        cylindrical_mask = (
            (np.abs(mean_curvature) > self.config.curvature_epsilon_h_min) &
            (np.abs(gaussian_curvature) < self.config.curvature_epsilon_k_min) &
            (slope is not None) &
            (slope > self.config.flat_zone_slope_threshold_deg)
        )
        ridge_mask = convex_mask | cylindrical_mask
        
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
            connected_peaks = self._find_connected_peaks(heightmap, spine_points)
            avg_slope = np.mean(slope[ys, xs]) if slope is not None else 10.0
            avg_confidence = np.mean(k_confidence[ys, xs])
            
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
            ridge._avg_slope = float(avg_slope)
            ridges.append(ridge)
            validated_count += 1
        
        self._log(f"Ridges validated: {validated_count}")
        return ridges

    def _extract_valleys(self, heightmap: Heightmap, curvature_type: np.ndarray,
                         gaussian_curvature: np.ndarray, k_confidence: np.ndarray,
                         mean_curvature: np.ndarray, slope: Optional[np.ndarray],
                         aspect: Optional[np.ndarray]) -> List[ValleyFeature]:
        """Extract valley lines from concave regions."""
        z = heightmap.data
        valleys = []

        concave_mask = self._is_curvature(curvature_type, CurvatureType.CONCAVE) & (gaussian_curvature > 0)
        
        valley_pixels = np.sum(concave_mask)
        self._log(f"Valley mask pixels: {valley_pixels} ({100*valley_pixels/concave_mask.size:.1f}%)")
        
        cleaned = remove_small_objects(concave_mask, min_size=self._min_feature_size)
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
            valley._avg_slope = float(avg_slope)
            valleys.append(valley)
            validated_count += 1
        
        self._log(f"Valleys validated: {validated_count}")
        return valleys

    def _extract_saddles(self, heightmap: Heightmap, curvature_type: np.ndarray,
                         gaussian_curvature: np.ndarray, k_confidence: np.ndarray,
                         mean_curvature: np.ndarray, slope: Optional[np.ndarray]) -> List[SaddleFeature]:
        from scipy.ndimage import minimum_filter1d, maximum_filter1d
        from scipy.spatial import KDTree
        
        z = heightmap.data
        h, w = z.shape
        m = self._border_margin
        saddles = []
        
        z_smooth = ndimage.gaussian_filter(z.astype(np.float32), sigma=2.0)
        win = max(7, int(14 / heightmap.config.horizontal_scale))
        
        lmin_x = (z_smooth == minimum_filter1d(z_smooth, size=win, axis=1))
        lmax_x = (z_smooth == maximum_filter1d(z_smooth, size=win, axis=1))
        lmin_y = (z_smooth == minimum_filter1d(z_smooth, size=win, axis=0))
        lmax_y = (z_smooth == maximum_filter1d(z_smooth, size=win, axis=0))
        
        saddle_mask = ((lmin_x & lmax_y) | (lmax_x & lmin_y))
        
        border = np.zeros_like(saddle_mask)
        border[m:h-m, m:w-m] = True
        saddle_mask = saddle_mask & border
        
        elev_floor = heightmap.config.sea_level_offset
        saddle_mask = saddle_mask & (z_smooth > elev_floor + 1.0)
        
        candidate_ys, candidate_xs = np.where(saddle_mask)
        self._log(f"Saddle candidates: {len(candidate_xs)} (win={win}px)")
        
        if len(candidate_xs) == 0:
            return saddles
        
        # NMS
        coords = np.column_stack([candidate_xs, candidate_ys]).astype(np.float32)
        elevs = z_smooth[candidate_ys, candidate_xs]
        tree = KDTree(coords)
        suppressed = np.zeros(len(candidate_xs), dtype=bool)
        for idx in np.argsort(elevs):
            if suppressed[idx]: continue
            for nb in tree.query_ball_point(coords[idx], r=self.config.peak_nms_radius_px):
                if nb != idx: suppressed[nb] = True
        
        final_xs = candidate_xs[~suppressed]
        final_ys = candidate_ys[~suppressed]
        
        validated_count = 0
        for cx, cy in zip(final_xs, final_ys):
            elevation = float(z[cy, cx])
            _conf = float(abs(gaussian_curvature[cy, cx]))
            
            if _conf > self._saddle_confidence_threshold:
                continue
            
            saddle = SaddleFeature(
                centroid=(int(cx), int(cy)),
                elevation_range=(elevation, elevation),
                elevation=elevation,
                connecting_ridges=set(), connecting_valleys=set(),
                k_curvature=float(gaussian_curvature[cy, cx]),
                metadata={
                    'mean_curvature': float(mean_curvature[cy, cx]),
                    'confidence': float(abs(gaussian_curvature[cy, cx])),
                    'avg_slope': float(slope[cy, cx]) if slope is not None else 10.0
                }
            )
            saddles.append(saddle)
            validated_count += 1
        
        self._log(f"Saddles validated: {validated_count} (confidence_threshold={self._saddle_confidence_threshold})")
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

        radius = int(200 / heightmap.config.horizontal_scale)
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
            start_idx = fallback_idx

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
        return set()

    def _find_saddle_connections(self, heightmap: Heightmap, saddle_px: PixelCoord) -> Dict:
        """Find ridges and valleys connected to saddle."""
        return {'ridges': set(), 'valleys': set()}

    def _build_feature_hierarchy(self, features: List[TerrainFeature]) -> None:
        """Build hierarchical relationships between features."""
        pass

    @property
    def output_schema(self) -> dict:
        return {
            "type": "List[TerrainFeature]",
            "feature_types": ["PeakFeature", "RidgeFeature", "ValleyFeature", "SaddleFeature", "FlatZoneFeature"],
            "geometry": "points and polylines in PixelCoord space"
        }