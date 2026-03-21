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
        self._min_feature_size = config.min_peak_size_px
        self._min_ridge_length = config.min_ridge_length_px
        self._ridge_width = config.min_ridge_length_px // 3
        self._saddle_confidence_threshold = config.saddle_confidence_threshold
        self._saddle_k_min_threshold = config.saddle_k_min_threshold
        self._border_margin = config.border_margin_px
        self._annular_inner_px = max(3, int(5 / config.horizontal_scale))
        self._annular_outer_px = max(7, int(12 / config.horizontal_scale))
        
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
    
    def _extract_peaks(self, heightmap, curvature_type, gaussian_curvature,
                   k_confidence, mean_curvature, slope) -> List[PeakFeature]:
        """
        Extract peaks using regional maxima + annular curvature validation.
        
        Strategy:
        1. Pre-smooth to reduce noise on gentle terrain
        2. Find regional maxima (connected plateau components)
        3. Validate each candidate by checking curvature in surrounding annulus
        """
        from skimage import morphology
        from scipy import ndimage
        
        z = heightmap.data
        cell_size = heightmap.config.horizontal_scale
        peaks = []
        
        # 1. PRE-SMOOTHING (Critical for gentle terrain with 44m range)
        # Sigma 1.5 balances noise reduction vs feature preservation
        z_smooth = ndimage.gaussian_filter(z, sigma=1.5)
        
        # 2. CANDIDATE DETECTION: Regional Maxima (not pixel-wise)
        # This collapses a plateau into a single candidate
        regional_max_mask = morphology.local_maxima(z_smooth, connectivity=2)
        labeled_peaks, num_candidates = ndimage.label(regional_max_mask)
        
        print(f"_extract_peaks(): regional maxima candidates={num_candidates}")
        
        if num_candidates == 0:
            print("No regional maxima found, using elevation fallback...")
            return self._extract_peaks_fallback(heightmap, slope)
        
        # 3. CURVATURE CALCULATION (for validation)
        # Use smoothed data for stable derivatives
        zy, zx = np.gradient(z_smooth, cell_size)
        zxx, zxy = np.gradient(zx, cell_size)
        zyx, zyy = np.gradient(zy, cell_size)
        H = (zxx + zyy) / 2.0  # Mean curvature
        
        # 4. ANNULAR VALIDATION
        # Create ring mask (annulus) for sampling curvature around peaks
        ring_radius_inner = int(5 / cell_size)  # ~5m inner radius
        ring_radius_outer = int(12 / cell_size)  # ~12m outer radius
        ring_radius_inner = max(3, min(ring_radius_inner, 10))
        ring_radius_outer = max(7, min(ring_radius_outer, 20))
        
        # Pre-compute ring coordinates relative to center
        y_grid, x_grid = np.ogrid[:z.shape[0], :z.shape[1]]
        center = np.array([z.shape[0] // 2, z.shape[1] // 2])
        dist_from_center = np.sqrt((x_grid - center[1])**2 + (y_grid - center[0])**2)
        ring_mask = (dist_from_center >= ring_radius_inner) & (dist_from_center <= ring_radius_outer)
        ring_coords = np.argwhere(ring_mask)
        
        # Threshold for convex curvature (based on H_std from config)
        h_threshold = self.config.curvature_epsilon_h_min * 10  # ~1e-4 for gentle terrain
        
        for i in range(1, num_candidates + 1):
            # Get pixels belonging to this peak region
            peak_pixels = np.argwhere(labeled_peaks == i)
            if len(peak_pixels) < self._min_feature_size:
                continue
            
            # Centroid of the plateau
            centroid_y, centroid_x = peak_pixels.mean(axis=0).astype(int)
            
            # Extract curvature values in the ring around the centroid
            ring_y = centroid_y + (ring_coords[:, 0] - center[0])
            ring_x = centroid_x + (ring_coords[:, 1] - center[1])
            
            # Boundary check
            valid = (ring_y >= 0) & (ring_y < z.shape[0]) & (ring_x >= 0) & (ring_x < z.shape[1])
            if np.sum(valid) < 10:  # Skip edge peaks
                continue
            
            sample_h = H[ring_y[valid], ring_x[valid]]
            mean_h = np.mean(sample_h)
            
            # Criterion: Shoulder must be convex (H > threshold)
            if mean_h > h_threshold:
                # Calculate prominence and create peak
                prominence = self._calculate_prominence(heightmap, (centroid_x, centroid_y))
                if prominence > 1.0:
                    # Create mask for this peak (the regional max region)
                    mask = (labeled_peaks == i)
                    peaks.append(self._create_peak_feature(
                        heightmap, mask, (centroid_x, centroid_y), prominence,
                        mean_curvature, gaussian_curvature, slope,
                        method="curvature_annular"
                    ))
        
        # 5. FALLBACK: If no peaks found, use elevation-based detection
        if len(peaks) == 0:
            print("No curvature-validated peaks detected, using elevation fallback...")
            return self._extract_peaks_fallback(heightmap, slope)
        
        return peaks

    def _extract_peaks_fallback(self, heightmap, slope) -> List[PeakFeature]:
        """Elevation-based fallback for maps where curvature fails."""
        z = heightmap.data
        peaks = []
        
        # Use larger window for game maps (7x7 instead of 3x3)
        local_max_fallback = (z == ndimage.maximum_filter(z, size=7))
        
        # Exclude border pixels
        border_mask = np.zeros_like(z, dtype=bool)
        border_mask[self._border_margin:-self._border_margin, 
                    self._border_margin:-self._border_margin] = True
        local_max_fallback = local_max_fallback & border_mask
        
        labeled, num_features = ndimage.label(local_max_fallback)
        
        for i in range(1, num_features + 1):
            mask = (labeled == i)
            if np.sum(mask) < self._min_feature_size * 2:
                continue
                
            ys, xs = np.where(mask)
            centroid_px = (int(np.mean(xs)), int(np.mean(ys)))
            prominence = self._calculate_prominence(heightmap, centroid_px)
            
            # Prominence threshold for game significance (3m = meaningful high ground)
            if prominence > 3.0:
                peaks.append(self._create_peak_feature(
                    heightmap, mask, centroid_px, prominence,
                    None, None, slope,
                    method="elevation_fallback"
                ))
        
        return peaks

    def _create_peak_feature(self, heightmap, mask, centroid_px, prominence,
                             mean_curvature, gaussian_curvature, slope, method="curvature"):
        """Helper to create a PeakFeature with consistent metadata."""
        z = heightmap.data
        x, y = centroid_px
        elevation = float(z[y, x])
        
        # Calculate defensive rating based on prominence and slope
        # For game maps: high prominence + moderate slope = good defensive position
        avg_slope = np.mean(slope[mask]) if slope is not None else 10.0
        defensive_rating = min(1.0, (prominence / 15.0) * (1 - abs(avg_slope - 15) / 30))
        
        peak = PeakFeature(
            centroid=centroid_px,
            elevation_range=(float(np.min(z[mask])), elevation),
            prominence=prominence,
            #avg_slope=float(avg_slope),  see __init__ signature flag
            metadata={
                'size': int(np.sum(mask)),
                'elevation': elevation,
                'confidence': float(prominence / 20.0),  # Confidence based on prominence
                'defensive_rating': defensive_rating,
                'detection_method': method,
                'mean_curvature': float(mean_curvature[y, x]) if mean_curvature is not None else 0,
                'gaussian_curvature': float(gaussian_curvature[y, x]) if gaussian_curvature is not None else 0
            }
        )
        peak._avg_slope = float(avg_slope) # store slope
        return peak
    
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
                #_avg_slope=float(avg_slope),
                metadata={
                    'length': len(spine_points),
                    'width_meters': float(width_meters),
                    'confidence': float(avg_confidence),
                    'avg_curvature': float(avg_curvature)
                }
            )
            ridge._avg_slope = float(avg_slope) # store slope
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
                #_avg_slope=float(avg_slope),
                metadata={
                    'length': len(spine_points),
                    'confidence': float(avg_confidence),
                    'avg_slope': float(avg_slope),
                    'avg_curvature': float(np.mean(mean_curvature[ys, xs]))
                }
            )
            valley._avg_slope = float(avg_slope) # store slope
            valleys.append(valley)
        
        return valleys


    def _extract_saddles(self, heightmap: Heightmap, curvature_type: np.ndarray,
                         gaussian_curvature: np.ndarray, k_confidence: np.ndarray,
                         mean_curvature: np.ndarray, slope: Optional[np.ndarray]) -> List[SaddleFeature]:
        """
        Extract saddle points as topographic passes between peaks.

        A saddle is a local elevation minimum that sits between higher terrain on
        at least two sides. The key filters that prevent coastline false-positives:
        1. Elevation gate: must be above the map's meaningful terrain floor
           (excludes sea pixels whose elevation ≈ 0)
        2. Higher-terrain check: at least two opposing quadrants around the candidate
           must contain pixels higher than the candidate (confirms pass geometry)
        3. NMS: suppress duplicates within peak_nms_radius_px
        """
        from scipy.ndimage import minimum_filter, maximum_filter
        from scipy.spatial import KDTree

        z = heightmap.data
        h, w = z.shape
        m = self._border_margin
        cell_size = heightmap.config.horizontal_scale
        saddles = []

        # Smooth for stable local minima
        z_smooth = ndimage.gaussian_filter(z.astype(np.float32), sigma=2.0)

        # Local minima in a terrain-scale window
        window = max(5, int(10 / cell_size))
        local_min_mask = (z_smooth == minimum_filter(z_smooth, size=window))
        local_max_mask = (z_smooth == maximum_filter(z_smooth, size=window))
        local_min_mask = local_min_mask & ~local_max_mask

        # Strip border
        border = np.zeros_like(local_min_mask)
        border[m:h-m, m:w-m] = True
        local_min_mask = local_min_mask & border

        # ── KEY GATE 1: elevation floor ──────────────────────────────────────
        # Compute a meaningful terrain floor: use the Nth percentile of non-zero
        # elevations. Sea pixels are 0 (or near-0); real saddles are above the coast.
        nonzero_elevs = z[z > 0]
        if len(nonzero_elevs) > 0:
            elev_floor = float(np.percentile(nonzero_elevs, 10))  # bottom 10% of land
        else:
            elev_floor = 0.0
        elev_floor = max(elev_floor, heightmap.config.sea_level_offset + 0.5)
        local_min_mask = local_min_mask & (z_smooth > elev_floor)

        candidate_ys, candidate_xs = np.where(local_min_mask)
        print(f"_extract_saddles(): candidates after elev_floor({elev_floor:.2f}m)={len(candidate_xs)}, window={window}px")

        if len(candidate_xs) == 0:
            return saddles

        # ── KEY GATE 2: higher terrain on opposing sides ─────────────────────
        # Sample 4 quadrant directions around each candidate (N/S and E/W).
        # A real saddle has higher terrain in at least 2 opposing directions.
        check_r = max(window, int(15 / cell_size))
        valid_candidates = []

        for cx, cy in zip(candidate_xs, candidate_ys):
            elev_c = float(z_smooth[cy, cx])
            # Sample mean elevation in each cardinal quadrant at check_r distance
            samples = {
                'N': z_smooth[max(0, cy - check_r): cy, cx],
                'S': z_smooth[cy + 1: min(h, cy + check_r + 1), cx],
                'E': z_smooth[cy, cx + 1: min(w, cx + check_r + 1)],
                'W': z_smooth[cy, max(0, cx - check_r): cx],
            }
            # Count directions where terrain rises above the candidate
            higher = sum(1 for arr in samples.values()
                         if len(arr) > 0 and float(np.max(arr)) > elev_c + 0.2)
            if higher >= 2:
                valid_candidates.append((cx, cy))

        print(f"_extract_saddles(): after higher-terrain check → {len(valid_candidates)}")

        if not valid_candidates:
            return saddles

        # ── NMS: keep lowest elevation within each radius ────────────────────
        vx = np.array([c[0] for c in valid_candidates])
        vy = np.array([c[1] for c in valid_candidates])
        coords = np.column_stack([vx, vy]).astype(np.float32)
        elevs = z_smooth[vy, vx]
        tree = KDTree(coords)

        suppressed = np.zeros(len(vx), dtype=bool)
        for idx in np.argsort(elevs):          # process lowest first
            if suppressed[idx]:
                continue
            for nb in tree.query_ball_point(coords[idx], r=self.config.peak_nms_radius_px):
                if nb != idx:
                    suppressed[nb] = True

        final_xs = vx[~suppressed]
        final_ys = vy[~suppressed]
        print(f"_extract_saddles(): after NMS (radius={self.config.peak_nms_radius_px}px) → {len(final_xs)} saddles")

        for cx, cy in zip(final_xs, final_ys):
            elevation = float(z[cy, cx])
            saddle = SaddleFeature(
                centroid=(int(cx), int(cy)),
                elevation_range=(elevation, elevation),
                elevation=elevation,
                connecting_ridges=set(),
                connecting_valleys=set(),
                k_curvature=float(gaussian_curvature[cy, cx]),
                metadata={
                    'mean_curvature': float(mean_curvature[cy, cx]),
                    'confidence': float(abs(gaussian_curvature[cy, cx])),
                    'avg_slope': float(slope[cy, cx]) if slope is not None else 10.0
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