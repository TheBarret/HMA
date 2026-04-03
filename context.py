import numpy as np
from typing import Dict, Any, Optional, Callable
from typing import List, Tuple, Union, Set
from dataclasses import dataclass, field

from core import ( Heightmap, ScalarField, ClassifiedFeature, PeakFeature, RidgeFeature, ValleyFeature, SaddleFeature, FlatZoneFeature, Traversability )

#
#  Spatial Query Tables
#  - Quantifiers
#  - Descriptions
#  
FEATURE_QUANTIFIERS: Dict[str, List[tuple]] = {
    "prominence" : [(0, 20, "small"), (20, 50, "moderate"), (50, 100, "prominent"), (100, 9999, "major")],
    "slope"      : [(0, 10, "gentle"), (10, 25, "moderate"), (25, 40, "steep"), (40, 90, "very steep")],
    "area_m2"    : [(0, 500, "small"), (500, 2000, "medium-sized"), (2000, 10000, "large"), (10000, 999999, "extensive")],
    "length_m"   : [(0, 100, "short"), (100, 300, "moderate-length"), (300, 800, "long"), (800, 9999, "extended")],
    "elevation_m": [(0, 100, "low"), (100, 300, "moderate"), (300, 800, "high"), (800, 9999, "very high")],
    "depth_m"    : [(0, 20, "shallow"), (20, 50, "moderate-depth"), (50, 100, "deep"), (100, 9999, "very deep")],
}
 
FEATURE_DESC: Dict[str, str] = {
    "peak"           : "{prominence_quant} peak rising {prominence_value} above {reference} with {slope_quant} slopes",
    "peak_short"     : "{prominence_quant} peak",
    "valley"         : "{length_quant} valley with {slope_quant} floor slopes{stream_suffix}",
    "valley_short"   : "{length_quant} valley",
    "ridge"          : "{length_quant} ridge with {slope_quant} slopes{peak_connection_suffix}",
    "ridge_short"    : "{length_quant} ridge",
    "saddle"         : "{elevation_quant} saddle at {elevation_value} elevation{connection_suffix}",
    "saddle_short"   : "{elevation_quant} saddle",
    "flat_zone"      : "{area_quant} flat area spanning {area_value} with {slope_quant} slopes",
    "flat_zone_short": "{area_quant} flat area",
}
 
FEATURE_SUFFIX: Dict[str, str] = {
    "stream_single"          : ", part of a drainage network",
    "stream_multi"           : ", part of a {order}{ordinal_suffix}-order stream",
    "peak_connection_single" : " connecting two high points",
    "peak_connection_multi"  : " connecting {count} high points",
    "saddle_connection_single": " providing passage between surrounding areas",
    "saddle_connection_multi" : " connecting {count} surrounding high points",
    "reference_ground"        : "surrounding terrain",
    "reference_sea"           : "sea level",
}
 
# filter operators and their callables
_OPERATORS: Dict[str, Callable] = {
    "eq"      : lambda a, b: a == b,
    "neq"     : lambda a, b: a != b,
    "gt"      : lambda a, b: a > b,
    "gte"     : lambda a, b: a >= b,
    "lt"      : lambda a, b: a < b,
    "lte"     : lambda a, b: a <= b,
    "in"      : lambda a, b: a in b,
    "contains": lambda a, b: b in a,
}
 
# Type to feature list
_TYPE_MAP: Dict[type, str] = {
    PeakFeature    : "peaks",
    RidgeFeature   : "ridges",
    ValleyFeature  : "valleys",
    SaddleFeature  : "saddles",
    FlatZoneFeature: "flat_zones",
}

#
#  Analyzed Terrain
#  

@dataclass
class AnalyzedTerrain:
    """
    Aggregated output of the full analysis pipeline.
    Entry point for all spatial queries via .query().
    """
 
    source_heightmap: Heightmap
 
    # Feature lists — typed at the list level, concrete subclasses at runtime
    peaks      : List[ClassifiedFeature] = field(default_factory=list)
    valleys    : List[ClassifiedFeature] = field(default_factory=list)
    ridges     : List[ClassifiedFeature] = field(default_factory=list)
    saddles    : List[ClassifiedFeature] = field(default_factory=list)
    flat_zones : List[ClassifiedFeature] = field(default_factory=list)
 
    # Relational graphs — all keyed by feature_id
    visibility_graph   : Dict[str, Set[str]]   = field(default_factory=dict)
    flow_network       : Dict[str, List[str]]   = field(default_factory=dict)
    connectivity_graph : Dict[str, Set[str]]    = field(default_factory=dict)
    watersheds         : Dict[str, Set[str]]    = field(default_factory=dict)
    flow_accumulation  : Optional[ScalarField]  = None
 
    def __post_init__(self):
        self._feature_by_id  : Dict[str, ClassifiedFeature] = {}
        self._feature_position: Dict[str, PixelCoord]        = {}
        self._rebuild_index()
 
    def _rebuild_index(self):
        """Rebuild lookup tables from feature lists. Call after any mutation."""
        self._feature_by_id.clear()
        self._feature_position.clear()
        for feature in self._all_features():
            self._feature_by_id[feature.feature_id]   = feature
            self._feature_position[feature.feature_id] = feature.centroid
 
    def _all_features(self) -> List[ClassifiedFeature]:
        return self.peaks + self.ridges + self.valleys + self.saddles + self.flat_zones
 
    # -------------------------------------------------------------------------
    #  Internal lookups
    # -------------------------------------------------------------------------
 
    def _get_feature(self, feature_id: str) -> ClassifiedFeature:
        """Look up a single feature by ID. Raises KeyError if not found."""
        try:
            return self._feature_by_id[feature_id]
        except KeyError:
            raise KeyError(f"Feature '{feature_id}' not found in terrain")
 
    def _features_by_type(self, feature_type: type) -> List[ClassifiedFeature]:
        """Return a copy of the feature list for a given concrete type."""
        attr = _TYPE_MAP.get(feature_type)
        if attr is None:
            raise ValueError(
                f"Unknown feature type '{feature_type.__name__}'. "
                f"Expected one of: {[t.__name__ for t in _TYPE_MAP]}"
            )
        return list(getattr(self, attr))  # copy — queries must not mutate source
 
    # -------------------------------------------------------------------------
    #  Direct spatial lookups (not part of the query chain)
    # -------------------------------------------------------------------------
 
    def ping_at(self, x: float, y: float, radius_m: float,
                feature_type: Optional[type] = None) -> List[ClassifiedFeature]:
        """
        Find features within radius of a pixel coordinate.
 
        Args:
            x, y        : Pixel coordinates (image space)
            radius_m    : Search radius in metres
            feature_type: Optional type filter (e.g. PeakFeature). None = all types.
 
        Returns:
            Features sorted by distance, closest first.
        """
        scale      = self.source_heightmap.config.horizontal_scale
        radius_px  = radius_m / scale
        radius_sq  = radius_px * radius_px
 
        candidates = (
            self._features_by_type(feature_type)
            if feature_type is not None
            else self._all_features()
        )
 
        hits: List[Tuple[ClassifiedFeature, float]] = []
        for feature in candidates:
            fx, fy = self._feature_position[feature.feature_id]
            dx, dy = fx - x, fy - y
            dist_sq = dx * dx + dy * dy
            if dist_sq <= radius_sq:
                hits.append((feature, dist_sq))
 
        hits.sort(key=lambda t: t[1])
        return [f for f, _ in hits]
 
    def ping_from(self, feature_id: str, radius_m: float,
                  feature_type: Optional[type] = None) -> List[ClassifiedFeature]:
        """
        Find features within radius of a known feature's centroid.
        The source feature itself is excluded from results.
 
        Args:
            feature_id  : Source feature ID
            radius_m    : Search radius in metres
            feature_type: Optional type filter. None = all types.
 
        Returns:
            Features sorted by distance, closest first.
        """
        source = self._get_feature(feature_id)
        x, y   = source.centroid
        results = self.ping_at(x, y, radius_m, feature_type)
        return [f for f in results if f.feature_id != feature_id]
 
    # -------------------------------------------------------------------------
    #  Human-readable descriptions
    # -------------------------------------------------------------------------
 
    def describe(self, feature_id: str, brief: bool = False) -> str:
        """
        Return a human-readable description of a feature.
 
        Args:
            feature_id: Target feature ID
            brief     : If True, use short template ("moderate peak" vs full sentence)
        """
        feature     = self._get_feature(feature_id)
        feature_key = self._feature_type_key(feature)
        template    = FEATURE_DESC.get(f"{feature_key}_short" if brief else feature_key,
                                       f"[unknown feature type: {feature_key}]")
 
        import re
        placeholders = re.findall(r'\{([^}]+)\}', template)
        result = template
        for ph in placeholders:
            result = result.replace(f"{{{ph}}}", self._resolve_placeholder(feature, ph))
        return result
 
    def _feature_type_key(self, feature: ClassifiedFeature) -> str:
        mapping = {
            PeakFeature    : "peak",
            ValleyFeature  : "valley",
            RidgeFeature   : "ridge",
            SaddleFeature  : "saddle",
            FlatZoneFeature: "flat_zone",
        }
        return mapping.get(type(feature), "unknown")
 
    def _resolve_placeholder(self, feature: ClassifiedFeature, placeholder: str) -> str:
        if placeholder.endswith("_quant"):
            metric = placeholder[:-6]
            return self._quantify(metric, self._metric_value(feature, metric))
        if placeholder.endswith("_value"):
            metric = placeholder[:-6]
            return self._format_value(self._metric_value(feature, metric))
        if placeholder == "stream_suffix":
            if not isinstance(feature, ValleyFeature):
                return ""
            order = getattr(feature, "stream_order", 0)
            if order <= 0:
                return ""
            if order == 1:
                return FEATURE_SUFFIX["stream_single"]
            ordinal = self._ordinal_suffix(order)
            return FEATURE_SUFFIX["stream_multi"].format(order=order, ordinal_suffix=ordinal)
        if placeholder == "peak_connection_suffix":
            if not isinstance(feature, RidgeFeature):
                return ""
            count = len(getattr(feature, "connected_peaks", []))
            if count == 0:
                return ""
            if count == 2:
                return FEATURE_SUFFIX["peak_connection_single"]
            return FEATURE_SUFFIX["peak_connection_multi"].format(count=count)
        if placeholder == "connection_suffix":
            if not isinstance(feature, SaddleFeature):
                return ""
            count = len(getattr(feature, "connected_peaks", []))
            if count == 0:
                return FEATURE_SUFFIX["saddle_connection_single"]
            return FEATURE_SUFFIX["saddle_connection_multi"].format(count=count)
        if placeholder == "reference":
            cx, cy = feature.centroid
            elev   = self.source_heightmap.elevation_at((int(cx), int(cy))) or 0.0
            sea    = self.source_heightmap.config.sea_level_offset
            return (FEATURE_SUFFIX["reference_sea"]
                    if elev < sea + 10.0
                    else FEATURE_SUFFIX["reference_ground"])
        return f"[unknown:{placeholder}]"
 
    def _metric_value(self, feature: ClassifiedFeature, metric: str) -> float:
        if metric == "prominence" and isinstance(feature, PeakFeature):
            return feature.prominence
        if metric == "slope":
            return feature.avg_slope
        if metric == "length" and hasattr(feature, "spine_points"):
            pts = feature.spine_points
            if len(pts) < 2:
                return 0.0
            scale = self.source_heightmap.config.horizontal_scale
            return sum(
                np.sqrt((pts[i+1][0]-pts[i][0])**2 + (pts[i+1][1]-pts[i][1])**2) * scale
                for i in range(len(pts) - 1)
            )
        if metric == "area" and isinstance(feature, FlatZoneFeature):
            return feature.area_pixels * (self.source_heightmap.config.horizontal_scale ** 2)
        if metric == "elevation":
            cx, cy = feature.centroid
            return self.source_heightmap.elevation_at((int(cx), int(cy))) or 0.0
        if metric == "depth" and isinstance(feature, ValleyFeature):
            return 20.0  # TODO: computed in Layer 3
        return 0.0
 
    def _quantify(self, metric: str, value: float) -> str:
        for low, high, label in FEATURE_QUANTIFIERS.get(metric, []):
            if low <= value < high:
                return label
        return "unknown"
 
    def _format_value(self, value: float) -> str:
        if value >= 1000:
            return f"{value / 1000:.1f}km"
        if value >= 1:
            return f"{value:.0f}m"
        return f"{value:.1f}m"
 
    def _ordinal_suffix(self, n: int) -> str:
        if 10 <= n % 100 <= 20:
            return "th"
        return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
 
    # -------------------------------------------------------------------------
    #  Query entry point
    # -------------------------------------------------------------------------
 
    def query(self) -> 'TerrainQuery':
        """
        Start a fluent query against this terrain.
 
        Usage:
            results = (terrain.query()
                       .select(PeakFeature)
                       .where(prominence__gt=50)
                       .in_watershed("basin_42")
                       .visible_to(other_query)
                       .order_by("prominence", descending=True)
                       .limit(5)
                       .execute())
        """
        return TerrainQuery(self)

# =============================================================================
#  TerrainQuery  —  immutable-style filter builder
#
#  Each method returns a NEW TerrainQuery with updated state.
#  This means a partially-built query can be safely branched:
#
#      base   = terrain.query().select(PeakFeature).where(prominence__gt=50)
#      large  = base.where(prominence__gt=200).execute()
#      nearby = base.in_watershed("basin_1").execute()
#
#  Both branches read from `base` without interfering.
# =============================================================================
 
class TerrainQuery:
 
    def __init__(self, terrain: 'AnalyzedTerrain'):
        self._terrain       : AnalyzedTerrain              = terrain
        self._feature_type  : Optional[type]               = None
        # Each filter is a parsed triple: (field_name, operator, value)
        self._filters       : List[Tuple[str, str, Any]]   = []
        # Each relationship is a (rel_type, rel_value) pair, applied in order
        self._relationships : List[Tuple[str, Any]]        = []
        self._order_field   : Optional[str]                = None
        self._order_desc    : bool                         = False
        self._limit_n       : Optional[int]                = None
        self._offset_n      : Optional[int]                = None
        self._executing     : bool                         = False  # cycle guard
 
    def _copy(self) -> 'TerrainQuery':
        """Shallow copy of query state — each builder method calls this."""
        q = TerrainQuery(self._terrain)
        q._feature_type  = self._feature_type
        q._filters       = list(self._filters)
        q._relationships = list(self._relationships)
        q._order_field   = self._order_field
        q._order_desc    = self._order_desc
        q._limit_n       = self._limit_n
        q._offset_n      = self._offset_n
        return q
 
    # -------------------------------------------------------------------------
    #  Builder methods — each returns a new TerrainQuery
    # -------------------------------------------------------------------------
 
    def select(self, feature_type: type) -> 'TerrainQuery':
        """Set the feature type to query. Must be called before execute()."""
        if feature_type not in _TYPE_MAP:
            raise ValueError(
                f"Unknown feature type '{feature_type.__name__}'. "
                f"Expected one of: {[t.__name__ for t in _TYPE_MAP]}"
            )
        q = self._copy()
        q._feature_type = feature_type
        return q
 
    def where(self, **kwargs) -> 'TerrainQuery':
        """
        Add one or more attribute filters using Django-style dunder syntax.
 
            .where(prominence__gt=100)
            .where(slope__lte=30, confidence__gte=0.8)
 
        Supported operators: eq, neq, gt, gte, lt, lte, in, contains
        Filters are AND-ed. Multiple .where() calls accumulate safely.
        None-valued attributes never match any filter.
        """
        q = self._copy()
        for key, value in kwargs.items():
            parts      = key.split("__", 1)
            field_name = parts[0]
            op         = parts[1] if len(parts) > 1 else "eq"
            if op not in _OPERATORS:
                raise ValueError(
                    f"Unknown filter operator '{op}'. "
                    f"Valid operators: {list(_OPERATORS)}"
                )
            q._filters.append((field_name, op, value))
        return q
 
    def in_watershed(self, watershed_id: str) -> 'TerrainQuery':
        """Restrict to features whose feature_id appears in the given watershed set."""
        q = self._copy()
        q._relationships.append(("in_watershed", watershed_id))
        return q
 
    def visible_to(self, other: Union['TerrainQuery', List[ClassifiedFeature], str]) -> 'TerrainQuery':
        """
        Restrict to features visible to at least one feature in `other`.
 
        `other` may be:
            - Another TerrainQuery (resolved lazily at execute time)
            - A list of ClassifiedFeature objects (already resolved)
            - A single feature_id string
        """
        q = self._copy()
        q._relationships.append(("visible_to", other))
        return q
 
    def connected_to(self, feature_id: str) -> 'TerrainQuery':
        """Restrict to features directly connected in the connectivity graph."""
        q = self._copy()
        q._relationships.append(("connected_to", feature_id))
        return q
 
    def upstream_of(self, feature_id: str) -> 'TerrainQuery':
        """Restrict to features upstream of a given node in the flow network."""
        q = self._copy()
        q._relationships.append(("upstream_of", feature_id))
        return q
 
    def downstream_of(self, feature_id: str) -> 'TerrainQuery':
        """Restrict to features downstream of a given node in the flow network."""
        q = self._copy()
        q._relationships.append(("downstream_of", feature_id))
        return q
 
    def order_by(self, field: str, descending: bool = False) -> 'TerrainQuery':
        """Sort results by a feature attribute. None values sort last."""
        q = self._copy()
        q._order_field = field
        q._order_desc  = descending
        return q
 
    def limit(self, n: int) -> 'TerrainQuery':
        """Return at most n results (applied after sort and offset)."""
        if n < 1:
            raise ValueError(f"limit must be >= 1, got {n}")
        q = self._copy()
        q._limit_n = n
        return q
 
    def offset(self, n: int) -> 'TerrainQuery':
        """Skip the first n results (applied after sort, before limit)."""
        if n < 0:
            raise ValueError(f"offset must be >= 0, got {n}")
        q = self._copy()
        q._offset_n = n
        return q
 
    # -------------------------------------------------------------------------
    #  Execution
    # -------------------------------------------------------------------------
 
    def execute(self) -> List[ClassifiedFeature]:
        """
        Materialise the query. Steps:
            1. Resolve feature type → base list (copy)
            2. Apply attribute filters   (AND semantics)
            3. Apply relationship filters (AND semantics, in declaration order)
            4. Sort
            5. Offset → Limit
 
        Raises ValueError  if select() was never called.
        Raises RuntimeError if called recursively (circular visible_to guard).
        """
        if self._feature_type is None:
            raise ValueError("Call .select(FeatureType) before .execute()")
 
        if self._executing:
            raise RuntimeError(
                "Circular query detected: a query cannot be visible_to itself "
                "(directly or transitively)."
            )
 
        self._executing = True
        try:
            results = self._terrain._features_by_type(self._feature_type)
            results = self._run_filters(results)
            results = self._run_relationships(results)
            results = self._run_sort(results)
            results = self._run_slice(results)
        finally:
            self._executing = False
 
        return results
 
    # -------------------------------------------------------------------------
    #  Internal execution helpers
    # -------------------------------------------------------------------------
 
    def _run_filters(self, features: List[ClassifiedFeature]) -> List[ClassifiedFeature]:
        for field_name, op, value in self._filters:
            fn       = _OPERATORS[op]
            features = [f for f in features
                        if (v := self._resolve_field(f, field_name)) is not None
                        and fn(v, value)]
        return features
 
    def _run_relationships(self, features: List[ClassifiedFeature]) -> List[ClassifiedFeature]:
        for rel_type, rel_value in self._relationships:
            features = self._apply_relationship(features, rel_type, rel_value)
        return features
 
    def _run_sort(self, features: List[ClassifiedFeature]) -> List[ClassifiedFeature]:
        if self._order_field is None:
            return features
        field = self._order_field
        return sorted(
            features,
            key=lambda f: (self._resolve_field(f, field) is None,   # None → last
                           self._resolve_field(f, field)),
            reverse=self._order_desc,
        )
 
    def _run_slice(self, features: List[ClassifiedFeature]) -> List[ClassifiedFeature]:
        start = self._offset_n or 0
        end   = (start + self._limit_n) if self._limit_n is not None else None
        return features[start:end]
 
    def _resolve_field(self, feature: ClassifiedFeature, field: str) -> Any:
        """
        Resolve a field name to its value on a feature.
        Resolution order:
            1. Named attribute / property
            2. metadata dict fallback
            3. None (never raises — let the filter handle it)
        """
        if hasattr(feature, field):
            val = getattr(feature, field)
            # Properties that return callables are not values — skip them
            if callable(val) and not isinstance(val, (bool, int, float, str)):
                return None
            return val
        if hasattr(feature, "metadata") and field in feature.metadata:
            return feature.metadata[field]
        return None
 
    def _apply_relationship(self, features: List[ClassifiedFeature],
                            rel_type: str, rel_value: Any) -> List[ClassifiedFeature]:
 
        if rel_type == "in_watershed":
            member_ids = self._terrain.watersheds.get(rel_value, set())
            return [f for f in features if f.feature_id in member_ids]
 
        if rel_type == "visible_to":
            target_ids = self._resolve_visible_to_targets(rel_value)
            vis        = self._terrain.visibility_graph
            return [f for f in features
                    if vis.get(f.feature_id, set()) & target_ids]
 
        if rel_type == "connected_to":
            connected = self._terrain.connectivity_graph.get(rel_value, set())
            return [f for f in features if f.feature_id in connected]
 
        if rel_type == "upstream_of":
            upstream = self._collect_flow(rel_value, direction="up")
            return [f for f in features if f.feature_id in upstream]
 
        if rel_type == "downstream_of":
            downstream = self._collect_flow(rel_value, direction="down")
            return [f for f in features if f.feature_id in downstream]
 
        return features
 
    def _resolve_visible_to_targets(self, other: Any) -> Set[str]:
        """
        Resolve the argument to visible_to() into a set of feature IDs.
        Accepts a TerrainQuery, a list of ClassifiedFeature, or a str.
        """
        if isinstance(other, TerrainQuery):
            # Execute the subquery — the cycle guard on _executing will catch
            # mutual references before we recurse infinitely
            return {f.feature_id for f in other.execute()}
        if isinstance(other, list):
            return {f.feature_id for f in other if isinstance(f, ClassifiedFeature)}
        if isinstance(other, str):
            return {other}
        raise TypeError(
            f"visible_to() expects a TerrainQuery, List[ClassifiedFeature], or str. "
            f"Got: {type(other).__name__}"
        )
 
    def _collect_flow(self, start_id: str, direction: str) -> Set[str]:
        """
        Iterative BFS over the flow network.
 
        direction == "up"  : collect nodes that drain INTO start_id
        direction == "down": collect nodes that start_id drains INTO
        """
        flow    = self._terrain.flow_network
        visited : Set[str] = set()
        queue   : List[str] = [start_id]
 
        while queue:
            fid = queue.pop()
            if fid in visited:
                continue
            visited.add(fid)
 
            if direction == "down":
                # flow_network[fid] → list of nodes fid flows into
                for nid in flow.get(fid, []):
                    if nid not in visited:
                        queue.append(nid)
            else:
                # find all nodes whose downstream list contains fid
                for src, targets in flow.items():
                    if fid in targets and src not in visited:
                        queue.append(src)
 
        return visited