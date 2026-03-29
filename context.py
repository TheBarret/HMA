"""
TerrainContext — LLM-friendly terrain description layer.

Layered design: each context layer builds on the previous.
Verify each layer produces sensible output before enabling the next.

Current: Layer 0 (Calibration surface statistics)
Pending: Layer 1 (Slope/Aspect), Layer 2 (Curvature), Layer 3 (Features),
         Layer 4 (Relational), Layer 5 (Semantics)
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from core import (
    AnalyzedTerrain,
    PipelineConfig,
    Heightmap,
)


# =============================================================================
# Output contract
# =============================================================================

@dataclass
class ContextBlock:
    """
    A single layer's contribution to the terrain context.

    Each pipeline layer produces one ContextBlock.
    Blocks are accumulated into TerrainContext and can be
    serialized individually or combined into a full narrative.
    """
    layer:       int           # 0-5, which pipeline layer produced this
    label:       str           # short human label e.g. "CALIBRATION SURFACE"
    summary:     str           # one-paragraph plain text summary
    stats:       Dict[str, Any] = field(default_factory=dict)   # structured numbers
    notes:       List[str]     = field(default_factory=list)    # bullet observations
    confidence:  float         = 1.0                            # 0-1, data quality

    def to_text(self) -> str:
        """Render block as plain text for console or LLM prompt."""
        lines = [
            f"[L{self.layer}] {self.label}",
            "-" * 50,
            self.summary,
        ]
        if self.notes:
            lines.append("")
            for note in self.notes:
                lines.append(f"  • {note}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON export."""
        return {
            "layer":      self.layer,
            "label":      self.label,
            "summary":    self.summary,
            "stats":      self.stats,
            "notes":      self.notes,
            "confidence": self.confidence,
        }


# =============================================================================
# TerrainContext
# =============================================================================

class TerrainContext:
    """
    Stateless wrapper around pipeline outputs that produces
    human-readable and LLM-ready terrain descriptions.

    Usage (Layer 0 only):
        context = TerrainContext(config)
        context.ingest_layer0(heightmap)
        print(context.describe())

    Usage (full pipeline):
        context = TerrainContext(config)
        context.ingest_layer0(heightmap)
        context.ingest_layer1(slope, aspect)
        context.ingest_layer2(curvature, curvature_type)
        context.ingest_layer3(features)
        print(context.describe())
        print(context.to_json())
    """

    def __init__(self, config: PipelineConfig):
        self.config  = config
        self._blocks: List[ContextBlock] = []   # one per layer ingested

    # =========================================================================
    # Layer ingestion — call each after the corresponding pipeline layer runs
    # =========================================================================

    def ingest_layer0(self, heightmap: Heightmap) -> ContextBlock:
        """
        Describe the calibrated elevation surface.
        Produces: elevation statistics, terrain extent, surface character.
        """
        data      = heightmap.data
        cell      = heightmap.config.horizontal_scale
        h, w      = data.shape
        sea_off   = heightmap.config.sea_level_offset

        # --- Core statistics ---
        elev_min  = float(data.min())
        elev_max  = float(data.max())
        elev_mean = float(data.mean())
        elev_std  = float(data.std())
        elev_range = elev_max - elev_min

        # --- Extent ---
        width_m  = w * cell
        height_m = h * cell
        area_km2 = (width_m * height_m) / 1_000_000

        # --- Land vs sea estimate ---
        # Pixels above sea_level_offset are "land"
        land_mask    = data > sea_off
        land_pct     = 100.0 * float(land_mask.sum()) / data.size
        sea_pct      = 100.0 - land_pct

        # --- Elevation percentiles for character description ---
        p10  = float(np.percentile(data, 10))
        p25  = float(np.percentile(data, 25))
        p50  = float(np.percentile(data, 50))
        p75  = float(np.percentile(data, 75))
        p90  = float(np.percentile(data, 90))

        # --- Hypsometric character (elevation distribution shape) ---
        # Skew: positive = most terrain is low, negative = most terrain is high
        skew = float((elev_mean - p50) / (elev_std + 1e-8))

        upper_range = elev_max - p75   # top 25% elevation span
        lower_range = p25 - elev_min   # bottom 25% elevation span
        ratio = upper_range / (lower_range + 1e-8)

        if   ratio > 2.0:  hyps = "highland-dominant (significant upper elevation mass)"
        elif ratio < 0.5:  hyps = "lowland-dominant (most terrain near base elevation)"
        else:              hyps = "balanced relief (even spread of elevations)"

        # --- Relief character ---
        if   elev_range < 10:   relief = "flat (< 10m total relief)"
        elif elev_range < 50:   relief = "gentle (10–50m relief)"
        elif elev_range < 150:  relief = "moderate (50–150m relief)"
        elif elev_range < 500:  relief = "significant (150–500m relief)"
        else:                   relief = "extreme (> 500m relief)"

        # --- Quadrant elevation breakdown ---
        quads = _quadrant_stats(data)

        # --- Notes ---
        notes = []
        notes.append(f"Map extent: {width_m:.0f}m × {height_m:.0f}m ({area_km2:.2f} km²)")
        notes.append(f"Cell resolution: {cell:.1f}m/pixel  |  Grid: {w}×{h} px")
        notes.append(f"Elevation span: {elev_min:.1f}m → {elev_max:.1f}m  (Δ{elev_range:.1f}m)")
        notes.append(f"Median elevation: {p50:.1f}m  |  Mean: {elev_mean:.1f}m  |  σ: {elev_std:.1f}m")
        notes.append(f"Land coverage: {land_pct:.1f}%  |  Sea/below-reference: {sea_pct:.1f}%")
        notes.append(f"Relief character: {relief}")
        notes.append(f"Hypsometric character: {hyps}")
        notes.append(
            f"Elevation quartiles: "
            f"P25={p25:.1f}m  P50={p50:.1f}m  P75={p75:.1f}m  P90={p90:.1f}m"
        )

        # Quadrant notes
        quad_lines = []
        for name, (qmin, qmax, qmean) in quads.items():
            quad_lines.append(f"{name}({qmean:.0f}m avg)")
        notes.append("Quadrant means: " + "  ".join(quad_lines))

        # Anomaly flags
        if land_pct < 20:
            notes.append("! Very low land coverage — predominantly water/sea surface")
        if elev_std < 1.0:
            notes.append("! Near-zero elevation variance — likely featureless or synthetic input")
        if elev_range > self.config.max_elevation_range:
            notes.append(f"! Elevation range exceeds config limit ({self.config.max_elevation_range}m)")

        # --- Summary paragraph ---
        summary = (
            f"The calibrated surface covers {width_m:.0f}m × {height_m:.0f}m "
            f"({area_km2:.2f} km²) at {cell:.1f}m/pixel resolution.\n"
            f"Elevation ranges from {elev_min:.1f}m to {elev_max:.1f}m "
            f"(Δ{elev_range:.1f}m), with a median of {p50:.1f}m.\n"
            f"The terrain is {relief} and {hyps}.\n"
            f"Approximately {land_pct:.0f}% of the map is above the "
            f"{sea_off:.1f}m sea-level reference.\n"
        )

        block = ContextBlock(
            layer=0,
            label="CALIBRATION SURFACE",
            summary=summary,
            stats={
                "extent": {
                    "width_m":   width_m,
                    "height_m":  height_m,
                    "area_km2":  area_km2,
                    "grid_px":   [w, h],
                    "cell_m":    cell,
                },
                "elevation": {
                    "min":    elev_min,
                    "max":    elev_max,
                    "mean":   elev_mean,
                    "median": p50,
                    "std":    elev_std,
                    "range":  elev_range,
                    "p10":    p10,
                    "p25":    p25,
                    "p75":    p75,
                    "p90":    p90,
                },
                "coverage": {
                    "land_pct": land_pct,
                    "sea_pct":  sea_pct,
                },
                "character": {
                    "relief":       relief,
                    "hypsometric":  hyps,
                    "skew":         skew,
                },
                "quadrants": {
                    name: {"min": v[0], "max": v[1], "mean": v[2]}
                    for name, v in quads.items()
                },
            },
            notes=notes,
            confidence=1.0,
        )

        self._blocks.append(block)
        return block

    def ingest_layer1(self, slope: np.ndarray, aspect: np.ndarray) -> ContextBlock:
        """Layer 1: Slope and aspect description."""
        
        # --- Core statistics ---
        slope_mean = float(slope.mean())
        slope_max = float(slope.max())
        slope_std = float(slope.std())
        
        # Slope distribution buckets
        flat_pct = 100.0 * np.sum(slope < 5) / slope.size      # < 5°: walkable
        moderate_pct = 100.0 * np.sum((slope >= 5) & (slope < 15)) / slope.size
        steep_pct = 100.0 * np.sum(slope >= 15) / slope.size
        
        # Aspect dominance (which directions face most?)
        aspect_bins = np.histogram(aspect[~np.isnan(aspect)], bins=8, range=(0, 360))[0]
        dominant_aspect_idx = np.argmax(aspect_bins)
        aspect_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        dominant_aspect = aspect_labels[dominant_aspect_idx]
        
        # --- Notes ---
        notes = [
            f"Average slope: {slope_mean:.1f}° ± {slope_std:.1f}° (max: {slope_max:.1f}°)",
            f"Slope distribution: {flat_pct:.0f}% flat, {moderate_pct:.0f}% moderate, {steep_pct:.0f}% steep",
            f"Dominant aspect: {dominant_aspect}-facing slopes",
        ]
        
        # --- Summary ---
        if steep_pct > 30:
            slope_char = "predominantly steep, challenging for ground movement"
        elif moderate_pct > 50:
            slope_char = "moderately sloped, suitable for most vehicles with route planning"
        else:
            slope_char = "generally gentle, highly traversable"
        
        summary = (
            f"Surface slopes average {slope_mean:.1f}°, with {steep_pct:.0f}% exceeding 15°. "
            f"The terrain is {slope_char}. "
            f"{dominant_aspect}-facing slopes dominate the aspect distribution."
        )
        
        block = ContextBlock(
            layer=1,
            label="SLOPE & ASPECT",
            summary=summary,
            stats={
                "slope": {
                    "mean": slope_mean, "max": slope_max, "std": slope_std,
                    "flat_pct": flat_pct, "moderate_pct": moderate_pct, "steep_pct": steep_pct
                },
                "aspect": {
                    "dominant": dominant_aspect,
                    "distribution": aspect_bins.tolist()
                }
            },
            notes=notes,
            confidence=0.95,
        )
        
        self._blocks.append(block)
        return block
    
    # =========================================================================
    # To be implemented as each layer is verified
    # =========================================================================

    #def ingest_layer1(self, slope: np.ndarray, aspect: np.ndarray) -> ContextBlock:
    #    """Layer 1: Slope and aspect description. Not yet implemented."""
    #    raise NotImplementedError("Layer 1 context not yet implemented — verify Layer 0 first")

    def ingest_layer2(self, curvature: np.ndarray, curvature_type: np.ndarray) -> ContextBlock:
        """Layer 2: Curvature description. Not yet implemented."""
        raise NotImplementedError("Layer 2 context not yet implemented")

    def ingest_layer3(self, features: list) -> ContextBlock:
        """Layer 3: Topological feature description. Not yet implemented."""
        raise NotImplementedError("Layer 3 context not yet implemented")

    def ingest_layer4(self, terrain: AnalyzedTerrain) -> ContextBlock:
        """Layer 4: Relational/graph description. Not yet implemented."""
        raise NotImplementedError("Layer 4 context not yet implemented")

    def ingest_layer5(self, terrain: AnalyzedTerrain) -> ContextBlock:
        """Layer 5: Semantic/tactical description. Not yet implemented."""
        raise NotImplementedError("Layer 5 context not yet implemented")

    # =========================================================================
    # Output
    # =========================================================================

    def describe(self) -> str:
        """
        Full plain-text context string.
        Concatenates all ingested layer blocks in order.
        Suitable for console output or LLM system prompt injection.
        """
        if not self._blocks:
            return "[TerrainContext] No layers ingested yet."

        sections = []
        for block in sorted(self._blocks, key=lambda b: b.layer):
            sections.append(block.to_text())

        header = "TERRAIN CONTEXT REPORT"
        divider = "=" * 50
        return f"{divider}\n{header}\n{divider}\n\n" + f"\n\n{'—'*50}\n\n".join(sections)

    def describe_layer(self, layer: int) -> str:
        """Return plain text for a single layer block."""
        block = self._get_block(layer)
        if block is None:
            return f"[TerrainContext] Layer {layer} not yet ingested."
        return block.to_text()

    def to_json(self, indent: int = 2) -> str:
        """Serialize all ingested blocks as JSON."""
        return json.dumps(
            [b.to_dict() for b in sorted(self._blocks, key=lambda b: b.layer)],
            indent=indent
        )

    def to_llm_prompt(self, max_layers: Optional[int] = None) -> str:
        """
        Format context for injection into an LLM system prompt.
        Concise — summaries only, no bullet notes, to preserve token budget.
        """
        if not self._blocks:
            return ""

        blocks = sorted(self._blocks, key=lambda b: b.layer)
        if max_layers is not None:
            blocks = blocks[:max_layers]

        lines = ["TERRAIN ANALYSIS DATA", "=" * 40]
        for block in blocks:
            lines.append(f"\n[{block.label}]")
            lines.append(block.summary)

        return "\n".join(lines)

    def stats(self, layer: int) -> Optional[Dict[str, Any]]:
        """Return raw stats dict for a given layer, or None if not ingested."""
        block = self._get_block(layer)
        return block.stats if block else None

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _get_block(self, layer: int) -> Optional[ContextBlock]:
        for b in self._blocks:
            if b.layer == layer:
                return b
        return None


# =============================================================================
# Module-level helpers
# =============================================================================

def _quadrant_stats(data: np.ndarray) -> Dict[str, tuple]:
    """
    Split array into four quadrants and return (min, max, mean) per quadrant.
    Keys: northwest, northeast, southwest, southeast.
    """
    h, w = data.shape
    mh, mw = h // 2, w // 2
    quads = {
        "NW": data[:mh, :mw],
        "NE": data[:mh, mw:],
        "SW": data[mh:, :mw],
        "SE": data[mh:, mw:],
    }
    return {
        name: (float(q.min()), float(q.max()), float(q.mean()))
        for name, q in quads.items()
    }