"""
Unified Test: Full Heightmap Analysis Pipeline

Runs all layers from 0-5 and produces a consolidated visualization:
- Left panel: 3 key intermediate layers (Slope, Curvature, Features)
- Right panel: Final Analyzed Terrain with tactical features (symbol-based)
- Console output: ASCII pipeline log showing each stage's outcome
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import pickle
from pathlib import Path
import warnings
from datetime import datetime

from core import (
    PipelineConfig, Heightmap, NormalizationConfig, GameType,
    RawImageInput, ScaledTransform, AnalyzedTerrain
)
from calibration import Layer0_Calibration_With_QualityMetrics
from lgeometry import Layer1_LocalGeometry
from rgeometry import Layer2_RegionalGeometry
from topological import Layer3_TopologicalFeatures
from relational import Layer4_Relational
from semantics import Layer5_Semantics


class PipelineLogger:
    """Unified logging with ASCII pipeline visualization."""
    
    def __init__(self):
        self.layers = []
        self.timings = []
    
    def log_start(self, image_path: str, game: GameType):
        print("\n" + "═" * 70)
        print(f"  HEIGHTMAP ANALYSIS PIPELINE")
        print(f"  Input: {Path(image_path).name}")
        print(f"  Game:  {game.value.upper()}")
        print("═" * 70)
        print()
    
    def log_layer(self, name: str, status: str, details: str = "", timing_ms: float = None):
        symbol = "[OK]" if "PASS" in status else "[WARNING]" if "WARN" in status else "[FAILED]"
        timing_str = f" [{timing_ms:.0f}ms]" if timing_ms else ""
        print(f"  {symbol} {name:<20} {status:<10}{timing_str}")
        if details:
            print(f"    └─ {details}")
    
    def log_summary(self, analyzed: AnalyzedTerrain, total_time_ms: float):
        print()
        print("═" * 70)
        print(f"  RESULTS")
        print("═" * 70)
        
        # Extract tactical counts
        def_count = sum(1 for p in analyzed.peaks if 'defensive_position' in p.metadata.get('semantic_tags', []))
        obs_count = sum(1 for p in analyzed.peaks if 'observation_post' in p.metadata.get('semantic_tags', []))
        choke_count = sum(1 for s in analyzed.saddles if 'chokepoint' in s.metadata.get('semantic_tags', []))
        assembly_count = sum(1 for f in analyzed.flat_zones if 'assembly_area' in f.metadata.get('semantic_tags', []))
        
        print(f"  Tactical Features:")
        print(f"    • Defensive Positions: {def_count}")
        print(f"    • Observation Posts:   {obs_count}")
        print(f"    • Chokepoints:         {choke_count}")
        print(f"    • Assembly Areas:      {assembly_count}")
        print()
        print(f"  Terrain Statistics:")
        print(f"    • Peaks:   {len(analyzed.peaks):>4}")
        print(f"    • Ridges:  {len(analyzed.ridges):>4}")
        print(f"    • Valleys: {len(analyzed.valleys):>4}")
        print(f"    • Saddles: {len(analyzed.saddles):>4}")
        print()
        print(f"  Total Time: {total_time_ms:.0f}ms")
        print("═" * 70)


class UnifiedAnalyzer:
    """Unified heightmap analyzer with logging and visualization."""
    
    def __init__(self, game_type: GameType = GameType.ARMA_3, verbose: bool = True):
        self.config = PipelineConfig(game_type=game_type)
        self.config.verbose = verbose
        self.logger = PipelineLogger() if verbose else None
        self.timings = {}
        self._grad = None  # Store gradient results
        self._curvature = None  # Store curvature results
    
    def run(self, image_path: str, georeferencing: dict = None) -> AnalyzedTerrain:
        """Run full pipeline on a heightmap image."""
        
        start_total = datetime.now()
        
        # Log start
        if self.logger:
            self.logger.log_start(image_path, self.config.game_type)
        
        # ----- LAYER 0: Calibration -----
        t0 = datetime.now()
        raw_input = RawImageInput(data=self._load_image(image_path))
        calibrator = Layer0_Calibration_With_QualityMetrics(self.config)
        heightmap, cal_metrics = calibrator.execute_with_metrics(raw_input, georeferencing=georeferencing)
        t0_ms = (datetime.now() - t0).total_seconds() * 1000
        
        if self.logger:
            details = f"elev {cal_metrics['elevation_stats']['min']:.0f}-{cal_metrics['elevation_stats']['max']:.0f}m, noise={cal_metrics['noise_estimate']:.3f}m"
            self.logger.log_layer("0: Calibration", "PASS", details, t0_ms)
        
        # ----- LAYER 1: Slope & Aspect -----
        t1 = datetime.now()
        layer1 = Layer1_LocalGeometry(self.config)
        grad = layer1.execute(heightmap)
        self._grad = grad  # Store for visualization
        t1_ms = (datetime.now() - t1).total_seconds() * 1000
        
        if self.logger:
            details = f"slope {np.min(grad['slope']):.0f}°–{np.max(grad['slope']):.0f}°, mean={np.mean(grad['slope']):.1f}°"
            self.logger.log_layer("1: Local Geometry", "PASS", details, t1_ms)
        
        # ----- LAYER 2: Curvature -----
        t2 = datetime.now()
        layer2 = Layer2_RegionalGeometry(self.config)
        curvature = layer2.execute(heightmap)
        self._curvature = curvature  # Store for visualization
        t2_ms = (datetime.now() - t2).total_seconds() * 1000
        
        if self.logger:
            h_eps, k_eps = layer2.epsilon_used
            type_counts = {
                'FLAT': np.sum(curvature['curvature_type'] == 'FLAT'),
                'CONVEX': np.sum(curvature['curvature_type'] == 'CONVEX'),
                'CONCAVE': np.sum(curvature['curvature_type'] == 'CONCAVE'),
                'SADDLE': np.sum(curvature['curvature_type'] == 'SADDLE')
            }
            details = f"H_ε={h_eps:.2e}, K_ε={k_eps:.2e}, convex={type_counts['CONVEX']}, concave={type_counts['CONCAVE']}, saddle={type_counts['SADDLE']}"
            self.logger.log_layer("2: Regional Geometry", "PASS", details, t2_ms)
        
        # ----- LAYER 3: Topological Features -----
        t3 = datetime.now()
        bundle = {
            'heightmap': heightmap,
            'curvature': curvature['curvature'],
            'gaussian_curvature': curvature['gaussian_curvature'],
            'curvature_type': curvature['curvature_type'],
            'slope': grad['slope'],
            'aspect': grad['aspect']
        }
        layer3 = Layer3_TopologicalFeatures(self.config)
        features = layer3.execute(bundle)
        t3_ms = (datetime.now() - t3).total_seconds() * 1000
        
        if self.logger:
            peaks = len([f for f in features if f.__class__.__name__ == 'PeakFeature'])
            ridges = len([f for f in features if f.__class__.__name__ == 'RidgeFeature'])
            details = f"peaks={peaks}, ridges={ridges}, saddles={len([f for f in features if f.__class__.__name__ == 'SaddleFeature'])}"
            status = "PASS" if peaks > 0 else "WARN"
            self.logger.log_layer("3: Topological Features", status, details, t3_ms)
        
        # ----- LAYER 4: Relational Graphs -----
        t4 = datetime.now()
        bundle['features'] = features
        layer4 = Layer4_Relational(self.config)
        graphs = layer4.execute(bundle)
        t4_ms = (datetime.now() - t4).total_seconds() * 1000
        
        if self.logger:
            vis_edges = sum(len(v) for v in graphs['visibility_graph'].values()) // 2
            conn_edges = sum(len(v) for v in graphs['connectivity_graph'].values()) // 2
            details = f"visibility={vis_edges} edges, connectivity={conn_edges} edges"
            self.logger.log_layer("4: Relational Graphs", "PASS", details, t4_ms)
        
        # ----- LAYER 5: Semantics -----
        t5 = datetime.now()
        bundle.update(graphs)
        bundle['slope'] = grad['slope']
        layer5 = Layer5_Semantics(self.config)
        analyzed = layer5.execute(bundle)
        t5_ms = (datetime.now() - t5).total_seconds() * 1000
        
        if self.logger:
            def_count = sum(1 for p in analyzed.peaks if 'defensive_position' in p.metadata.get('semantic_tags', []))
            details = f"defensive={def_count}, observation={len([p for p in analyzed.peaks if 'observation_post' in p.metadata.get('semantic_tags', [])])}"
            self.logger.log_layer("5: Semantics", "PASS", details, t5_ms)
        
        total_ms = (datetime.now() - start_total).total_seconds() * 1000
        
        if self.logger:
            self.logger.log_summary(analyzed, total_ms)
        
        return analyzed
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and convert image to grayscale uint8."""
        from PIL import Image
        img = Image.open(image_path)
        if img.mode != 'L':
            img = img.convert('L')
        return np.array(img, dtype=np.uint8)
    
    def visualize(self, analyzed: AnalyzedTerrain, save_path: str = None):
        """
        Create consolidated visualization with tactical map symbols.
        Uses standard military/tactical symbology where applicable.
        """
        heightmap = analyzed.source_heightmap
        z = heightmap.data
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(16, 10))
        
        # Define GridSpec: 2 columns, left column has 3 rows, right column spans full height
        gs = GridSpec(3, 2, figure=fig, width_ratios=[0.5, 1], height_ratios=[1, 1, 1])
        
        # LEFT COLUMN - 3 smaller plots
        ax_slope = fig.add_subplot(gs[0, 0])
        ax_curvature = fig.add_subplot(gs[1, 0])
        ax_features = fig.add_subplot(gs[2, 0])
        
        # RIGHT COLUMN - Main portrait plot
        ax_main = fig.add_subplot(gs[:, 1])
        
        # ----- LEFT 1: Slope Map -----
        if self._grad is not None:
            slope = self._grad['slope']
            im_slope = ax_slope.imshow(slope, cmap='hot', vmax=45)
            ax_slope.set_title('Slope (degrees)', fontsize=10, fontweight='bold')
            ax_slope.axis('off')
            plt.colorbar(im_slope, ax=ax_slope, shrink=0.8)
        
        # ----- LEFT 2: Curvature Type Map -----
        if self._curvature is not None:
            curvature_type = self._curvature['curvature_type']
            type_to_int = {"FLAT": 0, "CONVEX": 1, "CONCAVE": 2, "SADDLE": 3}
            type_numeric = np.vectorize(type_to_int.get)(curvature_type)
            im_curv = ax_curvature.imshow(type_numeric, cmap='Set3', vmin=0, vmax=3)
            ax_curvature.set_title('Curvature Type\n(Flat, Convex, Concave, Saddle)', fontsize=10, fontweight='bold')
            ax_curvature.axis('off')
            cbar = plt.colorbar(im_curv, ax=ax_curvature, ticks=[0, 1, 2, 3], shrink=0.8)
            cbar.ax.set_yticklabels(['FLAT', 'CONVEX', 'CONCAVE', 'SADDLE'], fontsize=8)
        
        # ----- LEFT 3: Feature Overlay (Ridges + Valleys + Peaks) -----
        ax_features.imshow(z, cmap='terrain', alpha=0.6)
        
        # Draw ridges
        for ridge in analyzed.ridges:
            if ridge.spine_points:
                xs, ys = zip(*ridge.spine_points)
                ax_features.plot(xs, ys, 'r-', linewidth=1.5, alpha=0.8, label='_nolegend_')
        
        # Draw valleys
        for valley in analyzed.valleys:
            if valley.spine_points:
                xs, ys = zip(*valley.spine_points)
                ax_features.plot(xs, ys, 'b-', linewidth=1.5, alpha=0.8, label='_nolegend_')
        
        # Mark peaks
        for peak in analyzed.peaks:
            x, y = peak.centroid
            ax_features.plot(x, y, '^', markersize=8, color='red', markeredgecolor='white', markeredgewidth=0.5, label='_nolegend_')
        
        ax_features.set_title(f'Features: Ridges (red), Valleys (blue), Peaks (▲)', fontsize=10, fontweight='bold')
        ax_features.axis('off')
        
        # ----- RIGHT: Final Analyzed Terrain (Tactical Map Symbols) -----
        ax_main.imshow(z, cmap='terrain', alpha=0.7)
        
        # Draw defensive positions (▲ filled red)
        for peak in analyzed.peaks:
            tags = peak.metadata.get('semantic_tags', [])
            if 'defensive_position' in tags:
                x, y = peak.centroid
                ax_main.plot(x, y, '^', markersize=12, color='red',
                           markeredgecolor='black', markeredgewidth=1, linestyle='none',
                           markerfacecolor='red')
        
        # Draw observation posts (● with dot center - use circle)
        for peak in analyzed.peaks:
            tags = peak.metadata.get('semantic_tags', [])
            if 'observation_post' in tags:
                x, y = peak.centroid
                ax_main.plot(x, y, 'o', markersize=10, color='blue',
                           markeredgecolor='black', markeredgewidth=1, linestyle='none',
                           markerfacecolor='lightblue')
        
        # Draw chokepoints (◆ diamond)
        for saddle in analyzed.saddles:
            tags = saddle.metadata.get('semantic_tags', [])
            if 'chokepoint' in tags:
                x, y = saddle.centroid
                ax_main.plot(x, y, 'D', markersize=8, color='orange',
                           markeredgecolor='black', markeredgewidth=0.5, linestyle='none',
                           markerfacecolor='orange')
        
        # Draw assembly areas (□ square)
        for flat in analyzed.flat_zones:
            tags = flat.metadata.get('semantic_tags', [])
            if 'assembly_area' in tags or 'major_assembly_area' in tags:
                x, y = flat.centroid
                ax_main.plot(x, y, 's', markersize=10, color='green',
                           markeredgecolor='black', markeredgewidth=1, linestyle='none',
                           markerfacecolor='lightgreen')
        
        # Draw ridges with cover (thick brown line)
        for ridge in analyzed.ridges:
            if ridge.spine_points:
                xs, ys = zip(*ridge.spine_points)
                tags = ridge.metadata.get('semantic_tags', [])
                if 'defensive_cover' in tags:
                    ax_main.plot(xs, ys, '-', linewidth=3, color='brown', alpha=0.9)
                else:
                    ax_main.plot(xs, ys, '-', linewidth=1, color='gray', alpha=0.6)
        
        # Draw valleys with ambush potential (wavy line effect - use dashed)
        for valley in analyzed.valleys:
            if valley.spine_points:
                xs, ys = zip(*valley.spine_points)
                tags = valley.metadata.get('semantic_tags', [])
                if 'ambush_potential' in tags:
                    ax_main.plot(xs, ys, '--', linewidth=2, color='purple', alpha=0.8)
                else:
                    ax_main.plot(xs, ys, '-', linewidth=1, color='blue', alpha=0.6)
        
        # Draw peaks (▲ hollow - only those not already marked)
        for peak in analyzed.peaks:
            tags = peak.metadata.get('semantic_tags', [])
            if 'defensive_position' not in tags and 'observation_post' not in tags:
                x, y = peak.centroid
                ax_main.plot(x, y, '^', markersize=10, color='black',
                           markeredgecolor='black', markeredgewidth=1, linestyle='none',
                           markerfacecolor='none')
        
        # Legend with tactical symbols
        legend_elements = [
            plt.Line2D([0], [0], marker='^', color='red', linestyle='none', markersize=12, 
                      markerfacecolor='red', markeredgecolor='black', label='Defensive Position'),
            plt.Line2D([0], [0], marker='o', color='blue', linestyle='none', markersize=10,
                      markerfacecolor='lightblue', markeredgecolor='black', label='Observation Post'),
            plt.Line2D([0], [0], marker='D', color='orange', linestyle='none', markersize=8,
                      markerfacecolor='orange', markeredgecolor='black', label='Chokepoint'),
            plt.Line2D([0], [0], marker='s', color='green', linestyle='none', markersize=10,
                      markerfacecolor='lightgreen', markeredgecolor='black', label='Assembly Area'),
            plt.Line2D([0], [0], color='brown', linewidth=3, label='Cover Ridge'),
            plt.Line2D([0], [0], color='purple', linestyle='--', linewidth=2, label='Ambush Valley'),
            plt.Line2D([0], [0], marker='^', color='black', linestyle='none', markersize=10,
                      markerfacecolor='none', markeredgecolor='black', label='Peak')
        ]
        ax_main.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
        
        ax_main.set_title('TACTICAL TERRAIN ANALYSIS\nMilitary Map Symbology', fontsize=12, fontweight='bold')
        ax_main.axis('off')
        
        plt.suptitle(f'Heightmap Analysis: {self.config.game_type.value.upper()} Profile', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nVisualization saved: {save_path}")
        #plt.show()

def main():
    """Run unified test."""
    image_path = "heightmap.png"
    if not Path(image_path).exists():
        print(f"Error: {image_path} not found!")
        return
    analyzer = UnifiedAnalyzer(game_type=GameType.CUSTOM, verbose=True)
    analyzed = analyzer.run(image_path)
    analyzer.visualize(analyzed, save_path="unified_analysis.png")
    return analyzed


if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=UserWarning)
    analyzed = main()