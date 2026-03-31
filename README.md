# Heightmap Analysis Framework (WIP)

## Core Philosophy
A heightmap is a 2.5D scalar field: perfect geometric surface information, zero information about what lies above, 
below, or what the surface means. Analysis must proceed from fundamental mathematics to derived semantics.

The pipeline architecture and calibrating is done by generate `Ground Truth Validation`,  
using `synthetic terrain`  to generate heightmaps with known features.  

<img src="t_flatzone.png" /> <img src="t_single_peak.png" /> <img src="t_ridge.png" /> <img src="t_saddle.png" />

Compare detected output features against ground truth will calibrate our model, using predefined config parameters located `core.py`.  

<img width="400" src="https://github.com/user-attachments/assets/133cddbf-0c8c-488f-9b9d-3eeb8feea246" /> <img width="380" src="https://github.com/user-attachments/assets/6a75743e-2ca6-422e-9423-c49e607d0f88" />

# Early Testing: Relational Analysis

Relations are NOT derived from feature heuristics.  
Relations are derived from continuous mathematical fields, onto which features are mapped.  

1. Compute Pixel-Level Fields (Mathematics)  
   - Flow Direction, Accumulation, Cost Surfaces  
2. Extract Relational Structures (Topology)  
   - Stream Networks, Watershed Boundaries, Visibility Lines  
3. Map Discrete Features (Semantics)  
   - Assign Features to Basins, Build Feature Graphs  


<img width="1024" src="https://github.com/user-attachments/assets/421c5e97-b9e1-442d-89b7-7170cd34070c" />

# LLM Friendly context summary (PLANNED FEATURE, WIP...)

The pipeline comes with a `context` module that will wrap arround the data and contextualizes it,  
where the idea is that you can `query` the context:  

`context.question("Where can I set up an observation post?")`  


# Example output of `t_saddle.png`

```
==================================================
TERRAIN CONTEXT REPORT
==================================================

[L0] CALIBRATION SURFACE
--------------------------------------------------
The calibrated surface covers 256m × 256m (0.07 km²) at 2.0m/pixel resolution.
Elevation ranges from 0.1m to 99.7m (Δ99.6m), with a median of 0.1m.
The terrain is moderate (50–150m relief) and highland-dominant (significant upper elevation mass).
Approximately 26% of the map is above the 0.1m sea-level reference.


  • Map extent: 256m × 256m (0.07 km²)
  • Cell resolution: 2.0m/pixel  |  Grid: 128×128 px
  • Elevation span: 0.1m → 99.7m  (Δ99.6m)
  • Median elevation: 0.1m  |  Mean: 8.9m  |  σ: 21.1m
  • Land coverage: 26.4%  |  Sea/below-reference: 73.6%
  • Relief character: moderate (50–150m relief)
  • Hypsometric character: highland-dominant (significant upper elevation mass)
  • Elevation quartiles: P25=0.1m  P50=0.1m  P75=0.1m  P90=44.8m
  • Quadrant means: NW(8m avg)  NE(9m avg)  SW(9m avg)  SE(9m avg)

——————————————————————————————————————————————————

[L1] SLOPE & ASPECT
--------------------------------------------------
Surface slopes average 12.7°, with 21% exceeding 15°.
The terrain is generally gentle, highly traversable.
N-facing slopes dominate the aspect distribution.

  • Average slope: 12.7° ± 25.1° (max: 77.1°)
  • Slope distribution: 78% flat, 1% moderate, 21% steep
  • Dominant aspect: N-facing slopes
[Visualizer] Saved → t_saddle_topology.png
```


---

## The Analysis Pipeline

### Layer 0: Calibration (The Observable)
*Before examining data, define the measurement context.*

- **Input:** Raw grayscale image (0-255)
- **Operations:**
  - Normalize values (convert to real-world units if metadata available)
  - Apply noise reduction (Gaussian blur)
- **Rationale:** Calculus on noisy data produces unusable derivatives. Every pixel fluctuation becomes a false feature.
- **Output:** Clean mathematical surface ready for analysis

### Layer 1: Local Geometry (First Derivatives)
*How does the surface behave at this exact point?*

- **Input:** Calibrated surface
- **Operations:**
  - Calculate gradient magnitude → **Slope** (steepness)
  - Calculate gradient direction → **Aspect** (compass orientation)
- **Rationale:** Foundation for all flow-based analysis. Water flows down gradient; vehicles climb based on slope.
- **Output:** Slope map (0-90°) and Aspect map (0-360°)

### Layer 2: Regional Geometry (Second Derivatives)
*What shape does the neighborhood form around this point?*

*`Mean Curvature (H)` = "How bent is the surface?"*  
A `Large positive Surface` that bulges `outward`; Hilltop, ridge crest, nose of slope  
A `Large negative Surface` that bulges `inward`; Valley bottom, gully, depression  
Or `Near zero Surface` is `flat` or a  `saddle` (needs K to distinguish); Flat plain, planar slope, cylindrical ridge  

*`Gaussian Curvature (K)` = "What shape is the bend?"*  
A `Positive` same sign in both directions is a Dome (peak), bowl (depression)  
A `Negative` opposite signs is a Saddle, pass, twisted surface  
Or a `Near zero` one direction flat can be a Ridge (curved across, flat along), valley  

- **Input:** Gradient maps
- **Operations:**
  - Calculate profile curvature (direction of slope)
  - Calculate plan curvature (across slope)
  - Derive convexity/concavity
- **Rationale:** A ridge and valley can have identical slope but opposite curvature. Need second derivatives to distinguish.
- **Output:** Curvature map (positive = convex, negative = concave)

### Layer 3: Topological Features (Discrete Objects)
*Where do continuous shapes resolve into identifiable structures?*

- **Input:** Elevation + Curvature maps
- **Operations:**
  - Identify critical points: peaks (local maxima), pits (local minima), saddles
  - Extract linear features: ridgelines, valley bottoms, drainage networks
- **Rationale:** Transform continuous field into discrete entities. A "peak" requires both local maximum and convex surroundings.
- **Output:** Feature set: points (peaks/pits/saddles) and polylines (ridges/channels)

### Layer 4: Relational Analysis (Connectivity)
*How do these features interact spatially?*

- **Input:** Topological features + Slope + Aspect
- **Operations:**
  - Calculate viewsheds (visibility from points)
  - Compute flow accumulation (hydrological networks)
  - Build connectivity graphs (ridge networks, drainage hierarchies)
- **Rationale:** Can't determine visibility without observer locations; can't model flow without aspect.
- **Output:** Relational maps and graphs (viewshed polygons, stream networks)

### Layer 5: Semantic Interpretation (Meaning)
*What does this mean for an end user?*

- **Input:** Relational data + domain thresholds
- **Operations:**
  - Classify traversable zones (slope < vehicle limit)
  - Identify tactical features (chokepoints, defensive positions)
  - Apply domain-specific rules
- **Rationale:** A chokepoint = two steep ridges + narrow flat valley. Requires all previous layers to define.
- **Output:** Domain-classified terrain (tactical map, accessibility zones)


## License

Open source, for the world to use how they see fit with the promise it be legal.
