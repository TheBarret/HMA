# Heightmap Analysis Framework (WIP)

## Core Philosophy
A heightmap is a 2.5D scalar field: perfect geometric surface information, zero information about what lies above, 
below, or what the surface means. Analysis must proceed from fundamental mathematics to derived semantics.

The pipeline architecture and calibrating is done by generate `Ground Truth Validation`,  
using `synthetic terrain`  to generate heightmaps with known features.  

<img src="t_flatzone.png" /> <img src="t_single_peak.png" /> <img src="t_ridge.png" /> <img src="t_saddle.png" />

Compare detected output features against ground truth will calibrate our model, using predefined config parameters located `core.py`.  

<img width="400" src="https://github.com/user-attachments/assets/d0a403b5-128b-448f-9a8c-346ef5fc9089" /><img width="380" src="https://github.com/user-attachments/assets/6a75743e-2ca6-422e-9423-c49e607d0f88" />

# LLM Friendly context summary

```
==================================================
TERRAIN CONTEXT REPORT T_SADDLE.PNG
==================================================

[L0] CALIBRATION SURFACE
--------------------------------------------------
The calibrated surface covers 256m × 256m (0.07 km²) at 2.0m/pixel resolution.
Elevation ranges from 0.2m to 1016.2m (Δ1016.0m), with a median of 0.2m.
The terrain is extreme (> 500m relief) and highland-dominant (significant upper elevation mass).
Approximately 26% of the map is above the 0.2m sea-level reference.


  • Map extent: 256m × 256m (0.07 km²)
  • Cell resolution: 2.0m/pixel  |  Grid: 128×128 px
  • Elevation span: 0.2m → 1016.2m  (Δ1016.0m)
  • Median elevation: 0.2m  |  Mean: 90.4m  |  σ: 215.3m
  • Land coverage: 26.4%  |  Sea/below-reference: 73.6%
  • Relief character: extreme (> 500m relief)
  • Hypsometric character: highland-dominant (significant upper elevation mass)
  • Elevation quartiles: P25=0.2m  P50=0.2m  P75=0.2m  P90=455.8m
  • Quadrant means: NW(86m avg)  NE(89m avg)  SW(92m avg)  SE(95m avg)

——————————————————————————————————————————————————

[L1] SLOPE & ASPECT
--------------------------------------------------
Surface slopes average 19.2°, with 23% exceeding 15°. The terrain is generally gentle, highly traversable. N-facing slopes dominate the aspect distribution.

  • Average slope: 19.2° ± 35.4° (max: 88.7°)
  • Slope distribution: 76% flat, 0% moderate, 23% steep
  • Dominant aspect: N-facing slopes
```

# Early Testing

<img width="1024" src="https://github.com/user-attachments/assets/4ba4be03-a5b1-4ab7-9219-6e6bb7b93947" />


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
