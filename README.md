# Heightmap Analysis Framework (WIP)

## Core Philosophy
A heightmap is a 2.5D scalar field: perfect geometric surface information, zero information about what lies above, 
below, or what the surface means. Analysis must proceed from fundamental mathematics to derived semantics.

The plan of attack is to generate `Ground Truth Validation` using `synthetic terrain`  to generate heightmaps with known features.  

<img src="t_flatzone.png" /><img src="t_single_peak.png" /><img src="t_ridge.png" /><img src="t_saddle.png" />

Compare detected output features against ground truth will calibrate our model, using predefined config parameters located `core.py`.  

## Pipeline Architecture

| Layer | Name | Output | Key Parameters | Dependencies |
|:-----:|------|--------|----------------|--------------|
| **0** | **Calibration** | Clean Surface | `h_scale=2.0 m/px`<br/>`v_scale=0.2 m/unit`<br/>`noise_sigma=2.0` | Raw heightmap |
| **1** | **Local Geometry** | Slope (0-90°)<br/>Aspect (0-360°) | `gradient_method=central`<br/>`flat_threshold=0.5°` | Layer 0 |
| **2** | **Regional Geometry** | Mean Curvature H<br/>Gaussian Curvature K<br/>6-Type Classification | `H_eps_min=1e-4`<br/>`K_eps_min=1e-5`<br/>`adaptive=True` | Layer 1 |
| **3** | **Topology** | Peaks ▲<br/>Ridges ━<br/>Valleys ━<br/>Saddles ●<br/>Flat Zones ■ | `peak_prominence=2m`<br/>`ridge_length=6px`<br/>`saddle_K=2e-4`<br/>`flat_slope=3°` | Layer 2 |
| **4** | **Relational** | Visibility Graph<br/>Flow Network<br/>Connectivity Graph | `vis_range=1200m`<br/>`conn_radius=200m`<br/>`flow_step=10px` | Layer 3 |
| **5** | **Semantics** | Defensive Positions<br/>Observation Points<br/>Assembly Areas<br/>Chokepoints | `def_prominence=8m`<br/>`obs_visibility=10`<br/>`assembly_area=2000m²` | Layer 4 |

# Early Testing

<img width="1024" alt="sols_analysis" src="https://github.com/user-attachments/assets/7192dc64-5e85-4498-b084-9c43c29dbf90" />

console:  
```
python run-4.py .\assets\sols.jpg
Loading: .\assets\sols.jpg
[Calibration] Validating input: RawImageInput
[Calibration] Using provided calibration: h_scale=2.0, v_scale=0.2
[Calibration] Building coordinate transform
[Calibration] Using identity coordinate transform
[Calibration] Converting to elevation: shape=(896, 832)
[Calibration] Applying Gaussian blur: sigma=2.0
[Calibration] Validating output surface
[Calibration] Creating Heightmap object
[Geometry] Computing slope/aspect | shape=(896, 832), cell_size=2.0000m
[Geometry] Computing central differences gradient
[Geometry] Validating derivative fields
[Geometry] Flat areas: 36.7% of map, setting aspect=0
[Geometry] Output: slope=[0.00°, 48.58°]
[Regional] Computing curvature fields | shape=(896, 832), cell_size=2.0000m
[Regional] Computing gradients and second derivatives
[Regional] Validating curvature fields
[Regional] Determining epsilon thresholds
[Regional] Using adaptive epsilon [45.0 percentile]
[Regional] Epsilon clamp range MIN: H=5e-05, K=5e-06
[Regional] Epsilon clamp range MAX: H=0.05, K=0.001
[Regional] Adaptive epsilon is set to H=0.001879, K=0.000018
[Regional] Classifying curvature | H_eps=0.001879, K_eps=0.000018
[Regional] Classification results: CONVEX=40565 (5.4%), CONCAVE=57260 (7.7%), SADDLE=127160 (17.1%), FLAT=358146 (48.0%)
[Topological] Starting feature extraction | shape=(896, 832), cell_size=2.0000m
[Topological] Extracting peaks
[Topological] Peak candidates: 183 regional maxima
[Topological] Peaks validated: 80 (convex_ratio threshold=0.05)
[Topological] Peaks extracted: 80
[Topological] Extracting ridges
[Topological] Ridge mask pixels: 121859 (16.3%)
[Topological] Ridge candidates: 195 (min_length=10px)
[Topological] Ridges validated: 195
[Topological] Ridges extracted: 195
[Topological] Extracting valleys
[Topological] Valley mask pixels: 138307 (18.6%)
[Topological] Valley candidates: 187 (min_length=10px)
[Topological] Valleys validated: 187
[Topological] Valleys extracted: 187
[Topological] Extracting saddles
[Topological] Saddle candidates: 125231 from curvature classification
[Topological] Saddle statistics: K min=1.78e-05, K max=1.20e-03, K threshold=2.00e-04, confidence threshold=1.00
[Topological] Saddle parameters: k_threshold=2.00e-04, confidence_threshold=1.00
[Topological] Saddles extracted: 212
[Topological] Extracting flat zones
[Topological] Flat zone mask pixels: 362194 (48.6%)
[Topological] Flat zone candidates: 516 (min_size=450px)
[Topological] Flat zones validated: 30
[Topological] Flat zones extracted: 30
[Topological] Purged 152 features below 8.0m reference
[Topological] sea-level filter: 704 → 552 features (152 removed)
[Topological] Extraction complete | total=552, peaks=80, ridges=195, valleys=187, saddles=212, flat_zones=30
[Topological] Average peak confidence: 1.130
[Relational] Building Visibility Graph...
[Relational] Building Flow Network...
[Relational] Building Connectivity Graph...
[Relational] Delineating Watersheds
[Relational] Identified 146 watershed outlets
[Relational] Visibility: 371 connected nodes, 8129 edges
[Relational] Flow Network: 54 nodes with downstream, 54 edges
[Relational] Connectivity: 545 connected nodes, 1752 edges
[Relational] Watersheds: 146 basins
[Relational] Average basin size: 1.0 features
[Semantics] using template: GameType.CUSTOM
[Semantics] max n-% feature coverage: 0.5
[Semantics] analysing data bundle...
[Semantics] building index...
[Semantics]  -> Defensive Positions
[Semantics]  -> Observation Positions
[Semantics]  -> Chokepoint Positions
[Semantics]  -> Assembly Areas
[Semantics]  -> Cover Positions
[Semantics]  -> Ambush Positions
Saved: sols_analysis.png
```

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


## Scale-Space Framework

Features are scale-dependent. Analysis must consider observation radius:

| Scale | Pixel Range | Interpretation | Application |
|-------|-------------|----------------|-------------|
| Micro | 1-5px | Noise / small rocks | Filter out for vehicle-scale analysis |
| Meso | 10-30px | Gullies / small ridges | Infantry cover, tactical terrain |
| Macro | 50-100px | Mountains / valleys | Vehicle movement, operational planning |


## Detectable Feature Taxonomy

### 1. Elevation Features
- Peaks/summits (local maxima)
- Depressions/pits (local minima)
- Plateaus (flat high areas)
- Plains (flat low areas)
- Relative prominence

### 2. Slope Features
- Steepness magnitude
- Aspect direction
- Cliffs (slope > threshold)
- Gentle slopes (traversable)
- Slope uniformity

### 3. Curvature Features
- Ridges (convex, positive)
- Valleys/gullies (concave, negative)
- Saddles (min/max orthogonal)
- Shoulders/spurs (transitional)

### 4. Topological/Hydrological Features
- Watershed boundaries
- Flow direction networks
- Accumulation zones
- Catchment basins
- Drainage patterns

### 5. Visibility Features
- Viewshed extents
- Intervisibility between points
- Dead ground (hidden areas)
- Dominance relationships

### 6. Accessibility Features
- Traversability zones
- Least-cost paths
- Chokepoints
- Isolation zones


## Inherent Limitations

### Vertical Complexity (The 2.5D Problem)
Heightmaps cannot represent:
- Caves, tunnels, overhangs
- Bridges (only approaches visible)
- Multi-story structures
- Undercuts or cliff faces (only top edge)

### Discrete Objects
Cannot separate base terrain from superimposed features:
- Vegetation (unless encoded as height)
- Buildings/foundations
- Boulders, walls, vehicles
- Bridge decks (vs. terrain below)

### Surface Properties
Geometry provides no material information:
- Texture (grass, rock, sand, snow)
- Traction coefficients
- Soil composition
- Vegetation density
- Water saturation

### Semantic Ambiguity
Identical geometry can have different meanings:
- Flat area = plain, lake, or paved surface?
- Linear depression = stream bed, road cut, or erosion gully?
- Terraced slope = natural or man-made?

### Scale Dependency
Without metadata, absolute scale is unknown:
- 100m hill vs. 1000m mountain?
- Sea level reference?
- Geographic orientation?


## What Heightmaps Can Tell Us (With Certainty)

**Mathematically derivable:**
- "Local maximum at (x, y)"
- "Slope between A and B is X degrees"
- "Linear feature with negative curvature = potential drainage"
- "Viewshed from point P covers Q% of map"
- "Ridge connecting peaks with saddle at S"

**What requires assumptions:**
- "This is a water body" (assumes flat = water)
- "This is a road" (assumes consistent width/gradient)
- "This is a building" (assumes rectangular plateau)
- "Vertical cliff height" (only slope known, not extent)


## Iterative Discovery Principle

Later findings can re-contextualize earlier interpretations:

*Example:* A "flat ridge" (Layer 3) with "perfect linearity" (Layer 4) and "consistent width" (Layer 5) → Reclassified as "man-made terracing" (Semantic)

