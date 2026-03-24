# Heightmap Analysis Framework (WIP)

## Core Philosophy
A heightmap is a 2.5D scalar field: perfect geometric surface information, zero information about what lies above, 
below, or what the surface means. Analysis must proceed from fundamental mathematics to derived semantics.


##  `Mean Curvature (H)` = "How bent is the surface?"  

A `Large positive Surface` that bulges `outward`; Hilltop, ridge crest, nose of slope  
A `Large negative Surface` that bulges `inward`; Valley bottom, gully, depression  
Or `Near zero Surface` is `flat` or a  `saddle` (needs K to distinguish); Flat plain, planar slope, cylindrical ridge  

## `Gaussian Curvature (K)` = "What shape is the bend?"  
A `Positive` same sign in both directions is a Dome (peak), bowl (depression)  
A `Negative` opposite signs is a Saddle, pass, twisted surface  
Or a `Near zero` one direction flat can be a Ridge (curved across, flat along), valley  

## Dependency Chain

The plan of attack is to generate `Ground Truth Validation`  
Using `synthetic terrain` to generate heightmaps with known features.  
Compare detected output features against ground truth.  

```
Calibration → First Derivatives → Second Derivatives → Topology  →  Relations → Semantics
-------------------------------------------------------------------------------------------
     ↓              ↓                   ↓                 ↓             ↓           ↓
  Clean   →     Slope/Aspect    →    Curvature     →   Features  →  Networks  →  Meaning
```

## Debugging the pipeline

<img width="512" alt="debug-preview" src="https://github.com/user-attachments/assets/367d574e-4c44-42b0-b11c-0c69086d1970" />


## Demonstration Input

<img width="1024" alt="heightmap" src="https://github.com/user-attachments/assets/389fef25-d30a-423a-9a26-efbcce78c48e" />

## Output

<img width="1024" alt="unified_analysis" src="https://github.com/user-attachments/assets/736473f6-e60a-45f5-8c07-f68304f7f76f" />


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

