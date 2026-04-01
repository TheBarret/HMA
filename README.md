# Heightmap Analysis Framework (WIP)

## Core Philosophy
A heightmap is a 2.5D scalar field: perfect geometric surface information, zero information about what lies above, 
below, or what the surface means. Analysis must proceed from fundamental mathematics to derived semantics.

The pipeline architecture and calibrating is done by generate `Ground Truth Validation`,  
using `synthetic terrain`  to generate heightmaps with known features.  

<img src="t_flatzone.png" /> <img src="t_single_peak.png" /> <img src="t_ridge.png" /> <img src="t_saddle.png" />

<img width="128" height="1731" alt="t_flatzone_topology" src="https://github.com/user-attachments/assets/61be8b94-f567-4c67-821e-10a4b2d8cb7a" />
<img width="128" height="1731" alt="t_single_peak_topology" src="https://github.com/user-attachments/assets/38776da8-8c99-456b-b943-d2bf749e1cb9" />
<img width="128" height="1731" alt="t_ridge_topology" src="https://github.com/user-attachments/assets/f424c85a-4127-40dc-8e1a-3351b880d0b5" />
<img width="128" height="1731" alt="t_saddle_topology" src="https://github.com/user-attachments/assets/9e429b04-2fd5-45a4-8a1a-e78d2dc2a996" />

Compare detected output features against ground truth will calibrate our model, using predefined config parameters located `core.py`.  

<img width="1024" src="https://github.com/user-attachments/assets/6a75743e-2ca6-422e-9423-c49e607d0f88" />

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
  
<img width="1024" src="https://github.com/user-attachments/assets/35bf49e0-cbe8-4f0e-96cb-176c2bf7c478" />  


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
