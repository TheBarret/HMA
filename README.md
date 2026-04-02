# Heightmap Analysis Framework

<img width="1024" src="https://github.com/user-attachments/assets/456c477f-f91c-49bf-a20d-fe05f71a8038" />

## Roadmap

- Layers 0 - 3      : running and testing
- Layer 4           : running and testing
- Layer 5           : to do...
- auto calibration  : running and testing
- LLM-Friendly      : to do...
- visualizer        : running and testing

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

# Early Testing: Relational Analysis

Relations are NOT derived from feature heuristics.  
Relations are derived from continuous mathematical fields, onto which features are mapped.  

1. Compute Pixel-Level Fields (Mathematics)  
   - Flow Direction, Accumulation, Cost Surfaces  
2. Extract Relational Structures (Topology)  
   - Stream Networks, Watershed Boundaries, Visibility Lines  
3. Map Discrete Features (Semantics)  
   - Assign Features to Basins, Build Feature Graphs  

<img width="256" src="https://github.com/user-attachments/assets/8014c6fe-3f7e-46d3-af07-406f2ea4f567" /> <img width="400" src="https://github.com/user-attachments/assets/72fdff65-1975-4b8f-b193-a681ceb17a46" />

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

```
[Calibration] Validating input: RawImageInput
[cache] hashing: c81af214f5ba30ec
[cache] seek: cache\c81af214f5ba30ec.layer0.config.json, exists=False
[Calibration] generating new cache...
[Calibration] Computing heightmap: shape=(500, 500)
[cache] writing: cache\c81af214f5ba30ec.layer0.config.json...
```

### Layer 1: Local Geometry (First Derivatives)
*How does the surface behave at this exact point?*

- **Input:** Calibrated surface
- **Operations:**
  - Calculate gradient magnitude → **Slope** (steepness)
  - Calculate gradient direction → **Aspect** (compass orientation)
- **Rationale:** Foundation for all flow-based analysis. Water flows down gradient; vehicles climb based on slope.
- **Output:** Slope map (0-90°) and Aspect map (0-360°)

```
[Geometry] Computing slope/aspect | shape=(500, 500), cell_size=2.0000m
[cache] hashing: efe8278464805309
[cache] seek: cache\efe8278464805309.layer1.config.json, exists=False
[Geometry] Computing central differences gradient
[Geometry] Validating derivative fields
[Geometry] Flat areas: 6.5% of map, setting aspect=0
[Geometry] Output: slope=[0.00°, 80.34°]
[cache] writing: cache\efe8278464805309.layer1.config.json...
```

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

```
[Regional] Computing curvature fields, shape=(500, 500), cell_size=2.0000m
[cache] hashing: 50bfe831795edbde
[cache] seek: cache\50bfe831795edbde.layer2.config.json, exists=False
[cache] hashing: 50bfe831795edbde
[Regional] Computing gradients and second derivatives
[Regional] Validating curvature fields
[Regional] Determining epsilon thresholds
[Regional] Using adaptive epsilon [62.0 percentile]
[Regional] Epsilon clamp range MIN: H=5e-05, K=2e-05
[Regional] Epsilon clamp range MAX: H=0.05, K=0.001
[Regional] Adaptive epsilon is set to H=0.010258, K=0.000439
[Regional] Classifying curvature | H_eps=0.010258, K_eps=0.000439
[Regional] Classification results: CONVEX=20821 (8.3%), CONCAVE=12539 (5.0%), SADDLE=40009 (16.0%), FLAT=115749 (46.3%)
[cache] writing: cache\50bfe831795edbde.layer2.config.json...
```

### Layer 3: Topological Features (Discrete Objects)
*Where do continuous shapes resolve into identifiable structures?*

- **Input:** Elevation + Curvature maps
- **Operations:**
  - Identify critical points: peaks (local maxima), pits (local minima), saddles
  - Extract linear features: ridgelines, valley bottoms, drainage networks
- **Rationale:** Transform continuous field into discrete entities. A "peak" requires both local maximum and convex surroundings.
- **Output:** Feature set: points (peaks/pits/saddles) and polylines (ridges/channels)

```
[Topological] Starting feature extraction | shape=(500, 500), cell_size=2.0000m
[Topological] Extracting peaks
[Topological] Peak candidates: 78 regional maxima
[Topological] shoulder radius: inner=2, outer=20
[Topological] Peaks validated: 51 (convex_ratio threshold=0.12)
[Topological] Peaks extracted: 51
[Topological] Extracting ridges
[Topological] Ridge mask pixels: 60028 (24.0%)
[Topological] Ridge candidates: 34 (min_length=15px)
[Topological] Ridges validated: 34
[Topological] Ridges extracted: 34
[Topological] Extracting valleys
[Topological] Valley mask pixels: 34214 (13.7%)
[Topological] Valley candidates: 8 (min_length=15px)
[Topological] Valleys validated: 8
[Topological] Valleys extracted: 8
[Topological] Extracting saddles
[Topological] Saddle candidates: 38986 from curvature classification
[Topological]  * final_k=154, final_k_min=4.44e-04, final_k_max=1.11e-01. k_threshold=2.00e-04
[Topological]  * cconfidence_threshold=0.30
[Topological]  * query_ball_point(NMS_radius=30)
[Topological] Thresholds hits: (k_mag < k_threshold)=0, (conf < confidence_threshold)=9
[Topological] Saddles extracted: 145
[Topological] Extracting flat zones
[Topological] Flat zone mask pixels: 44847 (17.9%)
[Topological] Flat zone candidates: 1361 (min_size=150px)
[Topological] Thresholds hits: (size < min_flat_zone_size_px)=1310
[Topological] Flat zones validated: 51
[Topological] Flat zones extracted: 51
[Topological] Sea-level filter: ref=1.5m, 289 → 274 features (15 removed)
[Topological] Building feature hierarchy...
[Topological] Extraction complete | total=274, peaks=51, ridges=34, valleys=8, saddles=145, flat_zones=51
[Topological] Average peak confidence: 0.987
```

### Layer 4: Relational Analysis (Connectivity)
*How do these features interact spatially?*

- **Input:** Topological features + Slope + Aspect
- **Operations:**
  - Calculate viewsheds (visibility from points)
  - Compute flow accumulation (hydrological networks)
  - Build connectivity graphs (ridge networks, drainage hierarchies)
- **Rationale:** Can't determine visibility without observer locations; can't model flow without aspect.
- **Output:** Relational maps and graphs (viewshed polygons, stream networks)

```
[Relational] Config stream threshold: 65px
[Relational] Starting Relational Analysis | features=274
[Relational] Phase 1: Computing Relational Fields...
[Relational]        - Compute flow direction...[d8]
[Relational] Flow direction computed: 249655 pixels routed, 345 outlets
[Relational]        - Accumulation...
[Relational] Flow accumulation computed: max=23875px, mean=78.6px
[Relational]        - Compute cost surface...
[Relational] Cost surface computed: function=tobler, unit=time, range=[0.71, 116.27], mean=4.75
[Relational] Phase 2: Extracting Relational Structures...
[Relational] Stream network: 16983 pixels (6.79%) above threshold=65px
[Relational] Watershed outlets identified: 345
[Relational] Phase 3: Mapping Features to Structures...
[Relational] Basin: basin_1, area_m2=1404.0
[Relational] Basin: basin_82, area_m2=9448.0
[Relational] Assigned 272 features to 62 basins
[Relational] Building flow network: 274 features, snap_distance=10px
[Relational] Flow network: 53 features have downstream connections, 53 total edges
[Relational] Building visibility graph: 230 features, max_range=250.0px (500m)
[Relational] Visibility checking, verified=1, visible_pairs=1
[Relational] Visibility checking, verified=1025, visible_pairs=345
[Relational] Visibility checking, verified=2049, visible_pairs=645
[Relational] Visibility checking, verified=3073, visible_pairs=1039
[Relational] Visibility checking, verified=4097, visible_pairs=1333
[Relational] Visibility checking, verified=5121, visible_pairs=1671
[Relational] Visibility checking, verified=6145, visible_pairs=1923
[Relational] Visibility checking, verified=7169, visible_pairs=1982
[Relational] Visibility checking, verified=8193, visible_pairs=2096
[Relational] Visibility checking, verified=9217, visible_pairs=2261
[Relational] Visibility checking, verified=10241, visible_pairs=2443
[Relational] Visibility checking, verified=11265, visible_pairs=2635
[Relational] Visibility checking, verified=12289, visible_pairs=2867
[Relational] Visibility checking, verified=13313, visible_pairs=3035
[Relational] Visibility graph complete: 14014 checks, 3205 visible pairs
[Relational] Building connectivity graph: 274 features, radius=25.0px, max_cost=1000.0, buffer=1.5
[Relational]     #1 [22cb0590-3d57-44b8-82ce-2585f9bfbd50] pairs=3
[Relational]     #65 [0f453923-43e4-4c4f-a3ef-8945284aaacc] pairs=168
[Relational]     #257 [def2d050-d63f-4d3e-babd-96d86bcc72a6] pairs=238
[Relational] Connectivity graph complete: 207/274 features connected, 245 edges, 245 pairs evaluated
```

### Layer 5: Semantic Interpretation (Meaning) (WIP)
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
