# Heightmap Analysis Framework

<img width="1024" src="https://github.com/user-attachments/assets/456c477f-f91c-49bf-a20d-fe05f71a8038" />

## Codebase

- core.py **Foundation and configuration parameters**  
- calibration.py	**Layer 0 | Noise reduction, Pre filter**  
- lgeometry.py		**Layer 1 | Slope and Aspect mapping**  
- rgeometry.py		**Layer 2 | Curvature and Elevation mapping**  
- topology.py		**Layer 3 | Primitives mapping**  
- relational.py	**Layer 4 | Spatial mapping**  
- semantics.py		**Layer 5 | Extensions**  
- shell.py			**Interactive shell**  
- context.py		**Terrain data [spatial query]**  
- tools.py			**Test tools**  
- factory.py		**Test tools**  

## Philosophy
A heightmap is a 2.5D scalar field: perfect geometric surface information, zero information about what lies above, 
below, or what the surface means. Analysis must proceed from fundamental mathematics to derived semantics.
Relations are not derived from feature heuristics but from continuous mathematical fields, onto which features are mapped.  

Calibrating is done by generate `Ground Truth Validation`, using `synthetic terrain`  to generate heightmaps with known features.  

<img src="t_flatzone.png" /> <img src="t_single_peak.png" /> <img src="t_ridge.png" /> <img src="t_saddle.png" />

<img width="128" height="1731" alt="t_flatzone_topology" src="https://github.com/user-attachments/assets/61be8b94-f567-4c67-821e-10a4b2d8cb7a" />
<img width="128" height="1731" alt="t_single_peak_topology" src="https://github.com/user-attachments/assets/38776da8-8c99-456b-b943-d2bf749e1cb9" />
<img width="128" height="1731" alt="t_ridge_topology" src="https://github.com/user-attachments/assets/f424c85a-4127-40dc-8e1a-3351b880d0b5" />
<img width="128" height="1731" alt="t_saddle_topology" src="https://github.com/user-attachments/assets/9e429b04-2fd5-45a4-8a1a-e78d2dc2a996" />

Compare detected output features against ground truth will calibrate our model, using predefined config parameters located `core.py`.  

## To do

- Optimize
- Debug
- Refactor

## Interactive Shell

Start the pipeline with an input (see run.py), this will print all debug symbols per layer.  

```
python run.py
[Calibration] Validating input: RawImageInput
[Calibration] cache exists: c81af214f5ba30ec
[Geometry] Computing slope/aspect | shape=(500, 500), cell_size=2.0000m
[Regional] Computing curvature fields, shape=(500, 500), cell_size=2.0000m
[Topological] Starting feature extraction | shape=(500, 500), cell_size=2.0000m
[Topological] Extracting peaks
[Topological] Peak candidates: 78 regional maxima
[Topological] Shoulder radius: inner=2, outer=20
[Topological] Thresholds hits: (len(peak_px) < min_feature_size)=0, (len(shoulder_types) < peak_min_shoulder_samples)=0
[Topological] Thresholds hits: (convex_ratio > peak_shoulder_convex_ratio)=8, (prominence > peak_min_prominence_m)=19
[Topological] Peaks validated: 51 (convex_ratio threshold=0.08)
[Topological] Peaks extracted: 51
[Topological] Extracting ridges
[Topological] Ridge mask pixels: 60028 (24.0%)
[Topological] Ridge candidates: 34 (min_length=15px)
[Topological] Thresholds hits: (len(xs) < min_ridge_length_px)=0
[Topological] Ridges validated: 34, smoothing_window=5
[Topological] Ridges extracted: 34
[Topological] Extracting valleys
[Topological] Valley mask pixels: 34214 (13.7%)
[Topological] Valley candidates: 8 (min_length=15px)
[Topological] Thresholds hits: (len(xs) < min_ridge_length_px)=0
[Topological] Valleys validated: 8
[Topological] Valleys extracted: 8
[Topological] Extracting saddles
[Topological] Saddle candidates: 38986 from curvature classification
[Topological]  * final_k=154, final_k_min=4.44e-04, final_k_max=1.11e-01. k_threshold=2.00e-04
[Topological]  * confidence_threshold=0.50
[Topological]  * query_ball_point(NMS_radius=30)
[Topological] Thresholds hits: (k_mag < k_threshold)=0, (conf < confidence_threshold)=29
[Topological] Saddles extracted: 125
[Topological] Extracting flat zones
[Topological] Flat zone mask pixels: 65521 (26.2%)
[Topological] Flat zone candidates: 818 (min_size=80px)
[Topological] Thresholds hits: (size < min_flat_zone_size_px)=797
[Topological] Flat zones validated: 21
[Topological] Flat zones extracted: 21
[Topological] Sea-level filter: ref=1.5m, 239 → 236 features (3 removed)
[Topological] Building feature hierarchy...
[Topological] Discovered 2 peak(s)...
[Topological] Discovered 1 peak(s)...
[Topological] Discovered 1 peak(s)...
[Topological] Discovered 1 peak(s)...
[Topological] Extraction complete | total=236, peaks=51, ridges=34, valleys=8, saddles=125, flat_zones=21
[Topological] Average peak confidence: 0.987
[Relational] Config stream threshold: 120px
[Relational] Starting Relational Analysis | features=236
[Relational] Phase 1: Computing Relational Fields...
[Relational]        - Compute flow direction...[d8]
[Relational] Flow direction computed: 249655 pixels routed, 345 outlets
[Relational]        - Accumulation...
[Relational] Flow accumulation computed: max=23875px, mean=78.6px
[Relational]        - Compute cost surface...
[Relational] Cost surface computed: function=tobler, unit=time, range=[0.71, 116.27], mean=4.75
[Relational] Phase 2: Extracting Relational Structures...
[Relational] Stream network: 12471 pixels (4.99%) above threshold=120px
[Relational] Watershed outlets identified: 345
[Relational] Phase 3: Mapping Features to Structures...
[Relational] Basins: 62 found with a total of 989744.0m²
[Relational] Assigned 235 features to 62 basins
[Relational] Building flow network: 236 features, snap_distance=20px
[Relational] Flow network: 53 features have downstream connections, 53 total edges
[Relational] Building visibility graph: 210 features, max_range=250.0px (500m)
[Relational]     verified=8192, visible_pairs=2002 ...
[Relational] Visibility graph complete: 11844 checks, 2721 visible pairs
[Relational] Building connectivity graph: 236 features, radius=20.0px, max_cost=1000.0, buffer=1.5
[Relational]     #8 [6a0056e0-2cf6-4abf-9098-ae5c5ab9f363] pairs=17
[Relational]     #16 [f26bcade-d2e7-41f4-b0ec-eef4ac8deb12] pairs=32
[Relational]     #32 [6cb820ac-71d6-4a4f-96bc-ac958ec8bab1] pairs=68
[Relational]     #40 [bfc97de8-f59e-4865-a9cf-5bf149892b80] pairs=79
[Relational]     #48 [e92ec0ea-211e-4ec4-a1dd-10af6478ee05] pairs=97
[Relational]     #64 [25e4a26c-62ac-46b6-abf2-035476dc1322] pairs=116
[Relational]     #72 [533f3cef-c123-4562-bc98-ec25236cc1a9] pairs=129
[Relational]     #80 [73913e2a-6cd3-4df6-bc4c-44348da86254] pairs=139
[Relational]     #96 [1dcb89e6-141d-4884-9aca-73cfd0c15048] pairs=153
[Relational]     #104 [18931ca8-98d2-4f67-a860-eff9cc905fcc] pairs=154
[Relational] Connectivity graph complete: 157/236 features connected, 158 edges, 158 pairs evaluated
```

And finally launches into an interactive shell.
```
[Relational] Connectivity graph complete: 157/236 features connected, 158 edges, 158 pairs evaluated

  terrain shell  —  terrain
  236 features in memory
  type 'help' for commands, 'exit' to quit

[terrain]: stat

  terrain stat
  grid                   500x500 px  (1000x1000m)
  cell size              2.00 m/px
  elevation range        0.1m — 123.9m
  relief                 123.8m

  peaks                  51
  ridges                 34
  valleys                8
  saddles                125
  flat zones             18

  visibility pairs       2721
  flow edges             53
  watersheds             46
  connectivity edges     158

[terrain]: top  peak prominence 5

top 5  peak  by prominence  [5]
    1  397982db  prominent peak
       prominent peak rising 86m above surrounding terrain with steep slopes
    2  9ec1ef5d  prominent peak
       prominent peak rising 84m above surrounding terrain with steep slopes
    3  c3c113bd  prominent peak
       prominent peak rising 77m above surrounding terrain with steep slopes
    4  cf106e13  prominent peak
       prominent peak rising 73m above surrounding terrain with very steep slopes
    5  f131d1f6  prominent peak
       prominent peak rising 68m above surrounding terrain with moderate slopes

[terrain]:
```

## License

Open source, for the world to use how they see fit with the promise it be legal.
