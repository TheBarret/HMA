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

Project is not finished, progressively keep adding and fixing to it.

## Interactive Shell

Start the pipeline with an input (see run.py), this will print all debug symbols per layer.  

<img width="1024" src="https://github.com/user-attachments/assets/871a75bc-2b44-423a-b1fd-e9aaf894053b" />

<img width="1024" src="https://github.com/user-attachments/assets/68ffb875-cbaa-460f-9a23-1f6596f7ef18" />


## License

Open source, for the world to use how they see fit with the promise it be legal.
