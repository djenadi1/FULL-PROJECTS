---
license: mit
pretty_name: Alpine snow depth
task_categories:
  - depth-estimation
tags:
  - synthetic
  - snow
  - lidar
  - gps
  - unreal-engine
size_categories:
  - 1K<n<10K
---

# Alpine snow depth dataset

By Djenadi Mohammed. Download it from [Hugging Face](https://huggingface.co/datasets/Ab1231/alpine-snow-depth).

Synthetic alpine snow scenes rendered in Unreal Engine 5 with the path tracer, on real Swiss terrain around the Eiger and Mönch. Every image has exact depth, simulated LiDAR and simulated GPS, and the second frame of each pair also has simulated IMU motion.

- 1,556 images at 1280 x 720, in 778 pairs
- About 7.7 GB
- Rendered at 226 samples per pixel with the denoiser on and no motion blur

| Split | Sun | Flat light | Snowfall | Total |
|---|---:|---:|---:|---:|
| Train | 472 | 446 | 328 | 1,246 |
| Validation | 78 | 58 | 44 | 180 |
| Test | 50 | 52 | 28 | 130 |
| **Total** | **600** | **556** | **400** | **1,556** |

## Files

```
index.json                every image with its split, weather and previous frame
frames/
  <name>_rgb.png          the image
  <name>_depth.npy        exact depth
  <name>_lidar.npy        simulated LiDAR points
  <name>.json             camera, position, GPS and motion
```

Names look like `sun_0078_a`: the weather, the pair number, then `a` for the first frame of the pair and `b` for the second.

## Loading it

You only need numpy and pillow.

```python
import json
from pathlib import Path

import numpy as np
from PIL import Image

root = Path("alpine-snow-depth")
index = json.loads((root / "index.json").read_text())
names = [i["name"] for i in index if i["split"] == "train"]

name = names[0]
rgb = np.asarray(Image.open(root / "frames" / f"{name}_rgb.png").convert("RGB"))
depth = np.load(root / "frames" / f"{name}_depth.npy")
meta = json.loads((root / "frames" / f"{name}.json").read_text())

points = np.load(root / "frames" / f"{name}_lidar.npy")
column, row, z = points[:, 0].astype(int), points[:, 1].astype(int), points[:, 2]
lidar = np.zeros_like(depth)
lidar[row, column] = z
```

## Depth

`<name>_depth.npy` is float32 with shape 720 x 1280.

- It's z-depth, the distance along the camera's forward axis, and it isn't the straight-line distance from the camera.
- Sky is 0. Always mask it out with `depth > 0`.
- Values go up to about 49 km, because the scene is 90 km across.
- Falling snow isn't in the depth, so it shows the ground and trees behind the flakes.

To turn a pixel `(u, v)` into a 3D point in camera space (x right, y up, z forward):

```python
x = (u + 0.5 - cx) / fx * z
y = (cy - v - 0.5) / fy * z
```

## Position and GPS

| Field in `<name>.json` | What it is |
|---|---|
| `camera` | fx, fy, cx, cy, width, height and horizontal field of view (60 to 100 degrees) |
| `pose_true` | Exact position and direction |
| `gps_true` | Exact latitude, longitude and altitude |
| `gps_noisy` | The same with simulated error, about 1.5 m sideways and 3 m in height |
| `imu_motion` | Noisy motion since the previous frame, null on first frames |
| `pair_id`, `pair_role`, `previous_frame` | How the pair fits together |

In `pose_true`:

- `east_m` and `north_m` are metres from the scene centre, at Swiss LV95 coordinates (2641000, 1159000).
- `up_m` is metres above the lowest point of the terrain, and `altitude_m` is height above sea level.
- `heading_from_east_deg` is the direction the camera faces, with 0 for east and 90 for north.
- `pitch_deg` is positive looking up. Roll is always 0.

`imu_motion` gives the move in metres and the turn in degrees between the two frames. They are 0.2 to 0.5 seconds apart and about 3 m apart in the typical pair. The simulated error is 1% of the move and 0.1 degrees of turn. It's simple noise with no drift, and it isn't calibrated against a real IMU. The GPS error isn't calibrated against a real receiver either.

## Simulated LiDAR

`<name>_lidar.npy` is float32 with shape N x 3: column, row and z-depth in metres. There is one fixed scan per image, and 753 of the 1,556 files are empty because the terrain is out of range.

Each scan has its own sensor, modelled on the Velodyne Alpha Prime: 32, 64 or 128 scan lines, a 40 degree vertical view, 0.2 degree steps and up to 5 degrees of tilt.

The false snow points and the range lost in snow are measured from real scans in the [Boreas](https://registry.opendata.aws/boreas/) dataset. I used 15 scans from each of 11 drives, but only 2 of those drives had falling snow. In that snow:

- 3.6% of beams returned a flake within 30 m, mostly 10 to 30 m away. This varied from 1.7% to 6% between scans.
- The furthest returns dropped from about 180 m to about 95 m.

These parts are not measured from real data:

- Snow lighter than that one heavy snowfall, and any flake returns beyond 30 m
- How range noise grows with distance, and how many points are lost at shallow angles
- How flakes clump together, since they're placed at random
- Different surface types, since rock, snow and ice are treated the same

A car recorded those Boreas drives on city roads, so I can't say any of this holds for a drone over mountains. Treat LiDAR results from this data as simulation only until they're tested on real LiDAR.

## Weather

| Name | What it looks like |
|---|---|
| `sun` | Clear deep blue sky, hard shadows |
| `flatlight` | Overcast with little contrast on the snow |
| `snowfall` | Overcast with flakes as 3D meshes in the scene, mostly light snow |

Weather changes the lighting and flakes only. The terrain and trees are the same in all three, and each pair has its own camera position.

## Splits

Whole 500 m squares of camera positions go to one split, about 80% train, 10% validation and 10% test. Both frames of a pair always stay in the same split.

That only keeps the camera positions apart. A view can reach many kilometres, so a mountain in a test image can also appear in training images from another angle, and test scores only tell you about new viewpoints in this region.

## Known limits

Everything comes from one 90 km area, so nothing here tests other mountain ranges.

In snowfall, frames with the terrain out of range only contain false flake points.

In 206 images, over 10% of the picture is closer than 10 m, and in 21 it's over 50%. Some of these are a large dark tree trunk right in front of the camera, such as `snowfall_0109_a`. I haven't gone through them all.

Not every frame looks photorealistic. Rock looks too orange in flat light, some horizons are flat, and tall cliffs have dark vertical streaks.

## Licence and credits

My work on this dataset is under the MIT licence, see `LICENSE`. The sources below keep their own terms, so keep these credits if you share it.

| Source | Used for | Terms |
|---|---|---|
| Federal Office of Topography swisstopo, swissALTI3D and SWISSIMAGE | Terrain and aerial photo textures | Source credit required |
| Copernicus WorldDEM-30 | Far terrain | See the notice below |
| [Poly Haven](https://polyhaven.com) | Rock and snow textures, skies | CC0 |
| [Boreas](https://registry.opendata.aws/boreas/) (Burnett et al., 2023) | LiDAR snow measurements only, none of its data is included | CC BY 4.0 |

Produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved. The organisations in charge of the Copernicus programme by law or by delegation do not incur any liability for any use of the Copernicus WorldDEM-30.
