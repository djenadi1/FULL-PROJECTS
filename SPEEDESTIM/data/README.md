---
license: cc-by-nc-4.0
pretty_name: City Sample freeway speed
task_categories:
  - object-detection
  - image-segmentation
tags:
  - synthetic
  - traffic
  - vehicle-speed
  - tracking
  - yolo
  - unreal-engine
size_categories:
  - 10K<n<100K
---

# City Sample freeway speed dataset

By Djenadi Mohammed. Download it from [Hugging Face](https://huggingface.co/datasets/Ab1231/city-sample-freeway-speed). Version 2 is also on [Kaggle](https://www.kaggle.com/datasets/mohammeddjenadi/city-sample-freeway-speed).

Synthetic freeway traffic rendered in Unreal Engine 5.8 on the elevated freeway of Epic's City Sample (the Small City map). Every car has its exact speed on every frame, a segmentation mask and a track ID, so it can train a detector, test a tracker and train a speed model from the same footage.

<video src="https://huggingface.co/datasets/Ab1231/city-sample-freeway-speed/resolve/main/showcase/showcase_4k.mp4" controls width="100%"></video>

The video is a 20 second 4K render of the same freeway with the ground truth drawn on. Each car's current speed is worked out from where it is 30 frames before and 30 frames after, next to its maximum and average since it came into view and its true speed. It isn't part of the splits below.

There are two versions. Version 2, in `v2/`, is the one to use. Version 1 is the first set and is described further down.

## Version 2

150 cameras instead of 24, and masks that walls and railings hide properly.

**Updated 6 November 2026:** version 2 was re-rendered so the random paint colours really show. In the first upload they didn't, because City Sample's paint took its colour from a built-in colour table and ignored the one I set. The cameras, traffic and speeds are the same as before, only the images and videos changed.

- 150 clips of 20 seconds at 1920 x 1080 and 30 fps, 90,000 frames in all
- 9,000 labelled images for YOLO segmentation, every 10th frame
- 5,281 cars, from 22 to 150 km/h
- About 9.8 GB

| Split | Clips | Images | Masks | Cars | Camera height | Field of view | Pitch |
|---|---:|---:|---:|---:|---|---|---|
| Train | 110 | 6,600 | 71,316 | 3,845 | 4.0 to 21.3 m | 25 to 80 degrees | -33 to -3 degrees |
| Validation | 20 | 1,200 | 11,267 | 721 | 4.0 to 17.4 m | 28 to 79 degrees | -24 to -3 degrees |
| Test | 20 | 1,200 | 12,841 | 715 | 4.5 to 17.9 m | 30 to 77 degrees | -22 to -4 degrees |

Every clip has its own camera, so a model sees 110 different heights, angles and lenses in training. That's meant to help it work on a new camera without much calibration.

Every vehicle gets a random paint colour, weighted like real traffic: white 22%, black 19%, grey and silver 14% each, blue and red 8% each, and a little yellow, green, beige, orange, teal and light blue. That includes taxis, the police car, vans, buses and trucks, so a model can't learn that a colour means a vehicle type.

A mask now only covers a car where the camera really sees it. Version 1 masks didn't check the city around the cars, so a car behind the divider wall, a railing or a lamp post still got a mask there.

The splits use the same three stretches of freeway as version 1, and the files are laid out the same way under `v2/`. Clip names start with `C2_`, like `C2_Train_017`.

```python
from ultralytics import YOLO

YOLO("yolo26s-seg.pt").train(data="city-sample-freeway-speed/v2/yolo/data.yaml")
```

## Version 1

The first set, in the top folders. It has fewer cameras, Epic's default paint and the mask problem above, so it's mostly here for comparison.

- 24 clips of 60 seconds at 1920 x 1080 and 30 fps, 43,200 frames in all
- 8,640 labelled images for YOLO segmentation, every 5th frame
- 1,923 cars, from 22 to 150 km/h
- About 6 GB

| Split | Clips | Images | Masks | Cars | Speeds |
|---|---:|---:|---:|---:|---|
| Train | 16 | 5,760 | 59,175 | 1,213 | 22 to 150 km/h |
| Validation | 4 | 1,440 | 22,792 | 382 | 23 to 146 km/h |
| Test | 4 | 1,440 | 8,258 | 328 | 38 to 147 km/h |

## Files

```
yolo/
  data.yaml                       classes and split folders, ready for Ultralytics
  images/<split>/<clip>_<frame>.jpg
  labels/<split>/<clip>_<frame>.txt
tracks/<split>/<clip>.json        every car on every frame, with its true speed
videos/<split>/<clip>.mp4         the whole clip
showcase/showcase_4k.mp4          the video above
v2/                               version 2, with the same yolo/, tracks/ and videos/ inside
```

The top folders are version 1.

Clip names look like `Train_04`, and image names add the frame number, so `Train_04_00090.jpg` is frame 90 of that clip. Frame `n` of a tracks file is frame `n` of the clip's video.

## YOLO labels

Each line is one vehicle: the class, then its outline as x, y pairs from 0 to 1.

| Class | Name | Masks in version 2 | Masks in version 1 |
|---:|---|---:|---:|
| 0 | car | 48,089 | 47,949 |
| 1 | van | 13,431 | 13,809 |
| 2 | truck | 24,661 | 20,302 |
| 3 | bus | 9,242 | 8,165 |

The outlines only cover what you can see, so a car half behind a truck gets half an outline. Vehicles under 40 visible pixels have no label. A few images have no vehicles in view and an empty label file.

`data.yaml` has no `path` line, so Ultralytics finds the split folders next to it:

```python
from ultralytics import YOLO

YOLO("yolo26s-seg.pt").train(data="city-sample-freeway-speed/yolo/data.yaml")
```

## Tracks

Each tracks file has the clip's camera at the top and then one list per frame, with a row for every vehicle you can see.

| Field | What it is |
|---|---|
| `id` | The car's ID, the same on every frame of the clip |
| `class` | 0 car, 1 van, 2 truck, 3 bus |
| `kmh` | True speed on this frame |
| `box` | x1, y1, x2, y2 in pixels around the visible part |
| `pixels` | How many pixels of the car you can see |
| `lowest` | x, y of the lowest visible pixel, roughly where the tyres meet the road |
| `road_m` | Centre of the car in Unreal world coordinates, in metres |

`kmh` and `road_m` are the answers. A speed model shouldn't take them as input, because real video doesn't have them.

The camera fields are `x_m`, `y_m`, `z_m`, `yaw`, `pitch`, `fov` and `height_above_road_m`. The field of view is horizontal, the camera has no roll and the principal point is the middle of the image. Unreal world axes are X and Y flat and Z up, and yaw turns from X towards Y.

Speeds come straight from the traffic simulation, which moves each car a set distance every frame, so they're exact.

## How it was made

Traffic runs on 3 lanes each way with the Intelligent Driver Model. Every car has its own wish speed and changes it now and then, so cars speed up, slow down and queue behind slower ones. Each clip has its own pace and density, from slow packed traffic to a nearly empty fast road.

The 13 vehicle models are Epic's City Sample cars, vans, trucks and bus. City Sample's own traffic was turned off, so every vehicle in a frame has a label.

The masks come from a second render pass where each car writes its own ID into the stencil buffer. In version 2 a pixel only counts when the car's depth matches the scene's depth there, so anything in front of the car hides it.

Each clip has one fixed camera, on a pole beside the road or over it, facing either way. Version 2's cameras are in its table above, and these are version 1's:

| Split | Height above road | Pitch | Field of view |
|---|---|---|---|
| Train | 5.9 to 14.4 m | -22 to -6 degrees | 31 to 61 degrees |
| Validation | 7.8 to 13.7 m | -15 to -8 degrees | 47 to 68 degrees |
| Test | 7.6 to 11.8 m | -18 to -10 degrees | 31 to 72 degrees |

## Splits

The freeway is about 1.9 km long. Each split has its own stretch of it, with gaps between them:

| Split | Metres along the freeway | Shape |
|---|---|---|
| Train | 880 to 1,700 | A long straight and a gentle curve |
| Validation | 560 to 740 | A short straight between two bends |
| Test | 150 to 520 | A big curve |

No camera position is shared between splits. A camera can still see far down the road, so parts of another split's freeway sometimes show up small in the background.

## Known limits

In version 1, a car behind the divider wall, a railing or a lamp post still has a mask over that part, because the mask pass only checked cars against other cars. Most of its cameras look straight down the road, so this hits a minority of masks, but the outlines and the `lowest` points of those cars are wrong. Version 2 fixes it.

Everything is from one freeway in one city, at one time of day, with no rain or night.

Version 1 has only 16 training cameras, so a model can partly learn those camera angles instead of what camera height and angle do in general.

Version 1 cars keep Epic's default paint, so most cars of one model look the same, like many silver sedans and yellow taxis. A model trained on it can miss dark cars: an early one missed a black SUV in a real photo. Version 2 has random colours on every vehicle.

The cars are moved along the lanes rather than driven, so their wheels don't spin. There's motion blur on the bodies.

At mask edges JPEG smears the colours, so about 3% of mask pixels were noise and got cleaned up. Outlines can be a pixel or two off at the edges.

All of it is rendered. Treat scores on this data as approximate until they're tested on real traffic video.

## Licence and credits

My work on this dataset (the renders, labels, tracks, traffic and camera setups) is under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). You can use and share it for non-commercial work if you credit me.

**For any commercial use, contact me first.** That covers training models for a product, selling the data or anything made from it, and using it inside a company's work. Open a discussion in this dataset's Community tab and I'll get back to you.

The scenes and vehicles are Epic Games' City Sample, rendered in Unreal Engine.
