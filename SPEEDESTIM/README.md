# Vehicle speed estimation

**Status:** the dataset is rendered and public, and a first 10 epoch YOLO26-seg test run is done. The tracker and the speed transformer aren't built yet, and the test split hasn't been used.

**Dataset:** [Ab1231/city-sample-freeway-speed](https://huggingface.co/datasets/Ab1231/city-sample-freeway-speed) on Hugging Face

**Demo:** [20 s 4K video](https://huggingface.co/datasets/Ab1231/city-sample-freeway-speed/resolve/main/showcase/showcase_4k.mp4) with the ground truth drawn on

## Folder layout

| Folder | What's in it |
|---|---|
| `CODE/` | The YOLO training script, and a notebook I've started for the speed transformer |
| `data/` | Where the YOLO part of the [dataset](https://huggingface.co/datasets/Ab1231/city-sample-freeway-speed) goes, not stored in git apart from its [README](data/README.md) |
| `speed_data/` | Where the tracks files go, not stored in git |

The goal is to measure how fast each car drives past a fixed traffic camera, from the video alone. Real footage with an exact speed for every car on every frame is hard to get, so I rendered freeway traffic in Unreal Engine 5.8 on Epic's City Sample, where every speed is known.

## The plan

| Part | What it is |
|---|---|
| Dataset (done) | 24 clips of 60 s at 1080p and 30 fps, 1,923 cars from 22 to 150 km/h. Each car has a mask, a track ID and its true speed on every frame |
| Detector | YOLO26-seg, fine-tuned from the pretrained weights on 8,640 labelled frames with 4 classes: car, van, truck and bus |
| Tracker | ByteTrack on the detector's boxes, so each car keeps one ID while it's in view. It isn't trained |
| Speed model | A small transformer over one car's frames from 30 before to 30 after the frame it predicts, plus the camera's height, pitch and field of view |
| Evaluation | km/h error on the test clips, plus each car's maximum and average speed from when it comes into view to when it leaves |

The speed for frame t will use frames up to t+30, so live video will run one second behind. Each frame of a car becomes a few numbers taken from its mask: the lowest visible pixel, where the tyres meet the road, the box around it and the log of its pixel count. The transformer won't see the image itself.

I'll train the first speed model on the true tracks. After that it'll train on what YOLO and ByteTrack actually produce, since that's what it gets on real video.

## The data

Each split has its own stretch of the 1.9 km freeway with its own cameras, so validation and test show roads the model never trained on:

| Split | Clips | Images | Masks | Cars | Road |
|---|---:|---:|---:|---:|---|
| Train | 16 | 5,760 | 59,175 | 1,213 | A long straight and a gentle curve |
| Validation | 4 | 1,440 | 22,792 | 382 | A short straight between two bends |
| Test | 4 | 1,440 | 8,258 | 328 | A big curve |

Traffic follows the Intelligent Driver Model on 3 lanes each way, so cars speed up, slow down and queue behind slower ones. The masks come from a second render pass where each car writes its own ID into the stencil buffer, so they only cover what you can see.

The [data README](data/README.md) has the file formats, how it was made and its limits.

## First YOLO results

A 10 epoch test run of `yolo26s-seg` at an image size of 992, on an RTX 3080, at about 5.5 minutes per epoch. These are validation scores:

| Score | Box | Mask |
|---|---:|---:|
| Precision | 0.915 | |
| Recall | 0.644 | |
| mAP50 | 0.699 | 0.672 |
| mAP50-95 | 0.548 | 0.446 |

It was still improving at epoch 10. When it finds a vehicle it's usually right, but it misses about a third of them, and I think most of those are the small cars far down the road. The next run is at an image size of 1280 with more epochs.

## Areas of improvement

- **Camera variety.** 16 training cameras is few for a model meant to learn what camera height and angle do in general.
- **Car looks.** The cars keep Epic's default paint, so most cars of one model look the same. The generator can randomise the paint for future renders.
- **One place and one light.** Everything is one freeway at one time of day, with no rain or night.
- **Spinning wheels.** The cars are moved along the lanes rather than driven, so their wheels don't turn.
- **Real video.** Every score so far is on renders. A real test needs traffic video with measured speeds, such as BrnoCompSpeed.

## Things I decided

- **Segmentation over boxes.** The lowest pixel of a mask sits on the road, while the bottom of a box can be a bumper or a mirror, and a mask still works when a truck hides half the car.
- **Camera settings as an input.** A single number can't turn pixels into metres for an angled camera, because a pixel far up the image covers much more road than one near the bottom.
- **The generator stays private.** The dataset is public under CC BY-NC 4.0, and commercial use needs my permission first. The generator code isn't in this repo.
