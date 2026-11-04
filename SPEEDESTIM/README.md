# Vehicle speed estimation

**Status:** version 2 of the dataset is rendered and public, with 150 cameras and cars in random colours, and YOLO26m-seg is trained on it, with a validation mask mAP50 of 0.884. The tracker and the speed transformer aren't built yet, and the test split hasn't been used.

**Dataset:** [Ab1231/city-sample-freeway-speed](https://huggingface.co/datasets/Ab1231/city-sample-freeway-speed) on Hugging Face

**Demo:** [20 s 4K video](https://huggingface.co/datasets/Ab1231/city-sample-freeway-speed/resolve/main/showcase/showcase_4k.mp4) with the ground truth drawn on

## Folder layout

| Folder | What's in it |
|---|---|
| `CODE/` | The YOLO training script, and a notebook I've started for the speed transformer |
| `data/` | Where the YOLO part of version 2 of the [dataset](https://huggingface.co/datasets/Ab1231/city-sample-freeway-speed) goes, not stored in git apart from its [README](data/README.md) |
| `speed_data/` | Where the tracks files go, `tracks/` for version 1 and `tracks_v2/` for version 2, not stored in git |

The goal is to measure how fast each car drives past a fixed traffic camera, from the video alone. Real footage with an exact speed for every car on every frame is hard to get, so I rendered freeway traffic in Unreal Engine 5.8 on Epic's City Sample, where every speed is known.

## The plan

| Part | What it is |
|---|---|
| Dataset (done) | Version 2: 150 clips of 20 s at 1080p and 30 fps, each from its own camera, with 5,281 cars from 22 to 150 km/h. Each car has a mask, a track ID and its true speed on every frame |
| Detector (first model done) | YOLO26m-seg, fine-tuned from the pretrained weights on 6,600 labelled frames with 4 classes: car, van, truck and bus |
| Tracker | ByteTrack on the detector's boxes, so each car keeps one ID while it's in view. It isn't trained |
| Speed model | A small transformer over one car's frames from 30 before to 30 after the frame it predicts, plus the camera's height, pitch and field of view |
| Evaluation | km/h error on the test clips, plus each car's maximum and average speed from when it comes into view to when it leaves |

The speed for frame t will use frames up to t+30, so live video will run one second behind. Each frame of a car becomes a few numbers taken from its mask: the lowest visible pixel, where the tyres meet the road, the box around it and the log of its pixel count. The transformer won't see the image itself.

I'll train the first speed model on the true tracks. After that it'll train on what YOLO and ByteTrack actually produce, since that's what it gets on real video.

## The data

Each split has its own stretch of the 1.9 km freeway with its own cameras, so validation and test show roads the model never trained on. These are the numbers for version 2:

| Split | Clips | Images | Masks | Cars | Road |
|---|---:|---:|---:|---:|---|
| Train | 110 | 6,600 | 71,316 | 3,845 | A long straight and a gentle curve |
| Validation | 20 | 1,200 | 11,267 | 721 | A short straight between two bends |
| Test | 20 | 1,200 | 12,840 | 715 | A big curve |

The cameras go from 4 to 21 m above the road, with fields of view from 25 to 80 degrees, on poles beside the road or over it, facing either way. Regular cars get a random paint colour weighted like real traffic, while taxis, the police car, buses and trucks keep their own.

Traffic follows the Intelligent Driver Model on 3 lanes each way, so cars speed up, slow down and queue behind slower ones. The masks come from a second render pass where each car writes its own ID into the stencil buffer, and a pixel only counts when the car's depth matches the scene's depth, so walls and railings hide cars behind them.

Version 1 had 24 clips of 60 s from 16 training cameras, all cars in Epic's default paint, and masks that only checked cars against other cars, so a car behind the divider wall still got a mask there. It's still on Hugging Face for comparison.

The [data README](data/README.md) has the file formats, how it was made and its limits.

## YOLO results

`yolo26m-seg` from its pretrained weights, at an image size of 1280 on version 2, on an RTX 3080. It trained for 2.7 hours, which was 13 epochs at about 13 minutes each, and the best weights are from epoch 12. These are validation scores:

| Score | Box | Mask |
|---|---:|---:|
| Precision | 0.940 | 0.940 |
| Recall | 0.844 | 0.811 |
| mAP50 | 0.911 | 0.884 |
| mAP50-95 | 0.766 | 0.635 |

| Class | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|
| Car | 0.840 | 0.510 |
| Van | 0.891 | 0.625 |
| Truck | 0.910 | 0.686 |
| Bus | 0.897 | 0.718 |

Cars score lowest, and I think that's mostly the small ones far down the road. The scores were still going up over the last few epochs, so a longer run should do better.

Before this I ran a 10 epoch test of `yolo26s-seg` at 992 on version 1. On version 1's validation set it reached a box recall of 0.644 and a mask mAP50 of 0.672, so it missed about a third of the vehicles. The two runs use different validation sets, so the numbers don't compare exactly.

I also tried that first model on one real photo of a street, taken from above at an angle. It found 7 of the 9 vehicles but missed a black SUV right at the front, and called most silver cars trucks. Version 1 had almost no dark cars, which is why version 2 has random paint. I haven't tried the new model on real photos yet.

## Areas of improvement

- **Small cars.** Car is the weakest class, at a mask mAP50-95 of 0.510.
- **Vehicle models.** There are only 13, so a real car that looks like none of them, or a familiar shape in a new colour, can still be missed or put in the wrong class.
- **One place and one light.** Everything is one freeway at one time of day, with no rain or night.
- **Version 1 masks.** The tracks in `tracks/` still have the old masks, so cars behind walls have wrong outlines and lowest points there.
- **Spinning wheels.** The cars are moved along the lanes rather than driven, so their wheels don't turn.
- **Real video.** Every score so far is on renders. A real test needs traffic video with measured speeds, such as BrnoCompSpeed.

## Things I decided

- **Segmentation over boxes.** The lowest pixel of a mask sits on the road, while the bottom of a box can be a bumper or a mirror, and a mask still works when a truck hides half the car.
- **Camera settings as an input.** A single number can't turn pixels into metres for an angled camera, because a pixel far up the image covers much more road than one near the bottom.
- **The generator stays private.** The dataset is public under CC BY-NC 4.0, and commercial use needs my permission first. The generator code isn't in this repo.
