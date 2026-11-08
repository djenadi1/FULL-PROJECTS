# Vehicle speed estimation

**Status:** version 2 of the dataset is rendered and public, with 150 cameras. YOLO26m-seg is trained on it, with a validation mask mAP50 of 0.884, and the first speed transformer is trained on the true tracks. With a high camera it's 2.75 km/h off on average for cars close to the camera, and 7.32 km/h over every camera and distance. The tracker isn't built yet, and the test split hasn't been used.

**Dataset:** [Ab1231/city-sample-freeway-speed](https://huggingface.co/datasets/Ab1231/city-sample-freeway-speed) on Hugging Face, and version 2 is also on [Kaggle](https://www.kaggle.com/datasets/mohammeddjenadi/city-sample-freeway-speed)

**Demo:** [20 s 4K video](https://huggingface.co/datasets/Ab1231/city-sample-freeway-speed/resolve/main/showcase/showcase_4k.mp4) with the ground truth drawn on

## Folder layout

| Folder | What's in it |
|---|---|
| `CODE/` | The YOLO training script, the speed transformer notebook (data, model, training and scores), and `calibrate_cameras.py`, which tests a scale factor per camera |
| `data/` | Where the YOLO part of version 2 of the [dataset](https://huggingface.co/datasets/Ab1231/city-sample-freeway-speed) goes, not stored in git apart from its [README](data/README.md) |
| `speed_data/` | Where the tracks files go, `tracks/` for version 1 and `tracks_v2/` for version 2, not stored in git |

The goal is to measure how fast each car drives past a fixed traffic camera, from the video alone. Real footage with an exact speed for every car on every frame is hard to get, so I rendered freeway traffic in Unreal Engine 5.8 on Epic's City Sample, where every speed is known.

## The plan

| Part | What it is |
|---|---|
| Dataset (done) | Version 2: 150 clips of 20 s at 1080p and 30 fps, each from its own camera, with 5,281 cars from 22 to 150 km/h. Each car has a mask, a track ID and its true speed on every frame |
| Detector (first model done) | YOLO26m-seg, fine-tuned from the pretrained weights on 6,600 labelled frames with 4 classes: car, van, truck and bus |
| Tracker | ByteTrack on the detector's boxes, so each car keeps one ID while it's in view. It isn't trained |
| Speed model (first model done) | A small transformer over one car's frames from 30 before to 30 after the frame it predicts, plus the camera's height, pitch and field of view |
| Evaluation | km/h error on the test clips, plus each car's maximum and average speed from when it comes into view to when it leaves |

The speed for frame t uses frames up to t+30, so live video runs one second behind. Each frame of a car becomes a few numbers taken from its mask: the lowest visible pixel, where the tyres meet the road, the box around it and the log of its pixel count. The transformer doesn't see the image itself.

The first speed model trained on the true tracks. Next it'll train on what YOLO and ByteTrack actually produce, since that's what it gets on real video.

## The data

Each split has its own stretch of the 1.9 km freeway with its own cameras, so validation and test show roads the model never trained on. These are the numbers for version 2:

| Split | Clips | Images | Masks | Cars | Road |
|---|---:|---:|---:|---:|---|
| Train | 110 | 6,600 | 71,316 | 3,845 | A long straight and a gentle curve |
| Validation | 20 | 1,200 | 11,267 | 721 | A short straight between two bends |
| Test | 20 | 1,200 | 12,841 | 715 | A big curve |

The cameras go from 4 to 21 m above the road, with fields of view from 25 to 80 degrees, on poles beside the road or over it, facing either way.

Every vehicle has a random paint colour, weighted like real traffic, and that includes taxis, vans, buses and trucks, so a colour can't give the class away. The first upload of version 2 didn't really have them: City Sample's paint read its colour from a built-in colour table, which overrode mine. That's fixed, and on 6 November 2026 version 2 was re-rendered with the same cameras, traffic and speeds.

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

I also tried both models on one real photo of a street, taken from above at an angle. The first one found 7 of the 9 vehicles but missed a black SUV right at the front, and called most silver cars trucks. The new one was more sure of itself, but it still missed the black SUV and still called a silver minivan a truck. Both datasets had almost no dark cars, and the only silver boxy vehicles in them are a pickup and a parcel truck, so I think colour was doing a lot of the work in both mistakes.

### After the colour fix

I fine-tuned the model above for 1 hour (5 epochs) on the re-rendered version 2, where every vehicle has a random colour, at a lower learning rate of 0.002. Both models were then scored on the coloured validation and test sets:

| Score | Before, val | After, val | Before, test | After, test |
|---|---:|---:|---:|---:|
| Box precision | 0.825 | 0.916 | 0.774 | 0.895 |
| Box recall | 0.724 | 0.789 | 0.623 | 0.681 |
| Mask mAP50 | 0.751 | 0.846 | 0.656 | 0.746 |
| Mask mAP50-95 | 0.519 | 0.592 | 0.440 | 0.509 |

The model trained on default paint drops a lot once the colours change, from 0.884 to 0.751 mask mAP50 on validation, which shows it had partly learned colours instead of shapes. After one hour on coloured cars it gets most of that back, and it was still improving when it stopped.

On the real photo it shows the same thing. Before, a blue hatchback and a teal car were called buses and a purple car a van, because in the old data those colours only came on buses and vans. After, all three are cars. It still misses the black SUV, and the silver minivans are still trucks, so shape and the small number of vehicle models matter here too. The weights are in `runs/segment/v2_colour_1h`.

### A larger model for 10 hours

Since the 1 hour run was still improving, I trained a new one for 10 hours. Four things changed from the 1 hour run:

1. The model is `yolo26l-seg` (31.5 million parameters), one size up from `yolo26m-seg`.
2. It started from the COCO weights, so it didn't build on my version 2 model.
3. I turned the hue and saturation augmentation right up, to `hsv_h=0.5` and `hsv_s=0.9`. The defaults are 0.015 and 0.7, which barely change a car's colour. At 0.5 a red car can come out green or blue, so colour gives the model nothing to go on. That's the shortcut that broke version 1.
4. It trained for 41 epochs in 10 hours, with a learning rate of 0.01 on a cosine schedule and a batch size of 2. I timed a few batch sizes first. Batch 3 needs 10.8 GB and batch 4 needs 14.5 GB, so both spill past the 10 GB on the RTX 3080 and run 8 to 30 times slower. Batch 2 uses 7.6 GB and does about 9 images a second, about 13 minutes an epoch.

Unlike the 1 hour run, this one did level off. Mask mAP50 sat at 0.873 to 0.874 from about epoch 29 on.

Both models scored with the same settings on the coloured sets:

| Score | 1 h M, val | 10 h L, val | 1 h M, test | 10 h L, test |
|---|---:|---:|---:|---:|
| Box precision | 0.917 | 0.944 | 0.895 | 0.902 |
| Box recall | 0.788 | 0.842 | 0.681 | 0.723 |
| Mask mAP50 | 0.847 | 0.875 | 0.746 | 0.768 |
| Mask mAP50-95 | 0.592 | 0.625 | 0.509 | 0.528 |

The bigger difference is on real footage it never saw, two motorway clips from Wikimedia Commons, tracked with the same tuned ByteTrack settings (`CODE/tracker.yaml`):

| | 1 h M, I-495 | 10 h L, I-495 | 1 h M, A40 | 10 h L, A40 |
|---|---:|---:|---:|---:|
| Tracked vehicles per frame | 10.5 | 19.4 | 7.5 | 13.1 |
| Track IDs lasting 15+ frames | 111 | 124 | 37 | 46 |
| Median track length (frames) | 81 | 134 | 124 | 173 |

On the A40 the new model finds the red lorry, the black car behind it and the traffic further down the road, which the 1 hour model missed completely. On the I-495 it calls a grey minivan a car, where the old one said truck. Both still call a silver SUV and a pickup trucks.

The clips have no labels, so these are counts of tracked boxes with a confidence of at least 0.25, not accuracy. Also, since four things changed at once, I can't say how much came from each. The gain on the synthetic test set is small (+0.022 mask mAP50) next to the gain on real video, so my guess is the colour shifts did more than the bigger model. These weights are in `runs/segment/v2_colour_l_10h`, and the Streamlit app now uses them (`CODE/models/yolo_colour.pt`). A bigger model means the app takes longer per video on its CPU.

## Speed model results

These come from a 4 layer encoder-only transformer with 804,000 parameters. It trained on the true tracks of version 2 for 30 epochs at a batch size of 512, which took about 7 minutes on an RTX 3080. The score is the mean absolute error at the frame the model predicts, on the validation cameras, which sit on a stretch of road the model never trained on.

The model does best when the camera is high and the car is close, which is the setup a real speed camera would use:

| Validation windows | Mean error |
|---|---:|
| **Cameras over 12 m high, cars over 1,500 pixels in the picture** | **2.75 km/h** |
| Cameras over 12 m high, every car | 3.94 km/h |
| Every camera, cars over 1,500 pixels | 4.75 km/h |
| Every camera, every car | 7.32 km/h |
| Always guessing the average speed of 67.9 km/h | 21.00 km/h |

Over every camera and distance, it's 7.19 km/h off for cars, 6.67 for vans, 7.58 for trucks and 8.12 for buses.

### What I noticed

**Low cameras are the problem, and not because the model can't handle them.** On its own training cameras it does about as well whether the camera is low or high, but it doesn't carry low cameras over to new ones:

| Camera height | Training cameras | Validation cameras |
|---|---:|---:|
| Under 7 m | 3.19 km/h | 11.78 km/h |
| 7 to 12 m | 2.80 km/h | 5.94 km/h |
| Over 12 m | 2.32 km/h | 3.94 km/h |

I think it learned the 18 low training cameras one by one instead of learning what camera height and pitch do. From a low camera looking along the road, a pixel near the top of the picture covers many metres, so small changes in the camera change a lot. On top of that, 8 of the 20 validation cameras are under 7 m, against 18 of the 110 training ones, so the validation set is harder than the training mix.

**Distance matters even more.** Far cars are only a few pixels tall, and a 1 pixel mistake there is several metres:

| Car size at the frame it predicts | Cameras under 7 m | Cameras from 7 m up |
|---|---:|---:|
| Under 300 pixels | 21.78 km/h | 7.52 km/h |
| 300 to 1,500 pixels | 13.31 km/h | 5.50 km/h |
| 1,500 to 6,000 pixels | 8.35 km/h | 3.69 km/h |
| Over 6,000 pixels | 5.63 km/h | 3.60 km/h |

Cars partly hidden behind other cars weren't worse than the others, so hiding isn't what drives the error here. Distance and camera height are.

**One scale factor per camera doesn't fix it.** `calibrate_cameras.py` multiplies each validation camera's speeds by the factor between 0.70 and 1.30 that fits best. With the best factor for every camera, which a real camera could never know, the error drops from 7.32 to 5.44 km/h. With a factor taken from one car of known speed, like driving a test car past a new camera, it only drops to 6.74. A single factor fixes a camera where the model is too fast or too slow everywhere, but a low camera's error depends on where the car is in the picture, and one number can't follow that.

**Two settings I tried.** In 8 epoch runs, batch sizes of 128, 256 and 512 all ended near 8.1 to 8.4 km/h, while 1024 and 2048 got worse, at 9.06 and 11.34, without training much faster. A dropout of 0.3 instead of 0.1 ended at 8.92 against 7.69. That run was stopped early at epoch 15, before the learning rate came down, so it isn't a perfectly fair comparison, but it was also behind at the same epochs, so I don't think fitting the training data too closely is the main problem.

### What comes next

- **Only count speeds in the accurate zone.** The next version will only start measuring a car once it's close enough, like the measuring zone of a real speed camera. Over 1,500 pixels the error falls from 7.32 to 4.75 km/h over every camera, and to 2.75 km/h with a high camera. Each car's maximum and average speed will then come from that zone only.
- **Position on the road in metres.** From the camera's height, pitch and field of view, the lowest point of each car can be turned into a position on the road. Then the model doesn't have to learn the perspective of every new camera from the pixels.
- **More low cameras** in the next render, so the model sees many more of them.
- **YOLO and ByteTrack as input** instead of the true tracks, since that's what the model gets on real video.
- **A real test.** I couldn't find real traffic video with measured speeds from a camera like mine that's free to download. BrnoCompSpeed, which uses laser gates, and UTFPR, which uses loops in the road, both need a request to their authors, so I'll ask for them.

## Areas of improvement

- **Small cars.** Car is the weakest class for YOLO, at a mask mAP50-95 of 0.510, and far cars are the weakest for the speed model too.
- **Car colours.** Version 2 now has random colours, but YOLO still missed a black SUV in a real photo after one hour of training on it. A longer run should help.
- **Low cameras.** The speed model is 11.78 km/h off on new cameras under 7 m, against 3.94 km/h on new cameras over 12 m.
- **Vehicle models.** There are only 13, so a real car that looks like none of them, or a familiar shape in a new colour, can still be missed or put in the wrong class.
- **One place and one light.** Everything is one freeway at one time of day, with no rain or night.
- **Version 1 masks.** The tracks in `tracks/` still have the old masks, so cars behind walls have wrong outlines and lowest points there.
- **Spinning wheels.** The cars are moved along the lanes rather than driven, so their wheels don't turn.
- **Real video.** Every speed score so far is on renders, with the true tracks as input. A real test needs traffic video with measured speeds, such as BrnoCompSpeed.

## Things I decided

- **Segmentation over boxes.** The lowest pixel of a mask sits on the road, while the bottom of a box can be a bumper or a mirror, and a mask still works when a truck hides half the car.
- **Camera settings as an input.** A single number can't turn pixels into metres for an angled camera, because a pixel far up the image covers much more road than one near the bottom.
- **The generator stays private.** The dataset is public under CC BY-NC 4.0, and commercial use needs my permission first. The generator code isn't in this repo.
