# PCB Defect Data (DsPCBSD+)

The data for this project is the **DsPCBSD+** dataset, a public dataset of PCB surface defects. I didn't make it, I just use it to train my detector.

- **Paper:** Lv et al., *A dataset for deep learning based detection of printed circuit board surface defect*, Scientific Data, 2024 ([link](https://doi.org/10.1038/s41597-024-03656-8))
- **Download:** [figshare](https://doi.org/10.6084/m9.figshare.24970329)

## What's inside

- 10,259 images, all 226x226 JPGs (small crops of PCBs)
- 20,276 defects, each marked with a bounding box
- Split: 8,208 train / 2,051 val

The same images come in two label formats, so you can pick whatever your model needs:

```
Data/DsPCBSD+/
├── Data_COCO/      # one json per split, boxes in pixels [x, y, w, h]
├── Data_YOLO/      # one txt per image: class x_center y_center w h (0 to 1)
└── Hash.py         # finds near duplicate images
```

I use the YOLO version. The `2017` in the COCO folder names is just the COCO naming style, the data is from 2023.

## Classes

| YOLO id | Code | Defect | Train boxes |
|---|---|---|---|
| 0 | SH | Short | 360 |
| 1 | SP | Spur | 2,359 |
| 2 | SC | Spurious copper | 436 |
| 3 | OP | Open | 459 |
| 4 | MB | Mouse bite | 936 |
| 5 | HB | Hole breakout | 1,304 |
| 6 | CS | Conductor scratch | 1,097 |
| 7 | CFO | Conductor foreign object | 750 |
| 8 | BMFO | Base material foreign object | 320 |

Note: the classes are not balanced, there are about 7x more spurs than shorts. In COCO the ids start at 1 instead of 0.
