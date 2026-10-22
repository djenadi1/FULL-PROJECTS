# App Test Data (PKU-Market-PCB)

My model is trained on small crops, but the app is for full high res boards. So to test the app I use the **PKU-Market-PCB** dataset, which has full PCB images with labeled defects. I didn't make it and I don't train on it, it's only for testing.

- **Paper:** Huang et al., *A PCB Dataset for Defects Detection and Classification*, 2019 ([link](https://arxiv.org/abs/1901.08204))
- **Official download:** [Peking University lab page](https://robotics.pkusz.edu.cn/resources/datasetENG/) (Baidu Pan link)
- **Mirror I used:** [RobotHuman/PCB_defect on Hugging Face](https://huggingface.co/datasets/RobotHuman/PCB_defect)

The images are not uploaded to this repo because they are almost 1 GB. Download them and put them in `PKU-Market-PCB/` here.

## What's inside

- 693 full board images, around 2300 to 3000 px per side (most common sizes are 3034x1586 and 3056x2464)
- 2,953 defects, about 4 per image
- The defects were added with Photoshop on real boards, so they are not real factory defects

```
Data_app_test/PKU-Market-PCB/
├── Missing_hole/       # one folder per defect type
│   ├── missing_hole01.jpg
│   └── missing_hole01_annotation.xml
├── Mouse_bite/
├── Open_circuit/
├── Short/
├── Spur/
└── Spurious_copper/
```

Each image only has defects of the type its folder is named after.

## Labels

The labels are Pascal VOC xml files, not YOLO txt. Every defect is an `<object>` with its name and the box corners in pixels:

```xml
<object>
    <name>spur</name>
    <bndbox>
        <xmin>1444</xmin>
        <ymin>1123</ymin>
        <xmax>1474</xmax>
        <ymax>1156</ymax>
    </bndbox>
</object>
```

## Classes

| Name in xml | Defects | Same as my model |
|---|---|---|
| spurious_copper | 503 | SC |
| missing_hole | 497 | - (not in my training data) |
| mouse_bite | 492 | MB |
| short | 491 | SH |
| spur | 488 | SP |
| open_circuit | 482 | OP |

5 of the 6 types are also in my training data. Missing hole is not, so I skip those images when I measure the score.

## Defect size

A typical defect here is about **64x63 px** (smallest about 24 px, biggest about 283 px). In my training images a defect takes about 10-30% of the image, so tiles of about **256-320 px** should make the defects look the same size the model is used to.
