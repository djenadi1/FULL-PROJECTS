import json
from pathlib import Path

import cv2
import numpy as np
from depth_anything_3.api import DepthAnything3

DATA = Path("../data")
OUT = Path("../DA3_OUT")
OUT.mkdir(exist_ok=True)

model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE").to("cuda").eval()
index = json.loads((DATA / "index.json").read_text())

for number, item in enumerate(index):
    name = item["name"]
    path = OUT / f"{name}_da3.npy"
    if path.exists():
        continue

    camera = json.loads((DATA / "frames" / f"{name}.json").read_text())["camera"]
    output = model.inference([str(DATA / "frames" / f"{name}_rgb.png")]).depth[0]

    # focal length at the size DA3 ran at, then metres = focal * output / 300
    fx = camera["fx"] * output.shape[1] / camera["width"]
    fy = camera["fy"] * output.shape[0] / camera["height"]
    metres = (fx + fy) / 2 * output / 300

    metres = cv2.resize(metres, (camera["width"], camera["height"]), interpolation=cv2.INTER_LINEAR)
    np.save(path, metres.astype(np.float32))

    if number % 100 == 0:
        print(number, "of", len(index), flush=True)

print("saved to", OUT)
