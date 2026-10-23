# PCBD/app.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

tile = 320     # same tile size as in pku_to_yolo.py and train.py
overlap = 64   # tiles overlap a bit so a defect on the border is not cut in half
step = tile - overlap
half = overlap // 2

st.title("PCB Defect Detector")

@st.cache_resource              # load the model only once, not on every click
def load_model():
    return YOLO("models/pcb_yolo26l_320.pt")

model = load_model()

uploaded = st.file_uploader("Upload a PCB image", type=["jpg", "jpeg", "png"])
conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)

if uploaded:
    img = np.array(Image.open(uploaded).convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)   # YOLO wants BGR like cv2, not RGB
    img_h, img_w = img.shape[:2]

    out = img.copy()
    counts = {}
    n_tiles = 0
    n_defects = 0

    # go over the image tile by tile, each tile is checked and drawn right away
    for y in range(0, max(img_h - overlap, 1), step):
        for x in range(0, max(img_w - overlap, 1), step):
            piece = img_bgr[y:y + tile, x:x + tile]   # last tiles on the edge can be smaller
            r = model(piece, conf=conf, imgsz=tile, verbose=False)[0]
            n_tiles += 1

            # the overlap is cut in the middle: this tile only keeps the boxes
            # whose center is in its own part, the other half goes to the next tile
            # (first tile starts at 0, last tile goes until the image edge)
            left = x + half if x > 0 else 0
            top = y + half if y > 0 else 0
            right = x + tile - half if x + tile < img_w else img_w
            bottom = y + tile - half if y + tile < img_h else img_h

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # box is in tile position, add the tile offset to put it on the full image
                x1, x2 = x1 + x, x2 + x
                y1, y2 = y1 + y, y2 + y

                # same defect is also seen by the neighbour tile, skip it here
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                if cx < left or cx >= right or cy < top or cy >= bottom:
                    continue

                name = model.names[int(box.cls)]
                c = float(box.conf)
                counts[name] = counts.get(name, 0) + 1
                n_defects += 1
                cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)   # red (image is RGB)
                cv2.putText(out, f"{name} {c:.2f}", (x1, max(y1 - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    st.image(out, caption=f"{n_defects} defects found ({n_tiles} tiles)", use_container_width=True)
    st.write(counts if counts else "No defects found")
