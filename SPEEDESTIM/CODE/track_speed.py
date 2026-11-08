"""Tracks the cars in a video and estimates their speed.

YOLO finds the cars, ByteTrack gives each car an ID, and the speed transformer
looks at 30 frames before and after to guess the speed.

Usage: python track_speed.py video.mp4 [camera height in m] [pitch in degrees] [fov in degrees]
Example: python track_speed.py road.mp4 8 -8 65
The output is saved next to the video as video_speed.mp4
"""
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import imageio_ffmpeg
import torch
import torch.nn as nn
from ultralytics import YOLO

HERE = Path(__file__).parent
VIDEO = sys.argv[1]
CAMERA = [float(x) for x in sys.argv[2:5]] if len(sys.argv) >= 5 else [8.0, -8.0, 60.0]
MAX_SECONDS = float(sys.argv[5]) if len(sys.argv) >= 6 else None  # only use the start of a long video
OUT = VIDEO.rsplit(".", 1)[0] + "_speed.mp4"

W, H = 1920, 1080  # the size the models were trained on
WINDOW = 30  # 30 frames before and after
MIN_FRAMES = 15  # cars seen for less than half a second are skipped
NAMES = ["car", "van", "truck", "bus"]
COLOURS = [(255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49), (72, 249, 10),
           (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
           (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255), (82, 0, 133), (203, 56, 255)]


# the same model as in the notebook
class SpeedTransformer(nn.Module):
    def __init__(self, n_features=12, d_model=128, n_heads=4, n_layers=4, dropout=0.1):
        super().__init__()
        self.token_in = nn.Linear(n_features, d_model)
        self.position = nn.Embedding(2 * WINDOW + 1, d_model)
        self.class_emb = nn.Embedding(4, d_model)
        self.camera_in = nn.Linear(3, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, present, cls, cam):
        positions = torch.arange(x.shape[1], device=x.device)
        h = self.token_in(x) + self.position(positions)
        h = h + (self.class_emb(cls) + self.camera_in(cam))[:, None, :]
        h = self.encoder(h, src_key_padding_mask=~present)
        return self.head(self.norm(h)).squeeze(-1)


# 1. load the models
yolo = YOLO(HERE / "models/yolo_colour.pt")  # a copy of runs/segment/v2_colour_l_10h/weights/best.pt (YOLO26l-seg, 10 hours)
checkpoint = torch.load(HERE / "speed_transformer.pt", map_location="cpu")
model = SpeedTransformer()
model.load_state_dict(checkpoint["model"])
model.eval()
stats = {k: v.cpu() for k, v in checkpoint["stats"].items()}

# 2. find and track the cars, keep the same features the transformer was trained on
cars = defaultdict(dict)  # car id -> {frame: features}
car_classes = defaultdict(list)  # car id -> class on every frame
boxes = defaultdict(dict)  # frame -> {car id: box}

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if MAX_SECONDS:
    total = min(total, int(MAX_SECONDS * fps)) if total > 0 else int(MAX_SECONDS * fps)
n_frames = 0
while True:
    ok, frame = cap.read()
    if not ok or (MAX_SECONDS and n_frames >= total):
        break
    if total > 0 and n_frames % 10 == 0:
        print(f"progress {0.9 * n_frames / total:.3f}", flush=True)  # for the app's loading bar
    frame = cv2.resize(frame, (W, H))
    # conf has to be as low as the tracker's low threshold, ByteTrack uses the weak boxes to keep tracks going
    result = yolo.track(frame, imgsz=1280, conf=0.1, persist=True, tracker=str(HERE / "tracker.yaml"), verbose=False)[0]

    if result.boxes.id is not None and result.masks is not None:
        ids = result.boxes.id.int().tolist()
        classes = result.boxes.cls.int().tolist()
        for box, car_id, cls, polygon in zip(result.boxes.xyxy.tolist(), ids, classes, result.masks.xy):
            if len(polygon) < 3:
                continue
            x1, y1, x2, y2 = box
            # the lowest point of the mask, where the tyres touch the road
            bottom = polygon[:, 1].max()
            lowest_x = polygon[polygon[:, 1] >= bottom - 2, 0].mean()
            pixels = max(cv2.contourArea(polygon), 1.0)
            edge = float(x1 <= 0 or y1 <= 0 or x2 >= W - 1 or y2 >= H - 1)

            cars[car_id][n_frames] = [lowest_x / W, bottom / H, x1 / W, y1 / H, x2 / W, y2 / H, math.log(pixels), edge]
            car_classes[car_id].append(cls)
            boxes[n_frames][car_id] = box
    n_frames += 1
cap.release()
print(f"tracked {n_frames} frames, {len(cars)} car ids")


# 3. speed of every car on every frame
def make_window(frames, t):
    # same tokens as build_windows in the notebook
    lx_t, ly_t = frames[t][0], frames[t][1]
    tokens, present = [], []
    for k in range(-WINDOW, WINDOW + 1):
        if t + k in frames:
            f = frames[t + k]
            tokens.append([1.0] + f + [f[0] - lx_t, f[1] - ly_t, k / WINDOW])
            present.append(True)
        else:
            tokens.append([0.0] * 11 + [k / WINDOW])
            present.append(False)
    return tokens, present


SCALE_COLUMNS = [7, 9, 10]
speeds = defaultdict(dict)  # car id -> {frame: km/h}
for car_id, frames in cars.items():
    if len(frames) < MIN_FRAMES:
        continue
    times = sorted(frames)
    windows = [make_window(frames, t) for t in times]
    X = torch.tensor([w[0] for w in windows])
    M = torch.tensor([w[1] for w in windows])

    # normalise like in the notebook
    X[:, :, SCALE_COLUMNS] = (X[:, :, SCALE_COLUMNS] - stats["x_mean"]) / stats["x_std"]
    X[~M] = 0.0
    X[:, :, 11] = torch.arange(-WINDOW, WINDOW + 1) / WINDOW
    cam = (torch.tensor(CAMERA) - stats["cam_mean"]) / stats["cam_std"]

    cls = Counter(car_classes[car_id]).most_common(1)[0][0]  # the class it got most often
    with torch.no_grad():
        pred = model(X, M, torch.full((len(times),), cls), cam.repeat(len(times), 1))[:, WINDOW] * 100

    for t, kmh, n_present in zip(times, pred.tolist(), M.sum(1).tolist()):
        if n_present >= 10:
            speeds[car_id][t] = kmh * fps / 30  # the model learnt at 30 fps

# 4. draw everything and save the video
cap = cv2.VideoCapture(VIDEO)
# H.264 so the video plays everywhere (OpenCV's own mp4 codec doesn't play in most players)
writer = imageio_ffmpeg.write_frames(OUT, (W, H), fps=fps, codec="libx264", pix_fmt_in="bgr24", quality=7,
                                     macro_block_size=1)
writer.send(None)  # starts the writer
history = defaultdict(list)
for t in range(n_frames):
    if t % 30 == 0:
        print(f"progress {0.9 + 0.1 * t / n_frames:.3f}", flush=True)
    ok, frame = cap.read()
    frame = cv2.resize(frame, (W, H))
    for car_id, box in boxes[t].items():
        if len(cars[car_id]) < MIN_FRAMES:
            continue
        x1, y1, x2, y2 = map(int, box)
        colour = COLOURS[car_id % len(COLOURS)]
        cls = Counter(car_classes[car_id]).most_common(1)[0][0]
        text = f"#{car_id} {NAMES[cls]}"
        if t in speeds[car_id]:
            history[car_id].append(speeds[car_id][t])
            h = history[car_id]
            text += f" {h[-1]:.0f} km/h  avg {sum(h) / len(h):.0f}  max {max(h):.0f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    writer.send(frame)
writer.close()
print("saved", OUT)
