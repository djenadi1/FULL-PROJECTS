"""Does a per-camera scale factor fix the speed model's error?

For every validation camera it sweeps a multiplier s over the model's speeds and reports:
  - oracle:    s chosen with all of that camera's cars, the most a scaler could ever fix
  - reference: s chosen with one car only, like driving a car of known speed past a new camera,
               then scored on the camera's other cars
Uses the functions and model from transf_dataprep_train.ipynb.

Usage: python calibrate_cameras.py [checkpoint]
"""
import random
import sys

import nbformat
import torch

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "speed_transformer.pt"
SCALES = torch.arange(0.70, 1.3001, 0.005)

# The notebook's functions, normaliser and model class, without running its training
nb = nbformat.read("transf_dataprep_train.ipynb", as_version=4)
cells = {c.source.splitlines()[0]: c.source for c in nb.cells if c.cell_type == "code" and c.source.strip()}
for start in ("import pathlib as pl", "def build_cars(data):", "def build_windows(", "# Normalise with numbers", "class SpeedTransformer"):
    source = next(text for head, text in cells.items() if head.startswith(start))
    if start == "# Normalise with numbers":
        source = source.split("stats = fit_normaliser")[0]
    exec(source)

checkpoint = torch.load(CHECKPOINT)
stats = checkpoint["stats"]
model = SpeedTransformer().to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

# Windows per val camera, so each camera can get its own scale
val_cars = build_cars(load_split("val"))
predictions, truths, car_ids = {}, {}, {}
for clip in sorted({clip for clip, _ in val_cars}):
    cars = {key: car for key, car in val_cars.items() if key[0] == clip}
    owners = [key for key, car in cars.items() for t in sorted(car["frames"])[::5]
              if sum(t + k in car["frames"] for k in range(-WINDOW, WINDOW + 1)) >= 10]
    X, M, Y, CLS, CAM = build_windows(cars)
    X, CAM = normalise(X, M, CAM, stats)
    with torch.no_grad():
        pred = torch.cat([model(X[i:i + 2048].to(device), M[i:i + 2048].to(device), CLS[i:i + 2048].to(device),
                                CAM[i:i + 2048].to(device))[:, WINDOW].cpu() for i in range(0, len(X), 2048)]) * 100
    predictions[clip], truths[clip], car_ids[clip] = pred, Y[:, WINDOW], torch.tensor([key[1] for key in owners])


def best_scale(pred, true):
    errors = (pred[None, :] * SCALES[:, None] - true[None, :]).abs().mean(1)
    return SCALES[errors.argmin()].item()


random.seed(0)
rows = []
for clip in predictions:
    pred, true, ids = predictions[clip], truths[clip], car_ids[clip]
    raw = (pred - true).abs().mean().item()
    oracle_s = best_scale(pred, true)
    oracle = (pred * oracle_s - true).abs().mean().item()
    # Reference car: 5 tries per camera with a random car that's in view for a while, scored on the others
    candidates = [car for car in ids.unique().tolist() if (ids == car).sum() >= 5]
    tries = []
    for car in random.sample(candidates, min(5, len(candidates))):
        s = best_scale(pred[ids == car], true[ids == car])
        others = ids != car
        tries.append((pred[others] * s - true[others]).abs().mean().item())
    rows.append((clip, len(true), raw, oracle_s, oracle, sum(tries) / len(tries) if tries else float("nan")))

print(f"checkpoint {CHECKPOINT}")
print(f"{'camera':>6} {'windows':>8} {'raw MAE':>8} {'best s':>7} {'oracle':>7} {'ref car':>8}")
for clip, n, raw, s, oracle, reference in rows:
    print(f"{clip:6d} {n:8d} {raw:8.2f} {s:7.3f} {oracle:7.2f} {reference:8.2f}")
total = sum(n for _, n, *_ in rows)
mean = lambda column: sum(row[1] * row[column] for row in rows) / total
global_s = best_scale(torch.cat(list(predictions.values())), torch.cat(list(truths.values())))
print(f"\nall val cameras, weighted by windows: raw {mean(2):.2f}  oracle scale {mean(4):.2f}  reference car {mean(5):.2f} km/h")
print(f"one scale for every camera would be {global_s:.3f}")
