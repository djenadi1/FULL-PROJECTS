import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

root = Path("../data")
index = json.loads((root / "index.json").read_text())

# Modelled on the Velodyne Alpha Prime used in Boreas (40 degree view, 0.2 degree steps, 300 m at 10% reflectivity).
# Line count and tilt vary so the model doesn't depend on one mounting. Each frame draws its own values.
LIDAR_LINES = (32, 64, 128)
LIDAR_VERTICAL_FOV = 40.0
LIDAR_TILT_DEG = (-5.0, 5.0)
LIDAR_STEP_DEG = 0.2
LIDAR_MAX_M = (220.0, 300.0)
LIDAR_FADE_SHARE = 0.85
# Real Boreas beams aimed below the horizon returned nearly every shot, even on road at shallow angles, so only
# extreme grazing loses points. The exact limit and how noise grows with distance still aren't measured.
LIDAR_GRAZING_COS = (0.01, 0.04)
LIDAR_NOISE_M = (0.015, 0.03)
LIDAR_NOISE_GROWTH = (0.0001, 0.0003)
# Measured on two heavy snow Boreas drives (2021-01-26), minus the same count on nine drives without snow.
# 3.6% of beams returned a flake within 30 m. The lower limit stands in for lighter snow, which wasn't measured.
SNOW_RETURN_RATE = (0.002, 0.06)
HEAVY_SNOW_RATE = 0.036
SNOW_RETURN_EDGES = np.array([3, 4, 5, 6, 8, 10, 15, 20, 30], dtype=float)
SNOW_RETURN_COUNTS = np.array([1127, 1639, 2213, 6500, 8972, 31970, 36300, 35622], dtype=float)
# The furthest returns fell from about 180 m to about 95 m in that snow
HEAVY_SNOW_RANGE_SHARE = 0.52

SPREAD_PIXELS = 40


def simulate_lidar(depth, camera, weather, rng):
    """Exact z-depth to a sparse LiDAR dot image: scan lines, dropouts, range noise and snowflake returns."""
    height, width = depth.shape
    furthest = rng.uniform(*LIDAR_MAX_M)
    half_fov = LIDAR_VERTICAL_FOV / 2
    tilt = rng.uniform(*LIDAR_TILT_DEG)
    snow_rate = rng.uniform(*SNOW_RETURN_RATE) if weather == "snowfall" else 0.0
    # Heavier snow shortens the range, scaled from the one heavy snowfall measured
    furthest *= max(1 - (1 - HEAVY_SNOW_RANGE_SHARE) * snow_rate / HEAVY_SNOW_RATE, 0.4)
    fade = furthest * LIDAR_FADE_SHARE
    fx, fy, cx, cy = camera["fx"], camera["fy"], camera["cx"], camera["cy"]

    # Beam pattern: a forward-facing sensor, tilted a little against the camera
    half_width = np.degrees(np.arctan(cx / fx))
    azimuth = np.radians(np.arange(-half_width, half_width, LIDAR_STEP_DEG))
    elevation = np.radians(np.linspace(tilt - half_fov, tilt + half_fov, rng.choice(LIDAR_LINES)))
    azimuth, elevation = np.meshgrid(azimuth, elevation)
    u = np.floor(cx + fx * np.tan(azimuth)).astype(int)
    v = np.floor(cy - fy * np.tan(elevation) / np.cos(azimuth)).astype(int)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v = u[inside], v[inside]
    z = depth[v, u]
    to_range = np.sqrt(((u + 0.5 - cx) / fx) ** 2 + ((cy - v - 0.5) / fy) ** 2 + 1)
    distance = z * to_range

    # Dropouts: too far away, or hitting the surface at a grazing angle
    def point(column, row):
        d = depth[row, column]
        return np.stack([(column + 0.5 - cx) / fx * d, (cy - row - 0.5) / fy * d, d], 1)

    across = point(np.minimum(u + 1, width - 1), v) - point(np.maximum(u - 1, 0), v)
    down = point(u, np.minimum(v + 1, height - 1)) - point(u, np.maximum(v - 1, 0))
    normal = np.cross(across, down)
    here = point(u, v)
    facing = np.abs(np.sum(normal * here, 1)) / np.maximum(np.linalg.norm(normal, axis=1) * np.linalg.norm(here, axis=1), 1e-6)
    chance = np.clip((furthest - distance) / (furthest - fade), 0, 1) * np.clip(facing / rng.uniform(*LIDAR_GRAZING_COS), 0, 1)
    returned = (z > 0) & (rng.random(z.shape) < chance)

    # Range noise: a few centimetres, growing with distance
    distance = distance + rng.normal(0, rng.uniform(*LIDAR_NOISE_M) + rng.uniform(*LIDAR_NOISE_GROWTH) * distance)

    # Snow returns: the sensor keeps its strongest return, so a flake replaces whatever was behind it
    flakes = rng.random(z.shape) < snow_rate
    bins = rng.choice(len(SNOW_RETURN_COUNTS), flakes.sum(), p=SNOW_RETURN_COUNTS / SNOW_RETURN_COUNTS.sum())
    distance[flakes] = rng.uniform(SNOW_RETURN_EDGES[bins], SNOW_RETURN_EDGES[bins + 1])
    returned |= flakes

    # Back to z-depth so it matches the depth map. Where two beams hit one pixel the closer one wins
    u, v, new_z = u[returned], v[returned], (distance / to_range)[returned]
    order = np.argsort(-new_z)
    lidar = np.zeros((height, width), dtype=np.float32)
    lidar[v[order], u[order]] = new_z[order]
    return lidar


def remove_lone(lidar, radius=6.0, depth_weight=15.0):
    # real surfaces give neighbouring points at a similar depth, and a snowflake usually doesn't
    v, u = np.nonzero(lidar)
    if len(v) < 3:
        return np.zeros_like(lidar)
    tree = cKDTree(np.stack([u, v, depth_weight * np.log(lidar[v, u])], 1))
    keep = np.array([len(n) >= 3 for n in tree.query_ball_point(tree.data, radius)])
    out = np.zeros_like(lidar)
    out[v[keep], u[keep]] = lidar[v[keep], u[keep]]
    return out


def spread(lidar):
    # each pixel takes its nearest LiDAR point's depth, and how far away that point is
    if not (lidar > 0).any():
        return np.zeros_like(lidar), np.full_like(lidar, 1e3)
    distance, labels = cv2.distanceTransformWithLabels((lidar <= 0).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    v, u = np.nonzero(lidar)
    table = np.zeros(labels.max() + 1, dtype=np.float32)
    table[labels[v, u]] = lidar[v, u]
    filled = table[labels]
    filled[distance > SPREAD_PIXELS] = 0
    return filled, distance.astype(np.float32)


class SnowDataset(Dataset):
    def __init__(self, split, fresh=False):
        self.names = [i["name"] for i in index if i["split"] == split]
        self.fresh = fresh

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        # fresh LiDAR noise every epoch for training, and the same scan every time for validation
        rng = np.random.default_rng() if self.fresh else np.random.default_rng(i)
        rgb = np.asarray(Image.open(root / "frames" / f"{name}_rgb.png").convert("RGB"), dtype=np.float32) / 255
        exact = np.load(root / "frames" / f"{name}_depth.npy")
        base = np.load(f"../DA3_OUT/{name}_da3.npy")
        meta = json.loads((root / "frames" / f"{name}.json").read_text())
        lidar = remove_lone(simulate_lidar(exact, meta["camera"], meta["weather"], rng).astype(np.float32))

        if self.fresh:
            rgb = np.clip(rgb * rng.uniform(0.8, 1.2) + rng.uniform(-0.05, 0.05), 0, 1).astype(np.float32)
            if rng.random() < 0.5:
                rgb, base, lidar, exact = [np.flip(a, 1).copy() for a in (rgb, base, lidar, exact)]

        filled, distance = spread(lidar)
        return (
            torch.from_numpy(rgb).permute(2, 0, 1),
            torch.from_numpy(base)[None],
            torch.from_numpy(filled)[None],
            torch.from_numpy(distance)[None],
            torch.from_numpy(exact)[None],
        )
