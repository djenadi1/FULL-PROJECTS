import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

root = Path("../data")
index = json.loads((root / "index.json").read_text())


def rotation(pose):
    heading = np.radians(pose["heading_from_east_deg"])
    pitch = np.radians(pose["pitch_deg"])
    forward = np.array([np.cos(pitch) * np.cos(heading), np.cos(pitch) * np.sin(heading), np.sin(pitch)])
    right = np.array([np.sin(heading), -np.cos(heading), 0.0])
    return np.stack([right, np.cross(right, forward), forward], 1)


def warp_depth(previous, camera, pose, motion):
    # previous depth -> 3D points -> move by the noisy IMU motion -> back into the current view
    fx, fy, cx, cy = camera["fx"], camera["fy"], camera["cx"], camera["cy"]
    v, u = np.nonzero(previous > 0)
    z = previous[v, u]
    points = np.stack([(u + 0.5 - cx) / fx * z, (cy - v - 0.5) / fy * z, z], 1)

    new_pose = dict(pose)
    new_pose["heading_from_east_deg"] += motion["heading_change_deg"]
    new_pose["pitch_deg"] += motion["pitch_change_deg"]
    move = np.array([motion["east_m"], motion["north_m"], motion["up_m"]])
    points = (points @ rotation(pose).T - move) @ rotation(new_pose)
    points = points[points[:, 2] > 0.1]

    new_u = np.floor(points[:, 0] / points[:, 2] * fx + cx).astype(int)
    new_v = np.floor(cy - points[:, 1] / points[:, 2] * fy).astype(int)
    inside = (new_u >= 0) & (new_u < previous.shape[1]) & (new_v >= 0) & (new_v < previous.shape[0])
    new_u, new_v, new_z = new_u[inside], new_v[inside], points[inside, 2]

    # far points first, so the closest one wins where several land on one pixel
    order = np.argsort(-new_z)
    warped = np.zeros_like(previous)
    warped[new_v[order], new_u[order]] = new_z[order]
    return warped


class SnowDataset(Dataset):
    def __init__(self, split):
        self.names = [i["name"] for i in index if i["split"] == split]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        rgb = np.asarray(Image.open(root / "frames" / f"{name}_rgb.png").convert("RGB"), dtype=np.float32) / 255
        exact = np.load(root / "frames" / f"{name}_depth.npy")
        base = np.load(f"../DA3_OUT/{name}_da3.npy")
        meta = json.loads((root / "frames" / f"{name}.json").read_text())

        points = np.load(root / "frames" / f"{name}_lidar.npy")
        lidar = np.zeros_like(exact)
        lidar[points[:, 1].astype(int), points[:, 0].astype(int)] = points[:, 2]

        # first frames have no previous frame, so their warp stays all zeros
        warped = np.zeros_like(exact)
        if meta["previous_frame"]:
            previous = np.load(f"../DA3_OUT/{meta['previous_frame']}_da3.npy")
            previous_meta = json.loads((root / "frames" / f"{meta['previous_frame']}.json").read_text())
            warped = warp_depth(previous, meta["camera"], previous_meta["pose_true"], meta["imu_motion"])

        return (
            torch.from_numpy(rgb).permute(2, 0, 1),
            torch.from_numpy(base)[None],
            torch.from_numpy(lidar)[None],
            torch.from_numpy((lidar > 0).astype(np.float32))[None],
            torch.from_numpy(warped)[None],
            torch.from_numpy((warped > 0).astype(np.float32))[None],
            torch.from_numpy(exact)[None],
        )
