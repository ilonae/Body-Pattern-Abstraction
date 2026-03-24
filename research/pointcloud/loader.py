"""
loader.py
---------
Load body scan point clouds from common formats.

Supported formats:
  .ply   — most 3D scanners (Artec, iPhone LiDAR, RealSense)
  .obj   — common 3D model format
  .npy   — pre-processed numpy arrays (fast, no dependencies)
  .xyz   — plain text, one point per line (x y z)
  .txt   — same as .xyz

All loaders return a numpy array of shape (N, 3) — raw XYZ coordinates.
Normals and colour channels are optionally preserved but not required
by the pipeline.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np


def load_pointcloud(path: str | Path) -> np.ndarray:
    """
    Load a point cloud from file. Returns (N, 3) float32 array.

    Dispatcher — picks the right loader based on file extension.
    """
    path = Path(path)
    ext = path.suffix.lower()

    loaders = {
        ".ply":  _load_ply,
        ".obj":  _load_obj,
        ".npy":  _load_npy,
        ".npz":  _load_npz,
        ".xyz":  _load_xyz,
        ".txt":  _load_xyz,
    }

    if ext not in loaders:
        raise ValueError(f"Unsupported format: {ext}. "
                         f"Supported: {list(loaders.keys())}")

    points = loaders[ext](path)
    return points.astype(np.float32)


# ---------------------------------------------------------------------------
# Format-specific loaders
# ---------------------------------------------------------------------------

def _load_ply(path: Path) -> np.ndarray:
    """Load PLY file using plyfile (preferred) or manual parser."""
    try:
        from plyfile import PlyData
        ply = PlyData.read(path)
        v = ply["vertex"]
        return np.stack([v["x"], v["y"], v["z"]], axis=1)
    except ImportError:
        # Fallback: manual ASCII PLY parser
        return _load_ply_manual(path)


def _load_ply_manual(path: Path) -> np.ndarray:
    """Minimal ASCII PLY parser — no dependencies."""
    points = []
    in_data = False
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line == "end_header":
                in_data = True
                continue
            if in_data:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        points.append([float(parts[0]),
                                       float(parts[1]),
                                       float(parts[2])])
                    except ValueError:
                        continue
    return np.array(points, dtype=np.float32)


def _load_obj(path: Path) -> np.ndarray:
    """Load vertex positions from OBJ file."""
    points = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                points.append([float(parts[1]),
                                float(parts[2]),
                                float(parts[3])])
    return np.array(points, dtype=np.float32)


def _load_npy(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected (N, 3+) array, got {arr.shape}")
    return arr[:, :3]


def _load_npz(path: Path) -> np.ndarray:
    data = np.load(path)
    key = "points" if "points" in data else list(data.keys())[0]
    arr = data[key]
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected (N, 3+) array, got {arr.shape}")
    return arr[:, :3]


def _load_xyz(path: Path) -> np.ndarray:
    """Load whitespace-delimited XYZ text file."""
    points = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    points.append([float(parts[0]),
                                   float(parts[1]),
                                   float(parts[2])])
                except ValueError:
                    continue
    return np.array(points, dtype=np.float32)


def save_npy(points: np.ndarray, path: str | Path) -> Path:
    """Save point cloud as .npy for fast reloading."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, points.astype(np.float32))
    return path
