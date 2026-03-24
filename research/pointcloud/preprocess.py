"""
preprocess.py
-------------
Point cloud preprocessing pipeline for body scans.

Steps applied before any model sees the data:
  1. Outlier removal    — strip points far from the body
  2. Downsampling       — reduce to a fixed N points (voxel or random)
  3. Normalisation      — centre at origin, scale to unit height
  4. Canonical pose     — align body to Y-up, facing +Z

Design note: all functions are pure (no side effects), take and return
numpy arrays. The PreprocessPipeline class composes them in order.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Individual steps
# ---------------------------------------------------------------------------

def remove_outliers(points: np.ndarray,
                    k: int = 20,
                    std_ratio: float = 2.0) -> np.ndarray:
    """
    Statistical outlier removal.

    For each point, compute the mean distance to its k nearest neighbours.
    Remove points whose mean distance exceeds mean + std_ratio * std.

    Parameters
    ----------
    points    : (N, 3) float32
    k         : number of neighbours to consider
    std_ratio : threshold multiplier on the standard deviation

    Returns
    -------
    Filtered (M, 3) array, M <= N.
    """
    if len(points) < k + 1:
        return points

    # Compute pairwise distances (memory-efficient for moderate N)
    # For large clouds (>50k pts), use a KD-tree in Phase 3
    try:
        from scipy.spatial import KDTree
        tree = KDTree(points)
        dists, _ = tree.query(points, k=k + 1)   # includes self
    except ImportError:
        # Fallback: numpy-only brute force (slower, fine for N<5000)
        diff = points[:, None, :] - points[None, :, :]           # (N,N,3)
        all_dists = np.sqrt((diff ** 2).sum(axis=-1))            # (N,N)
        all_dists.sort(axis=1)
        dists = all_dists[:, :k + 1]
    mean_dists = dists[:, 1:].mean(axis=1)    # exclude self

    threshold = mean_dists.mean() + std_ratio * mean_dists.std()
    mask = mean_dists < threshold
    return points[mask]


def downsample_random(points: np.ndarray, n: int,
                      seed: int | None = None) -> np.ndarray:
    """
    Random downsampling to exactly n points.

    If len(points) < n, repeats (upsample) to reach n.
    """
    rng = np.random.default_rng(seed)
    if len(points) == n:
        return points
    idx = rng.choice(len(points), size=n, replace=len(points) < n)
    return points[idx]


def downsample_voxel(points: np.ndarray,
                     voxel_size: float = 0.01) -> np.ndarray:
    """
    Voxel grid downsampling — keeps one point per voxel (centroid).

    voxel_size is in the same units as the input (metres for most scans).
    """
    if voxel_size <= 0:
        return points
    # Bin points into voxels, keep centroid of each
    voxel_idx = np.floor(points / voxel_size).astype(np.int32)
    # Use structured key for grouping
    keys = voxel_idx[:, 0] * 1_000_000 + voxel_idx[:, 1] * 1_000 + voxel_idx[:, 2]
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    result = np.zeros((len(unique_keys), 3), dtype=np.float32)
    np.add.at(result, inverse, points)
    counts = np.bincount(inverse)
    result /= counts[:, None]
    return result


def normalise(points: np.ndarray,
              center: bool = True,
              scale_to_unit: bool = True) -> tuple[np.ndarray, dict]:
    """
    Centre the cloud and optionally scale to unit height.

    Returns
    -------
    normalised points, dict with 'offset' and 'scale' for inversion.
    """
    info: dict = {}

    if center:
        offset = points.mean(axis=0)
        points = points - offset
        info["offset"] = offset
    else:
        info["offset"] = np.zeros(3, dtype=np.float32)

    if scale_to_unit:
        height = points[:, 1].max() - points[:, 1].min()  # Y axis = height
        scale = height if height > 0 else 1.0
        points = points / scale
        info["scale"] = scale
    else:
        info["scale"] = 1.0

    return points.astype(np.float32), info


def align_canonical(points: np.ndarray) -> np.ndarray:
    """
    Rotate point cloud to canonical pose: Y-up, body centred at origin,
    facing +Z (anterior).

    This is a heuristic for upright body scans. For scans with arbitrary
    orientation, Phase 3 will add a learned pose estimator.

    Assumes input is already centred (after normalise()).
    """
    # Find the principal axis (tallest direction = body height)
    cov = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort by eigenvalue descending
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]

    # Align the principal axis to Y
    rotation = eigenvectors.T
    return (rotation @ points.T).T.astype(np.float32)


# ---------------------------------------------------------------------------
# Composable pipeline
# ---------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    n_points: int       = 2048   # target point count after downsampling
    voxel_size: float   = 0.005  # metres, 0 = skip voxel step
    outlier_k: int      = 20
    outlier_std: float  = 2.0
    center: bool        = True
    scale_to_unit: bool = True
    align: bool         = True
    random_seed: int    = 42


class PreprocessPipeline:
    """
    Composable preprocessing pipeline.

    Usage
    -----
    cfg = PreprocessConfig(n_points=2048)
    pipe = PreprocessPipeline(cfg)
    clean_pts, info = pipe(raw_pts)
    """

    def __init__(self, config: PreprocessConfig | None = None):
        self.cfg = config or PreprocessConfig()

    def __call__(self, points: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Run the full pipeline.

        Returns
        -------
        (processed_points, info_dict)
        info_dict contains normalisation parameters needed to invert.
        """
        info: dict = {"n_raw": len(points)}

        # Step 1: outlier removal
        points = remove_outliers(points,
                                 k=self.cfg.outlier_k,
                                 std_ratio=self.cfg.outlier_std)
        info["n_after_outlier"] = len(points)

        # Step 2: voxel downsample (optional)
        if self.cfg.voxel_size > 0:
            points = downsample_voxel(points, self.cfg.voxel_size)
            info["n_after_voxel"] = len(points)

        # Step 3: random downsample to exact target
        points = downsample_random(points, self.cfg.n_points,
                                   seed=self.cfg.random_seed)

        # Step 4: normalise
        points, norm_info = normalise(points,
                                      center=self.cfg.center,
                                      scale_to_unit=self.cfg.scale_to_unit)
        info.update(norm_info)

        # Step 5: canonical alignment
        if self.cfg.align:
            points = align_canonical(points)

        info["n_final"] = len(points)
        return points, info

    def invert_normalise(self, points: np.ndarray, info: dict) -> np.ndarray:
        """Undo normalisation (scale + offset) — useful for visualisation."""
        return (points * info.get("scale", 1.0)) + info.get("offset", 0.0)
