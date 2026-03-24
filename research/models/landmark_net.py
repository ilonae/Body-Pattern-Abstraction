"""
landmark_net.py
---------------
LandmarkNet: learned body landmark detector.

Architecture:
  PointNet encoder -> global feature -> MLP head -> 7 scalar body measurements

The 7 scalar outputs correspond to the BodyLandmarks.as_vector() fields:
  [shoulder_width, hip_width, waist_width,
   left_arm_length, right_arm_length,
   left_leg_length, right_leg_length]

All values are in normalised coordinates (0–1 relative to body height).

Training approach:
  - Phase 1: train on synthetic clouds from generate_synthetic_body.py
    (ground truth scalars are known from the SyntheticBodyConfig)
  - Phase 2: fine-tune on real scans (if available)

This follows the NeuralTailor principle of using point-level features
for reconstruction tasks, simplified for the body measurement use case.
"""

from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from research.models.encoder import PointNetEncoder

LANDMARK_DIM = 7   # number of scalar body measurements to predict


class LandmarkNet(nn.Module):
    """
    End-to-end network: point cloud to scalar body measurements.

    Input:  (B, N, 3) point cloud
    Output: (B, 7)    normalised scalar measurements

    Parameters
    ----------
    feature_dim : PointNet encoder output dimension
    dropout     : dropout rate in the prediction head
    """

    def __init__(self, feature_dim: int = 256, dropout: float = 0.3):
        super().__init__()

        self.encoder = PointNetEncoder(input_dim=3, feature_dim=feature_dim)

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, LANDMARK_DIM),
            nn.Sigmoid(),   # outputs in [0, 1] — normalised measurements
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N, 3) — normalised, Y-up point cloud

        Returns
        -------
        (B, 7) — predicted scalar measurements in [0, 1]
        """
        feat = self.encoder(x)     # (B, feature_dim)
        return self.head(feat)     # (B, 7)

    def predict_single(self, points: "np.ndarray") -> "np.ndarray":
        """
        Convenience method for single-cloud inference (no batching).

        Parameters
        ----------
        points : (N, 3) numpy array — preprocessed point cloud

        Returns
        -------
        (7,) numpy array — predicted scalar measurements
        """
        import numpy as np
        self.eval()
        with torch.no_grad():
            t = torch.from_numpy(points.astype("float32")).unsqueeze(0)
            out = self(t)
        return out.squeeze(0).numpy()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(),
                    "feature_dim": self.encoder.feature_dim}, path)

    @classmethod
    def load(cls, path: Path) -> "LandmarkNet":
        ckpt = torch.load(path, map_location="cpu")
        model = cls(feature_dim=ckpt["feature_dim"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

class LandmarkDataset(torch.utils.data.Dataset):
    """
    Dataset for training LandmarkNet on synthetic body clouds.

    Expects:
      data_dir/          - output of generate_synthetic_body.py
        standard_000.npy
        wheelchair_000.npy
        ...
        manifest.json    - with ground truth labels per file

    Ground truth labels are extracted from SyntheticBodyConfig via
    a separate label generation step (see scripts/generate_labels.py).
    """

    def __init__(self, data_dir: str | Path,
                 labels: dict[str, list[float]] | None = None,
                 n_points: int = 2048,
                 augment: bool = True):
        """
        Parameters
        ----------
        data_dir : directory with .npy point cloud files
        labels   : dict mapping filename -> [7 scalar measurements]
                   If None, uses zero labels (for inference / testing only)
        n_points : downsample to this many points
        augment  : apply random rotation + jitter during training
        """
        import json
        import numpy as np

        self.data_dir = Path(data_dir)
        self.n_points = n_points
        self.augment  = augment
        self.labels   = labels or {}

        manifest_path = self.data_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            self.files = [self.data_dir / m["file"] for m in manifest]
            self.file_names = [m["file"] for m in manifest]
        else:
            self.files = sorted(self.data_dir.glob("*.npy"))
            self.file_names = [f.name for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        import numpy as np

        points = np.load(self.files[idx]).astype("float32")

        # Subsample
        if len(points) > self.n_points:
            idxs = np.random.choice(len(points), self.n_points, replace=False)
            points = points[idxs]
        elif len(points) < self.n_points:
            idxs = np.random.choice(len(points), self.n_points, replace=True)
            points = points[idxs]

        # Augmentation: random Y-axis rotation + small jitter
        if self.augment:
            angle = np.random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot = np.array([[cos_a, 0, sin_a],
                             [0,    1, 0    ],
                             [-sin_a, 0, cos_a]], dtype="float32")
            points = (rot @ points.T).T
            points += np.random.normal(0, 0.002, points.shape).astype("float32")

        fname = self.file_names[idx]
        label = self.labels.get(fname, [0.0] * LANDMARK_DIM)
        label_t = torch.tensor(label, dtype=torch.float32)

        return torch.from_numpy(points), label_t


def train_one_epoch(model: LandmarkNet,
                    loader: torch.utils.data.DataLoader,
                    optimiser: torch.optim.Optimizer,
                    device: torch.device) -> float:
    """Train for one epoch, return mean loss."""
    model.train()
    total_loss = 0.0
    for points, labels in loader:
        points, labels = points.to(device), labels.to(device)
        optimiser.zero_grad()
        preds = model(points)
        loss  = F.mse_loss(preds, labels)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: LandmarkNet,
             loader: torch.utils.data.DataLoader,
             device: torch.device) -> dict[str, float]:
    """Evaluate model, return MAE per measurement axis."""
    NAMES = ["shoulder_w", "hip_w", "waist_w",
             "left_arm", "right_arm", "left_leg", "right_leg"]
    model.eval()
    all_preds, all_labels = [], []
    for points, labels in loader:
        preds = model(points.to(device)).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
    preds  = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    mae_per_axis = (preds - labels).abs().mean(dim=0)
    return {name: float(mae) for name, mae in zip(NAMES, mae_per_axis)}
