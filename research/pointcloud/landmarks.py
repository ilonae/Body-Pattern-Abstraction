"""
landmarks.py
------------
Extract anatomical body landmarks from a preprocessed body scan.

Landmarks are the key measurement points needed to derive BodyParams.
This is the bridge between raw 3D geometry and the discrete Level system.

Current approach: heuristic geometric detection on normalised clouds.
Phase 2 will replace this with a learned detector (LandmarkNet).

Coordinate system (after normalise + align_canonical):
  Y-up (0 = feet, 1 = top of head)
  X-right (positive = right side of body from viewer)
  Z-forward (positive = anterior/front)

Landmarks extracted (all in normalised coordinates):
  crown           : top of head (y_max)
  left_shoulder   : lateral edge of left shoulder
  right_shoulder  : lateral edge of right shoulder
  left_hip        : lateral edge of left hip (greater trochanter approx)
  right_hip       : lateral edge of right hip
  left_knee       : midpoint of left knee region
  right_knee      : midpoint of right knee region
  left_ankle      : bottom of left leg
  right_ankle     : bottom of right leg
  waist_front     : anterior midpoint at narrowest waist level
  waist_back      : posterior midpoint at narrowest waist level
  bust_width      : full width at bust level
  hip_width       : full width at hip level
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BodyLandmarks:
    """
    Detected landmark positions (normalised Y-up coordinates).
    Missing landmarks are None (detection failed for that point).
    """
    crown:           Optional[np.ndarray] = None  # (3,)
    left_shoulder:   Optional[np.ndarray] = None
    right_shoulder:  Optional[np.ndarray] = None
    left_hip:        Optional[np.ndarray] = None
    right_hip:       Optional[np.ndarray] = None
    left_knee:       Optional[np.ndarray] = None
    right_knee:      Optional[np.ndarray] = None
    left_ankle:      Optional[np.ndarray] = None
    right_ankle:     Optional[np.ndarray] = None
    waist_front:     Optional[np.ndarray] = None
    waist_back:      Optional[np.ndarray] = None

    # Derived scalar measurements (in normalised units; multiply by height_m to get metres)
    shoulder_width_norm:  float = 0.0
    hip_width_norm:       float = 0.0
    waist_width_norm:     float = 0.0
    left_arm_length_norm: float = 0.0
    right_arm_length_norm:float = 0.0
    left_leg_length_norm: float = 0.0
    right_leg_length_norm:float = 0.0
    torso_height_norm:    float = 0.0
    seated_rise_norm:     float = 0.0  # for wheelchair profiles

    def as_vector(self) -> np.ndarray:
        """Return a flat feature vector of all scalar measurements (7 dims)."""
        return np.array([
            self.shoulder_width_norm,
            self.hip_width_norm,
            self.waist_width_norm,
            self.left_arm_length_norm,
            self.right_arm_length_norm,
            self.left_leg_length_norm,
            self.right_leg_length_norm,
        ], dtype=np.float32)

    def asymmetry_arm(self) -> float:
        """Fractional asymmetry between left and right arm lengths."""
        if self.right_arm_length_norm == 0:
            return 0.0
        return abs(self.left_arm_length_norm - self.right_arm_length_norm) / self.right_arm_length_norm

    def asymmetry_leg(self) -> float:
        """Fractional asymmetry between left and right leg lengths."""
        if self.right_leg_length_norm == 0:
            return 0.0
        return abs(self.left_leg_length_norm - self.right_leg_length_norm) / self.right_leg_length_norm


class LandmarkDetector:
    """
    Heuristic landmark detector for normalised, Y-up body scans.

    Works by identifying anatomical regions based on height bands
    and lateral extremes. Reliable for upright standing scans;
    will be augmented by a learned model in Phase 2.
    """

    def __init__(self, band_width: float = 0.05):
        """
        band_width : height band (in normalised units) used for region detection.
        """
        self.band_width = band_width

    def detect(self, points: np.ndarray) -> BodyLandmarks:
        """
        Detect landmarks from a preprocessed, normalised (N, 3) point cloud.

        Y range is approximately [0, 1] after normalisation.
        """
        lm = BodyLandmarks()
        y = points[:, 1]
        y_min, y_max = y.min(), y.max()
        height = y_max - y_min

        # --- Crown ---
        lm.crown = points[y.argmax()]

        # --- Ankles (bottom 5% of height) ---
        ankle_mask = y < y_min + height * 0.05
        if ankle_mask.sum() > 10:
            ankle_pts = points[ankle_mask]
            # Split left / right by X
            left_ankle_pts  = ankle_pts[ankle_pts[:, 0] > 0]
            right_ankle_pts = ankle_pts[ankle_pts[:, 0] < 0]
            if len(left_ankle_pts):
                lm.left_ankle  = left_ankle_pts.mean(axis=0)
            if len(right_ankle_pts):
                lm.right_ankle = right_ankle_pts.mean(axis=0)

        # --- Knees (30-40% of height) ---
        lm.left_knee, lm.right_knee = self._detect_lateral_pair(
            points, y_min + height * 0.30, y_min + height * 0.42)

        # --- Hips (50-65% of height) — widest point in this band ---
        lm.left_hip, lm.right_hip = self._detect_lateral_pair(
            points, y_min + height * 0.50, y_min + height * 0.65,
            use_extremes=True)

        # --- Waist (narrowest X-width between hip and bust) ---
        waist_y = self._find_narrowest_band(points, y_min + height * 0.55,
                                             y_min + height * 0.72)
        if waist_y is not None:
            w_mask = np.abs(points[:, 1] - waist_y) < self.band_width
            w_pts  = points[w_mask]
            if len(w_pts):
                lm.waist_front = w_pts[w_pts[:, 2].argmax()]
                lm.waist_back  = w_pts[w_pts[:, 2].argmin()]
                lm.waist_width_norm = float(w_pts[:, 0].max() - w_pts[:, 0].min())

        # --- Shoulders (82-92% of height) ---
        lm.left_shoulder, lm.right_shoulder = self._detect_lateral_pair(
            points, y_min + height * 0.82, y_min + height * 0.92,
            use_extremes=True)

        # --- Derived scalar measurements ---
        lm.shoulder_width_norm = self._dist_x(lm.left_shoulder, lm.right_shoulder)
        lm.hip_width_norm      = self._dist_x(lm.left_hip,      lm.right_hip)

        lm.left_leg_length_norm  = self._dist_y(lm.left_ankle,  lm.left_hip)
        lm.right_leg_length_norm = self._dist_y(lm.right_ankle, lm.right_hip)

        # Arm: shoulder to wrist (ankle region is a proxy for wrist here)
        # Will be improved with a learned detector
        lm.left_arm_length_norm  = self._estimate_arm_length(points, side="left",
                                                               shoulder=lm.left_shoulder,
                                                               y_min=y_min, height=height)
        lm.right_arm_length_norm = self._estimate_arm_length(points, side="right",
                                                              shoulder=lm.right_shoulder,
                                                              y_min=y_min, height=height)

        lm.torso_height_norm = self._dist_y(lm.left_hip, lm.left_shoulder)

        return lm

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _detect_lateral_pair(self, points: np.ndarray,
                              y_lo: float, y_hi: float,
                              use_extremes: bool = False
                              ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Find left (x>0) and right (x<0) centroids in a height band."""
        mask = (points[:, 1] >= y_lo) & (points[:, 1] <= y_hi)
        if mask.sum() < 4:
            return None, None
        band = points[mask]
        left  = band[band[:, 0] > 0]
        right = band[band[:, 0] < 0]
        if use_extremes:
            l_pt = band[band[:, 0].argmax()] if len(left) else None
            r_pt = band[band[:, 0].argmin()] if len(right) else None
        else:
            l_pt = left.mean(axis=0)  if len(left)  else None
            r_pt = right.mean(axis=0) if len(right) else None
        return l_pt, r_pt

    def _find_narrowest_band(self, points: np.ndarray,
                              y_lo: float, y_hi: float) -> Optional[float]:
        """Find Y position of minimum X-width (waist)."""
        ys = np.arange(y_lo, y_hi, self.band_width)
        min_width = np.inf
        min_y = None
        for y in ys:
            mask = np.abs(points[:, 1] - y) < self.band_width * 0.5
            if mask.sum() < 4:
                continue
            width = float(points[mask, 0].max() - points[mask, 0].min())
            if width < min_width:
                min_width = width
                min_y = y
        return min_y

    def _estimate_arm_length(self, points: np.ndarray, side: str,
                              shoulder: Optional[np.ndarray],
                              y_min: float, height: float) -> float:
        """
        Rough arm length: from shoulder down to the hand (lateral column
        at shoulder height and below). Returns normalised length.
        """
        if shoulder is None:
            return 0.0
        x_sign = 1.0 if side == "left" else -1.0
        # Points lateral to body centre and below shoulder
        arm_mask = (
            (np.sign(points[:, 0]) == np.sign(x_sign)) &
            (np.abs(points[:, 0]) > 0.08) &   # exclude torso centre
            (points[:, 1] < shoulder[1]) &
            (points[:, 1] > y_min + height * 0.35)
        )
        if arm_mask.sum() < 4:
            return 0.0
        arm_pts = points[arm_mask]
        return float(shoulder[1] - arm_pts[:, 1].min())

    @staticmethod
    def _dist_x(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        if a is None or b is None:
            return 0.0
        return float(abs(a[0] - b[0]))

    @staticmethod
    def _dist_y(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        if a is None or b is None:
            return 0.0
        return float(abs(a[1] - b[1]))
