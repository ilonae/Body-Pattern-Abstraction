"""
research/mapping/scan_to_params.py
------------------------------------
Bridge layer: point cloud landmarks → BodyParams (discrete Level 0–4 system).

Overview
--------
This module is the translation layer between raw 3D geometry and the
discrete parameter system used by diverse-body-pattern-adjuster for
pattern adjustment. It converts either:

  A) A BodyLandmarks instance (from the heuristic LandmarkDetector), or
  B) A 7-dim numpy vector (from LandmarkNet's learned inference)

into a BodyParams instance describing what adjustments are needed.

Standalone vs. Integrated
--------------------------
This repo is fully self-contained. BodyParams is defined in
``research/core/body_params.py`` When the adjuster package is used 
alongside this repo, ScanToParams produces BodyParams objects that
can be passed directly to its PatternAdjuster and DXF export pipeline:

    # Standalone (research only)
    from research.core.body_params import BodyParams

    # Integrated (both repos installed)
    from diverse_body_pattern_adjuster.parameters.body_params import BodyParams

The ScanToParams class uses whichever is available, so your research
code doesn't need to change.

Threshold Calibration
---------------------
The THRESHOLDS dict maps normalised scalar measurements to Level values.
All measurements are expressed as a fraction of body height (so the
assignment is height-independent). Current values are empirically tuned
on synthetic data from generate_synthetic_body.py.

**These will need re-calibration once real scan data is available.**
The recommended workflow is:
  1. Collect annotated real scans (scan + known measurements)
  2. Run LandmarkDetector on each scan, record detected values
  3. Compare detected vs. ground-truth, adjust thresholds to minimise
     Level mis-assignments
  4. Retrain LandmarkNet with updated labels

Architecture in Context
-----------------------
                    .ply / .obj / .npy
                          │
                    loader.py (load)
                          │
                    preprocess.py (clean, normalise)
                          │
              ┌───────────┴──────────────┐
              │ Heuristic path           │ Learned path
              │ landmarks.py             │ landmark_net.py
              │ LandmarkDetector         │ LandmarkNet
              └───────────┬──────────────┘
                          │
                   scan_to_params.py  ◄── YOU ARE HERE
                   ScanToParams
                          │
                     BodyParams
                          │
          ┌───────────────┴────────────────────┐
          │ Standalone                          │ Integrated
          │ research/core/body_params.py        │ diverse-body-pattern-adjuster
          │ (this repo)                         │ PatternAdjuster → DXF
          └─────────────────────────────────────┘
"""

from __future__ import annotations
import types
import numpy as np

# ---------------------------------------------------------------------------
# Import strategy: prefer the adjuster package; fall back to local stub.
# This makes the repo fully standalone while remaining integrateable.
# ---------------------------------------------------------------------------
try:
    from diverse_body_pattern_adjuster.parameters.body_params import BodyParams, Level
    _USING_ADJUSTER = True
except ImportError:
    from research.core.body_params import BodyParams, Level
    _USING_ADJUSTER = False


# ---------------------------------------------------------------------------
# Threshold tables
# ---------------------------------------------------------------------------
# Each entry: list of (upper_bound, Level) pairs, checked in order.
# Measurement is assigned the Level of the first bound it falls BELOW.
# All values are normalised by body height (range ~0–1).
#
# Calibration basis: synthetic body clouds from generate_synthetic_body.py.
# Expected re-calibration interval: when first real scan dataset is available.

THRESHOLDS: dict[str, list[tuple[float, int]]] = {
    # Shoulder width as fraction of height
    # Reference adult: ~44cm / 170cm = 0.26
    "shoulder_width_norm": [
        (0.24, 0),
        (0.27, 1),
        (0.30, 2),
        (0.33, 3),
        (1.00, 4),
    ],
    # Hip width as fraction of height
    # Reference adult: ~38cm / 170cm = 0.22
    "hip_width_norm": [
        (0.23, 0),
        (0.26, 1),
        (0.29, 2),
        (0.33, 3),
        (1.00, 4),
    ],
    # Waist width as fraction of height
    "waist_width_norm": [
        (0.18, 0),
        (0.21, 1),
        (0.24, 2),
        (0.27, 3),
        (1.00, 4),
    ],
    # Fractional arm length asymmetry |L-R| / R
    "arm_asymmetry": [
        (0.04, 0),   # within noise floor
        (0.12, 1),
        (0.22, 2),
        (0.38, 3),
        (1.00, 4),
    ],
    # Fractional leg length asymmetry |L-R| / R
    "leg_asymmetry": [
        (0.04, 0),
        (0.10, 1),
        (0.18, 2),
        (0.28, 3),
        (1.00, 4),
    ],
    # Height deficit below reference (1.68m), in metres
    "height_deviation_m": [
        (0.05, 0),
        (0.12, 1),
        (0.22, 2),
        (0.38, 3),
        (1.00, 4),
    ],
    # Average leg length as fraction of height (shorter → higher inseam level)
    # Reference: ~50cm inseam / 170cm = 0.29
    "avg_leg_norm": [
        (1.00, 0),   # catch-all for standard and above
        (0.29, 0),
        (0.24, 1),
        (0.18, 2),
        (0.12, 3),
        (0.00, 4),
    ],
}


def _threshold_lookup(value: float, axis: str) -> int:
    """
    Map a normalised scalar measurement to a Level (0–4).

    Parameters
    ----------
    value : the measured value (fraction of body height, or deviation in m)
    axis  : key in THRESHOLDS

    Returns
    -------
    int (0–4)
    """
    if axis not in THRESHOLDS:
        return 0
    for upper_bound, level in THRESHOLDS[axis]:
        if value <= upper_bound:
            return level
    return 4


# ---------------------------------------------------------------------------
# Main mapping class
# ---------------------------------------------------------------------------

class ScanToParams:
    """
    Convert BodyLandmarks or LandmarkNet output into a BodyParams instance.

    Two entry points:
      - from_landmarks(lm)           : uses BodyLandmarks from LandmarkDetector
      - from_network_output(vector)  : uses LandmarkNet's 7-dim prediction

    Both paths go through the same threshold lookup, so the output is
    consistent regardless of which upstream detector was used.

    Attributes
    ----------
    REFERENCE_HEIGHT_M : float
        Body height used as the reference for height_level calculation.
        Adults below this height get a non-zero height_level.
    """

    REFERENCE_HEIGHT_M: float = 1.68

    def from_landmarks(self, lm, height_m: float | None = None) -> BodyParams:
        """
        Derive BodyParams from a BodyLandmarks instance.

        This is the heuristic path — no neural network involved.
        Use this for quick prototyping or when LandmarkNet is not trained yet.

        Parameters
        ----------
        lm       : BodyLandmarks from LandmarkDetector.detect()
        height_m : actual body height in metres, if known from scan metadata.
                   If None, height_level is not set.

        Returns
        -------
        BodyParams with relevant axes set to their detected Level.
        """
        params = BodyParams()

        # --- Shoulder width ---
        params.shoulder_width_level = Level(
            _threshold_lookup(lm.shoulder_width_norm, "shoulder_width_norm"))

        # --- Hip circumference proxy ---
        params.hip_level = Level(
            _threshold_lookup(lm.hip_width_norm, "hip_width_norm"))

        # --- Weight / volume proxy ---
        # Simple heuristic: average of hip and shoulder deviation levels
        params.weight_level = Level(
            min(4, (int(params.hip_level) + int(params.shoulder_width_level)) // 2))

        # --- Arm asymmetry → which side is affected ---
        arm_asym = lm.asymmetry_arm()
        arm_level = _threshold_lookup(arm_asym, "arm_asymmetry")
        if arm_level > 0:
            if lm.left_arm_length_norm < lm.right_arm_length_norm:
                params.left_upper_arm_step = Level(arm_level)
            else:
                params.right_upper_arm_step = Level(arm_level)

        # --- Leg asymmetry → which side is affected ---
        leg_asym = lm.asymmetry_leg()
        leg_level = _threshold_lookup(leg_asym, "leg_asymmetry")
        if leg_level > 0:
            if lm.left_leg_length_norm < lm.right_leg_length_norm:
                params.left_leg_step = Level(leg_level)
            else:
                params.right_leg_step = Level(leg_level)

        # --- Inseam: short average leg length ---
        avg_leg = (lm.left_leg_length_norm + lm.right_leg_length_norm) / 2.0
        if avg_leg > 0:
            il = _threshold_lookup(avg_leg, "avg_leg_norm")
            params.inseam_level = Level(il)

        # --- Height level (requires external height_m) ---
        if height_m is not None:
            deviation = max(0.0, self.REFERENCE_HEIGHT_M - height_m)
            params.height_level = Level(
                _threshold_lookup(deviation, "height_deviation_m"))

        return params

    def from_network_output(self, vector: np.ndarray,
                             height_m: float | None = None) -> BodyParams:
        """
        Derive BodyParams from LandmarkNet's 7-dimensional output vector.

        This is the learned path — LandmarkNet predicts body measurements
        from the raw point cloud, bypassing the heuristic detector.

        The 7 vector dimensions correspond to BodyLandmarks.as_vector():
          [0] shoulder_width_norm
          [1] hip_width_norm
          [2] waist_width_norm
          [3] left_arm_length_norm
          [4] right_arm_length_norm
          [5] left_leg_length_norm
          [6] right_leg_length_norm

        Parameters
        ----------
        vector   : (7,) float array, output of LandmarkNet.predict_single()
        height_m : actual body height in metres (from scan metadata if available)

        Returns
        -------
        BodyParams
        """
        # Build a minimal duck-typed landmark object from the vector
        class _VectorLandmarks:
            shoulder_width_norm:   float
            hip_width_norm:        float
            waist_width_norm:      float
            left_arm_length_norm:  float
            right_arm_length_norm: float
            left_leg_length_norm:  float
            right_leg_length_norm: float

            def asymmetry_arm(self) -> float:
                if self.right_arm_length_norm == 0:
                    return 0.0
                return abs(self.left_arm_length_norm - self.right_arm_length_norm) \
                       / self.right_arm_length_norm

            def asymmetry_leg(self) -> float:
                if self.right_leg_length_norm == 0:
                    return 0.0
                return abs(self.left_leg_length_norm - self.right_leg_length_norm) \
                       / self.right_leg_length_norm

        lm = _VectorLandmarks()
        lm.shoulder_width_norm   = float(vector[0])
        lm.hip_width_norm        = float(vector[1])
        lm.waist_width_norm      = float(vector[2])
        lm.left_arm_length_norm  = float(vector[3])
        lm.right_arm_length_norm = float(vector[4])
        lm.left_leg_length_norm  = float(vector[5])
        lm.right_leg_length_norm = float(vector[6])

        return self.from_landmarks(lm, height_m=height_m)

    def explain(self, lm, params: BodyParams) -> str:
        """
        Return a human-readable explanation of every mapping decision.

        This is the XAI interface for the mapping layer — it makes explicit
        which measurement drove each Level assignment, supporting clinical
        review and audit trails.

        Parameters
        ----------
        lm     : BodyLandmarks (or duck-typed equivalent)
        params : the BodyParams produced by from_landmarks(lm)

        Returns
        -------
        Multi-line explanation string.
        """
        lines = [
            "── Scan -> BodyParams Mapping Explanation ──",
            f"  shoulder_width (norm={lm.shoulder_width_norm:.3f})"
            f"  -> shoulder_width_level = {int(params.shoulder_width_level)}",
            f"  hip_width      (norm={lm.hip_width_norm:.3f})"
            f"  -> hip_level = {int(params.hip_level)}",
            f"  arm_asymmetry  (Δ={lm.asymmetry_arm():.3f})"
            f"  -> left_arm={int(params.left_upper_arm_step)}"
            f"  right_arm={int(params.right_upper_arm_step)}",
            f"  leg_asymmetry  (Δ={lm.asymmetry_leg():.3f})"
            f"  -> left_leg={int(params.left_leg_step)}"
            f"  right_leg={int(params.right_leg_step)}",
            f"  height_level   = {int(params.height_level)}",
            f"  inseam_level   = {int(params.inseam_level)}",
            f"  weight_level   = {int(params.weight_level)}",
        ]
        return "\n".join(lines)
