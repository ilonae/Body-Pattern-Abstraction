"""
tests/test_preprocessing.py
----------------------------
Basic tests for point cloud preprocessing and landmark detection.
Run with: pytest tests/ -v
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from research.pointcloud.preprocess import (
    PreprocessConfig, PreprocessPipeline,
    remove_outliers, downsample_random, normalise, align_canonical,
)
from research.pointcloud.landmarks import LandmarkDetector, BodyLandmarks
from research.mapping.scan_to_params import ScanToParams, _threshold_lookup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_body_cloud(n: int = 2048, seed: int = 0) -> np.ndarray:
    """Create a minimal synthetic body-shaped cloud for testing."""
    rng = np.random.default_rng(seed)
    # Rough cylinder representing a standing body, Y in [0, 1]
    theta = rng.uniform(0, 2 * np.pi, n)
    y = rng.uniform(0, 1.0, n)
    # Radius varies by height (wider at hips, narrower at waist)
    r = 0.12 + 0.04 * np.sin(y * np.pi) - 0.02 * np.sin(2 * y * np.pi)
    x = r * np.cos(theta)
    z = r * np.sin(theta)
    return np.stack([x, y, z], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class TestOutlierRemoval:
    def test_removes_obvious_outliers(self):
        cloud = _make_body_cloud(500)
        # Add 10 extreme outliers
        outliers = np.array([[100, 100, 100]] * 10, dtype=np.float32)
        noisy = np.concatenate([cloud, outliers])
        cleaned = remove_outliers(noisy, k=10, std_ratio=2.0)
        assert len(cleaned) < len(noisy), "Should remove outliers"
        assert cleaned[:, 0].max() < 50, "Outliers should be gone"

    def test_preserves_most_points(self):
        cloud = _make_body_cloud(1000)
        cleaned = remove_outliers(cloud)
        assert len(cleaned) >= len(cloud) * 0.90, "Should keep ≥90% of clean cloud"


class TestDownsampling:
    def test_exact_count(self):
        cloud = _make_body_cloud(2000)
        result = downsample_random(cloud, 512)
        assert result.shape == (512, 3)

    def test_upsample_when_too_few(self):
        cloud = _make_body_cloud(100)
        result = downsample_random(cloud, 512)
        assert result.shape == (512, 3)

    def test_reproducible_with_seed(self):
        cloud = _make_body_cloud(2000)
        r1 = downsample_random(cloud, 256, seed=42)
        r2 = downsample_random(cloud, 256, seed=42)
        assert np.allclose(r1, r2)


class TestNormalise:
    def test_centred_after_normalise(self):
        cloud = _make_body_cloud(500) + np.array([10, 5, 3])  # offset
        normed, info = normalise(cloud, center=True, scale_to_unit=False)
        assert abs(normed.mean(axis=0)[0]) < 0.01
        assert abs(normed.mean(axis=0)[1]) < 0.01

    def test_unit_height(self):
        cloud = _make_body_cloud(500) * 1.70  # scale to metres
        normed, info = normalise(cloud, center=True, scale_to_unit=True)
        height = normed[:, 1].max() - normed[:, 1].min()
        assert abs(height - 1.0) < 0.05

    def test_info_invertible(self):
        cloud = _make_body_cloud(200) * 1.5 + np.array([1, 0.5, -0.2])
        normed, info = normalise(cloud)
        pipe = PreprocessPipeline()
        recovered = pipe.invert_normalise(normed, info)
        assert np.allclose(recovered, cloud, atol=0.01)


class TestPipeline:
    def test_output_shape(self):
        cloud = _make_body_cloud(5000)
        pipe  = PreprocessPipeline(PreprocessConfig(n_points=1024))
        result, info = pipe(cloud)
        assert result.shape == (1024, 3)

    def test_info_keys(self):
        cloud = _make_body_cloud(1000)
        pipe  = PreprocessPipeline()
        _, info = pipe(cloud)
        assert "offset" in info
        assert "scale" in info
        assert "n_raw" in info


# ---------------------------------------------------------------------------
# Landmark detection
# ---------------------------------------------------------------------------

class TestLandmarkDetector:
    def test_crown_is_highest_point(self):
        cloud = _make_body_cloud(2048)
        normed, _ = normalise(cloud)
        det = LandmarkDetector()
        lm  = det.detect(normed)
        assert lm.crown is not None
        assert lm.crown[1] >= normed[:, 1].max() - 0.01

    def test_scalar_measurements_non_negative(self):
        cloud = _make_body_cloud(2048)
        normed, _ = normalise(cloud)
        det = LandmarkDetector()
        lm  = det.detect(normed)
        for val in lm.as_vector():
            assert val >= 0.0, f"Negative measurement: {val}"

    def test_symmetric_cloud_low_asymmetry(self):
        cloud = _make_body_cloud(2048, seed=7)  # symmetric by construction
        normed, _ = normalise(cloud)
        det = LandmarkDetector()
        lm  = det.detect(normed)
        # Symmetric cloud = arm/leg asymmetry should be small
        assert lm.asymmetry_arm() < 0.20, "Symmetric cloud should have low arm asymmetry"


# ---------------------------------------------------------------------------
# Scan to BodyParams mapping
# ---------------------------------------------------------------------------

class TestScanToParams:
    def _make_landmarks(self, **overrides):
        from research.pointcloud.landmarks import BodyLandmarks
        lm = BodyLandmarks(
            shoulder_width_norm  = overrides.get("shoulder_width_norm",  0.22),
            hip_width_norm       = overrides.get("hip_width_norm",       0.23),
            waist_width_norm     = overrides.get("waist_width_norm",     0.17),
            left_arm_length_norm = overrides.get("left_arm_length_norm", 0.28),
            right_arm_length_norm= overrides.get("right_arm_length_norm",0.28),
            left_leg_length_norm = overrides.get("left_leg_length_norm", 0.48),
            right_leg_length_norm= overrides.get("right_leg_length_norm",0.48),
        )
        return lm

    def test_standard_body_all_zero(self):
        mapper = ScanToParams()
        lm = self._make_landmarks()
        params = mapper.from_landmarks(lm)
        assert int(params.shoulder_width_level) == 0
        assert int(params.hip_level) == 0
        assert int(params.left_upper_arm_step) == 0
        assert int(params.left_leg_step) == 0

    def test_wide_hips_raises_level(self):
        mapper = ScanToParams()
        lm = self._make_landmarks(hip_width_norm=0.30)
        params = mapper.from_landmarks(lm)
        assert int(params.hip_level) >= 2

    def test_arm_asymmetry_detected(self):
        mapper = ScanToParams()
        lm = self._make_landmarks(left_arm_length_norm=0.18,
                                   right_arm_length_norm=0.28)
        params = mapper.from_landmarks(lm)
        assert int(params.left_upper_arm_step) > 0, "Left arm shorter = level > 0"

    def test_threshold_lookup(self):
        assert _threshold_lookup(0.20, "shoulder_width_norm") == 0
        assert _threshold_lookup(0.26, "shoulder_width_norm") >= 1

    def test_network_output_path(self):
        mapper = ScanToParams()
        vector = np.array([0.22, 0.23, 0.17, 0.28, 0.28, 0.48, 0.48], dtype=np.float32)
        params = mapper.from_network_output(vector)
        assert int(params.shoulder_width_level) == 0
