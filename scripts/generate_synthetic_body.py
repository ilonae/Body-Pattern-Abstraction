"""
generate_synthetic_body.py
--------------------------
Generate synthetic body point clouds for development and testing.

No real scan data needed to start. Produces anatomically plausible
(but not photorealistic) ellipsoidal body approximations with
configurable parameters.

Usage:
    python scripts/generate_synthetic_body.py --output data/samples/
    python scripts/generate_synthetic_body.py --profile wheelchair --n 5
    python scripts/generate_synthetic_body.py --list-profiles

Output: .npy files in the output directory, plus a manifest.json.
"""

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Body shape parameters (in metres, Y-up)
# ---------------------------------------------------------------------------

@dataclass
class SyntheticBodyConfig:
    """
    Anatomically-inspired ellipsoid parameters for a synthetic body.
    All measurements in metres.
    """
    label: str = "standard"

    # Overall
    height_m: float = 1.70        # total height
    weight_kg: float = 70.0       # influences torso/hip volumes

    # Torso (ellipsoid)
    shoulder_width: float = 0.44
    chest_depth: float    = 0.22
    chest_height: float   = 0.30  # vertical extent of chest region
    hip_width: float      = 0.38
    hip_depth: float      = 0.22
    hip_height: float     = 0.22

    # Arms (cylinders)
    upper_arm_radius: float = 0.045
    lower_arm_radius: float = 0.035
    upper_arm_length: float = 0.30
    lower_arm_length: float = 0.25
    arm_asymmetry: float    = 0.0   # 0=symmetric, >0 = left arm shorter by X%

    # Legs (cylinders)
    thigh_radius: float  = 0.075
    calf_radius: float   = 0.050
    thigh_length: float  = 0.42
    calf_length: float   = 0.38
    leg_asymmetry: float = 0.0    # left leg shorter by X%

    # Head
    head_radius: float = 0.095
    neck_height: float = 0.08

    # Noise (scan realism)
    noise_std: float = 0.003   # metres of Gaussian surface noise

    # Sampling
    n_points: int = 4096
    seed: int     = 42


# ---------------------------------------------------------------------------
# Profile presets
# ---------------------------------------------------------------------------

PROFILES: dict[str, SyntheticBodyConfig] = {
    "standard": SyntheticBodyConfig(label="standard"),

    "wheelchair": SyntheticBodyConfig(
        label="wheelchair",
        hip_width=0.42,          # seated hip spread
        hip_depth=0.26,          # forward compression
        thigh_length=0.30,       # shorter — seated, not hanging
        calf_length=0.30,
        height_m=1.65,
    ),

    "prosthesis_left_lower": SyntheticBodyConfig(
        label="prosthesis_left_lower",
        leg_asymmetry=0.35,      # left leg 35% shorter (residual limb)
        thigh_radius=0.065,      # slightly reduced on affected side
    ),

    "prosthesis_left_upper": SyntheticBodyConfig(
        label="prosthesis_left_upper",
        arm_asymmetry=0.40,
        upper_arm_radius=0.035,  # reduced on affected side
    ),

    "dwarfism_proportional": SyntheticBodyConfig(
        label="dwarfism_proportional",
        height_m=1.30,
        shoulder_width=0.38,
        hip_width=0.33,
        upper_arm_length=0.22,
        lower_arm_length=0.18,
        thigh_length=0.30,
        calf_length=0.27,
    ),

    "dwarfism_disproportional": SyntheticBodyConfig(
        label="dwarfism_disproportional",
        height_m=1.25,
        shoulder_width=0.42,     # near-standard torso
        hip_width=0.37,
        chest_height=0.32,
        upper_arm_length=0.18,   
        lower_arm_length=0.14,
        thigh_length=0.25,       # short limbs
        calf_length=0.22,
    ),

    "plus_size": SyntheticBodyConfig(
        label="plus_size",
        weight_kg=110.0,
        shoulder_width=0.52,
        chest_depth=0.30,
        hip_width=0.52,
        hip_depth=0.32,
        thigh_radius=0.095,
        upper_arm_radius=0.060,
    ),
}


# ---------------------------------------------------------------------------
# Point cloud generator
# ---------------------------------------------------------------------------

def _sample_ellipsoid(cx, cy, cz, rx, ry, rz, n, rng) -> np.ndarray:
    """Sample n points uniformly on the surface of an ellipsoid."""
    phi   = rng.uniform(0, 2 * np.pi, n)
    theta = np.arccos(rng.uniform(-1, 1, n))
    x = rx * np.sin(theta) * np.cos(phi) + cx
    y = ry * np.cos(theta) + cy
    z = rz * np.sin(theta) * np.sin(phi) + cz
    return np.stack([x, y, z], axis=1)


def _sample_cylinder(cx, cy_bot, cz, radius, length, n, rng) -> np.ndarray:
    """Sample n points on the curved surface of an upright cylinder."""
    theta = rng.uniform(0, 2 * np.pi, n)
    y     = rng.uniform(cy_bot, cy_bot + length, n)
    x = radius * np.cos(theta) + cx
    z = radius * np.sin(theta) + cz
    return np.stack([x, y, z], axis=1)


def generate_body(cfg: SyntheticBodyConfig) -> np.ndarray:
    """
    Generate a synthetic body point cloud from a SyntheticBodyConfig.
    Returns (N, 3) float32 array, Y-up, centred at foot level (y=0).
    """
    rng = np.random.default_rng(cfg.seed)
    parts = []
    h = cfg.height_m

    # Weight → volume scale (crude)
    vol_scale = (cfg.weight_kg / 70.0) ** (1/3)

    # --- Legs ---
    leg_base_y = 0.0
    thigh_y = leg_base_y
    calf_y  = thigh_y + cfg.thigh_length
    n_leg   = cfg.n_points // 8

    for side, asymmetry in [(0.12, cfg.leg_asymmetry), (-0.12, 0.0)]:
        tl = cfg.thigh_length * (1.0 - asymmetry)
        cl = cfg.calf_length  * (1.0 - asymmetry)
        parts.append(_sample_cylinder(side, thigh_y, 0.0, cfg.thigh_radius * vol_scale, tl,   n_leg, rng))
        parts.append(_sample_cylinder(side, calf_y,  0.0, cfg.calf_radius,              cl,   n_leg, rng))

    # --- Hips & torso ---
    hip_y    = cfg.thigh_length + cfg.calf_length * 0.0  # simplified: at top of thighs
    torso_cy = hip_y + cfg.hip_height * 0.5
    n_torso  = cfg.n_points // 4

    parts.append(_sample_ellipsoid(0, torso_cy, 0,
                                   cfg.hip_width * 0.5 * vol_scale,
                                   cfg.hip_height * 0.5,
                                   cfg.hip_depth * 0.5 * vol_scale,
                                   n_torso, rng))

    chest_cy = torso_cy + cfg.hip_height * 0.5 + cfg.chest_height * 0.5
    parts.append(_sample_ellipsoid(0, chest_cy, 0,
                                   cfg.shoulder_width * 0.5 * vol_scale,
                                   cfg.chest_height * 0.5,
                                   cfg.chest_depth * 0.5 * vol_scale,
                                   n_torso, rng))

    # --- Arms ---
    shoulder_y = chest_cy + cfg.chest_height * 0.4
    n_arm = cfg.n_points // 10

    for side, asymmetry in [(1, cfg.arm_asymmetry), (-1, 0.0)]:
        arm_x   = side * cfg.shoulder_width * 0.55
        ual     = cfg.upper_arm_length * (1.0 - asymmetry)
        lal     = cfg.lower_arm_length * (1.0 - asymmetry)
        uar     = cfg.upper_arm_radius * (1.0 - asymmetry * 0.5)
        elbow_y = shoulder_y - ual
        parts.append(_sample_cylinder(arm_x, elbow_y,   0, uar,              ual, n_arm, rng))
        parts.append(_sample_cylinder(arm_x, elbow_y - lal, 0, cfg.lower_arm_radius, lal, n_arm, rng))

    # --- Head ---
    neck_y = shoulder_y + cfg.neck_height
    head_y = neck_y + cfg.head_radius
    n_head = cfg.n_points // 12
    parts.append(_sample_ellipsoid(0, head_y, 0,
                                   cfg.head_radius,
                                   cfg.head_radius * 1.2,
                                   cfg.head_radius * 0.9,
                                   n_head, rng))

    # --- Combine ---
    cloud = np.concatenate(parts, axis=0).astype(np.float32)

    # Add surface noise
    if cfg.noise_std > 0:
        cloud += rng.normal(0, cfg.noise_std, cloud.shape).astype(np.float32)

    # Final downsample to n_points
    idx = rng.choice(len(cloud), size=cfg.n_points, replace=len(cloud) < cfg.n_points)
    return cloud[idx]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic body point clouds")
    parser.add_argument("--output",  default="data/samples", help="Output directory")
    parser.add_argument("--profile", default=None,
                        help="Single profile to generate. Omit for all profiles.")
    parser.add_argument("--n",       type=int, default=1,
                        help="Number of samples per profile (with different seeds)")
    parser.add_argument("--n-points", type=int, default=4096,
                        help="Points per cloud")
    parser.add_argument("--list-profiles", action="store_true",
                        help="Print available profiles and exit")
    args = parser.parse_args()

    if args.list_profiles:
        print("Available profiles:")
        for k, v in PROFILES.items():
            print(f"  {k:35s}  h={v.height_m}m  w={v.weight_kg}kg")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles_to_gen = ([args.profile] if args.profile else list(PROFILES.keys()))
    manifest = []

    for profile_name in profiles_to_gen:
        if profile_name not in PROFILES:
            print(f"Unknown profile: {profile_name}. Use --list-profiles.")
            continue
        cfg = PROFILES[profile_name]
        cfg.n_points = args.n_points

        for i in range(args.n):
            cfg.seed = i * 137 + 42  # deterministic but varied
            cloud = generate_body(cfg)
            fname = f"{profile_name}_{i:03d}.npy"
            fpath = output_dir / fname
            np.save(fpath, cloud)
            manifest.append({"file": fname, "profile": profile_name,
                              "seed": cfg.seed, "n_points": len(cloud)})
            print(f"  Saved: {fpath}  ({len(cloud)} pts)")

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")
    print(f"Total files: {len(manifest)}")


if __name__ == "__main__":
    main()
