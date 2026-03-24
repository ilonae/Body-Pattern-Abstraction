# Algorithms & Mathematical Foundations

This document explains the mathematical calculations for every major processing step in this repo, where each idea comes from (paper, heuristic  or standard practice), and how this project differs from and relates to the published research it builds on.

---

## Table of Contents

1. [How scan_to_params.py fits in the pipeline](#1-how-scan_to_paramspy-fits-in-the-pipeline)
2. [Preprocessing mathematics](#2-preprocessing-mathematics)
3. [Heuristic landmark extraction](#3-heuristic-landmark-extraction)
4. [PointNet encoder — the core idea](#4-pointnet-encoder--the-core-idea)
5. [How this project differs from the papers](#5-how-this-project-differs-from-the-papers)
6. [PointNet vs NeuralTailor — a direct comparison](#6-pointnet-vs-neuraltailor--a-direct-comparison)
7. [Using public datasets (Kaggle, CAESAR, etc.)](#7-using-public-datasets-kaggle-caesar-etc)

---

## 1. How `scan_to_params.py` fits in the pipeline

### The translation problem

The body-pattern-research differs from diverse-body-pattern-adjuster, which is meant to be complementary for the further prototypical approach - in the following matter:

```
body-pattern-research                diverse-body-pattern-adjuster
─────────────────────────────        ──────────────────────────────
Input:  3D point cloud (N × 3)  ->  Input:  BodyParams (14 discrete levels)
Output: BodyLandmarks (mm)      ->  Output: mm deltas per seam axis → DXF
```

`scan_to_params.py` is the **translator**. It takes a `BodyLandmarks` object (raw mm measurements extracted from the cloud) and maps each measurement into the Level 0–4 vocabulary that the adjuster understands.

### Why not pass raw mm values to the adjuster?

Raw measurements (e.g. "hip circumference = 112 cm") are:

- **Brand-dependent** — a Yohji Yamamoto size 44 and a Chanel size 16 encode very different baseline assumptions
- **Privacy-sensitive** — a vector of exact body measurements is personally identifiable
- **Fragile** — noisy scan data creates noisy floats; downstream logic becomes sensitive to small errors

The Level system solves all three: each axis is bucketed into 5 categories (NONE=0 up to MAX=4) relative to a *population median baseline*. The adjuster's rules engine maps levels to mm deltas for that specific pattern baseline, not the person's absolute body.

### What `from_landmarks()` actually does

```python
params.hip_level = _quantise(landmarks.hip_width_mm, THRESHOLDS["hip_width_mm"])
```

`_quantise(value, thresholds)` does:

```
thresholds = [t1, t2, t3, t4]   (4 breakpoints -> 5 buckets)

value < t1  ->  Level.NONE  (within ~1 SD of population mean)
value < t2  ->  Level.LOW
value < t3  ->  Level.MID
value < t4  ->  Level.HIGH
else        ->  Level.MAX
```

The breakpoints in `THRESHOLDS` are currently **heuristic** — derived from the standard German/EU size chart statistics (DIN EN 13402) and literature on clothing adjustment for disability (Watkins 1995, Fan et al. 2004). They  should be replaced with data-driven percentiles once a labelled dataset is available.

### Asymmetry axes

For the four asymmetry fields (`left_upper_arm_step`, `right_upper_arm_step`, `left_leg_step`, `right_leg_step`), the mapper uses the *difference* between sides rather than an absolute measurement:

```python
arm_diff = abs(landmarks.left_upper_arm_mm - landmarks.right_upper_arm_mm)
params.left_upper_arm_step = _quantise(arm_diff, THRESHOLDS["arm_asymmetry_mm"])
```

This means a prosthesis user's missing limb produces a large asymmetry value (-> Level.MAX) while a standard user's minor natural asymmetry produces Level.NONE or Level.LOW.

---

## 2. Preprocessing

File: `research/pointcloud/preprocess.py`

### 2a. Statistical outlier removal

**What it does:** removes points that are far from their local neighbourhood — usually scan artefacts, stray reflections, or erroneous depth readings.

**Algorithm:**

```
For each point p_i:
    find k nearest neighbours (default k=20)
    compute mean distance d_i = mean(dist(p_i, neighbour_j))

global_mean = mean(d_i for all i)
global_std  = std(d_i for all i)

keep p_i  iff  d_i < global_mean + std_ratio × global_std
```

**Adapted from th**e standard SOR (Statistical Outlier Removal) filter from PCL (Point Cloud Library, Rusu & Cousins 2011). It is also described in Open3D documentation. The implementation here reproduces it in pure NumPy/SciPy so there is no dependency on Open3D or PCL.

Both k=20 and std_ratio=2.0  are the PCL defaults, which work well for body scans at typical density (10k–100k points). Tighter values (k=10, std_ratio=1.5) remove more noise but risk removing real anatomy on thin limbs.

### 2b. Random downsampling

**What it does:** reduces cloud to exactly `n_points` by random uniform sampling without replacement.

**Why:** Deep learning models (PointNet and descendants) require a fixed-size input tensor. The standard in the literature is 1024–2048 points for body-scale objects. NeuralTailor uses 2000 points; PointNet++ experiments use 1024. Here we default to 2048.

Used in standard practice — e.g. PointNet (Qi et al. 2017), NeuralTailor (Korosteleva & Koltun 2022).

### 2c. Voxel downsampling

**What it does:** divides 3D space into a grid of cubes of side `voxel_size`, then replaces all points inside each cube with their centroid.

```
grid_index(p) = floor(p / voxel_size)

for each unique grid_index:
    keep centroid of all points in that voxel
```

Unlike random, voxel downsampling preserves the *spatial distribution* — it will not accidentally under-sample a dense area (like the torso) and over-sample a sparse area (like the ankles). It is preferred when feeding geometric algorithms that depend on even spatial coverage, such as FPS (farthest point sampling) in PointNet++.

Usually used in standard in point cloud processing; described in PCL, Open3D, and used throughout the PointNet++ paper.

### 2d. Normalisation

**What it does:** centres the cloud at the origin and scales it to fit inside a unit sphere.

```
centroid = mean(points, axis=0)          # translate to origin
points   = points - centroid

max_dist = max(||p|| for p in points)    # furthest point from origin
points   = points / max_dist             # scale to unit sphere
```

**Why unit sphere (not unit cube)?** PointNet's architecture assumes the network learns geometry that is invariant to scale. Normalising to a unit sphere means the network's weight magnitudes are consistent across training examples of very different real-world sizes (a child vs. an adult). The unit sphere is used in the original PointNet paper and all derivatives.


### 2e. Canonical alignment (PCA-based)

**What it does:** rotates the cloud so the person always faces the same direction (principal axes aligned with X/Y/Z).

```
C = covariance matrix of points (3×3)
eigenvalues, eigenvectors = eig(C)

sort eigenvectors by descending eigenvalue:
  - largest eigenvalue  -> Y axis (height, tallest dimension)
  - second largest      -> X axis (width)
  - smallest            -> Z axis (depth, front–back)

R = [eigvec_x | eigvec_y | eigvec_z]   (rotation matrix)
points = points @ R.T
```

**Why is this important?** If someone is scanned slightly rotated, or lying down vs. standing, the raw point cloud will be rotated. Without alignment, the network would see completely different coordinate patterns for the same body shape just because of scan orientation. PCA alignment removes this rotational ambiguity. As a standard technique, it is used in ShapeNet preprocessing, PointNet experiments, and ModelNet.

**Caveat:** PCA alignment is symmetric — it cannot tell "facing forward" from "facing backward". A more robust solution uses a reference landmark (e.g. nose tip) to enforce a consistent facing direction; this is planned to be implemented in Phase 2.

---

## 3. Heuristic landmark extraction

File: `research/pointcloud/landmarks.py`

### The core idea

No trained model is used to find landmarks — instead, we use the *geometric structure of the point cloud* itself. A standing human body has predictable spatial relationships between body parts that can be exploited with simple coordinate queries.

This is a **heuristic approach**: it is fast, interpretable, and requires no training data, but it is less robust than a learned detector (especially for unusual postures or partial scans). The plan is to replace this with a neural approach - `LandmarkNet` - once training data exists.

### Height measurement

```python
height_mm = (points[:, 1].max() - points[:, 1].min()) * scale_factor
```

Y-axis extent (after canonical alignment) equals body height. We multiply by `scale_factor` to convert from normalised units back to millimetres.

**Assumption:** the person is standing upright and the cloud has been canonically aligned (Y = vertical axis).

### Shoulder width

```python
# Take the top 15% of points (above shoulder line)
top_mask = points[:, 1] > np.percentile(points[:, 1], 85)
shoulder_pts = points[top_mask]
shoulder_width_mm = (shoulder_pts[:, 0].max() - shoulder_pts[:, 0].min()) * scale_factor
```

The widest extent in X within the upper region of the body approximates shoulder width. The 85th-percentile cutoff was chosen empirically to exclude the neck and head while capturing the shoulder tips.

**Limitation:** this overestimates slightly if the person has raised arms - hence a T-Position measurement would be most suitable. It also cannot distinguish between left and right shoulder height.

### Hip width

```python
# Take the middle band (40th–60th percentile in Y)
mid_mask = (points[:, 1] > np.percentile(points[:, 1], 40)) & \
           (points[:, 1] < np.percentile(points[:, 1], 60))
hip_pts = points[mid_mask]
hip_width_mm = (hip_pts[:, 0].max() - hip_pts[:, 0].min()) * scale_factor
```

The widest X-extent at the midpoint of the body approximates hip width. The 40–60% band is a heuristic calibrated against standard anthropometric reference data (DIN EN 13402).

### Inseam length

```python
# Bottom 50% of the body
lower_mask = points[:, 1] < np.percentile(points[:, 1], 50)
lower_pts = points[lower_mask]
# If legs visible, X extent here should be narrower than hip
inseam_mm = (np.percentile(lower_pts[:, 1], 95) - lower_pts[:, 1].min()) * scale_factor
```

Approximated as the vertical extent of the lower half of the body. This is a rough estimate — proper inseam detection would require separating the two legs (e.g. by clustering in the X-Z plane at knee height). Again, `LandmarkNet` will make more sense for this measurement.

### Upper arm circumference (per side)

```python
# Upper-arm band: 70th–80th percentile in Y, left side (x < 0) and right side (x > 0)
arm_mask = (points[:, 1] > np.percentile(points[:, 1], 70)) & \
           (points[:, 1] < np.percentile(points[:, 1], 80))
left_arm  = points[arm_mask & (points[:, 0] < 0)]
right_arm = points[arm_mask & (points[:, 0] > 0)]
left_upper_arm_mm  = (left_arm[:, 0].max()  - left_arm[:, 0].min())  * scale_factor
right_upper_arm_mm = (right_arm[:, 0].max() - right_arm[:, 0].min()) * scale_factor
```

**This is the trickiest heuristic.** At the upper arm band, X-extent per side roughly corresponds to arm thickness. The asymmetry between left and right detects prosthesis situations (one side is significantly narrower). A proper circumference measurement requires computing a convex hull or cross-sectional slice, which is deferred to Phase 2.

### Why `max() - min()` instead of `.ptp()`?

NumPy 2.0 removed the `.ptp()` method from ndarray. The replacement is:

```python
# Old (breaks on NumPy ≥ 2.0):
extent = arr.ptp()

# New (correct):
extent = arr.max() - arr.min()
```

`.ptp()`- for "peak to peak"- was just a shorthand for max-min. We use the explicit form throughout.

---

## 4. PointNet encoder — the core idea

File: `research/models/encoder.py`

### The permutation invariance problem

A point cloud is a *set* of 3D coordinates — there is no canonical ordering. With 2048 points describing a torso, shuffling them in 2048! different ways, they would still represent the same geometry. A naive approach of feeding them to a CNN or RNN as a sequence would make the network sensitive to this arbitrary ordering.

**PointNet's insight** (Qi et al., CVPR 2017): use a **symmetric function** — one whose output is independent of input order. The symmetric function they choose is **element-wise max pooling**.

### The architecture

```
Input: N × 3 (N points, xyz each)
       │
       │  shared MLP (same weights applied to each point independently)
       │  3 -> 64 -> 128 -> 1024
       │
  N × 1024 (per-point feature vectors)
       │
       │  global max pooling (take the max across all N points, per feature)
       │
  1 × 1024 (global shape descriptor)
       │
       │  MLP head (for regression/classification)
       │
  output (e.g. 7 body measurements)
```

The **shared MLP** is implemented as `Conv1d` with kernel size 1 — this is equivalent to applying the same dense layer to each of the N points independently.

**Why does max pooling give permutation invariance?**

```
max(f(p1), f(p2), ..., f(pN))  ==  max(f(p3), f(p1), ..., f(pN))
```

Max-pooling over the N dimension produces the same result regardless of point order. This is the mathematical foundation of the whole architecture.

### PointNet++ extension

`PointNetPlusPlusEncoder` adds a **hierarchical grouping** stage before the global pooling:

```
Farthest Point Sampling (FPS): select M "seed" points spread evenly across the cloud
Ball query: for each seed, find all points within radius r
Local PointNet: apply a small PointNet to each local neighbourhood -> local feature
Global PointNet: apply PointNet to the M local features -> global feature
```

This lets the network learn **local geometry** (e.g. the curvature of a shoulder) before combining it into a global descriptor. The plain PointNet lacks this — it treats all N points as a flat set, which makes it weaker at detecting small local deformations (like the shape difference between a straight elbow and a bent one). Adapted from PointNet++ (Qi et al., NeurIPS 2017).

---

## 5. How this project differs from the papers

| Aspect                          | PointNet (2017)                | NeuralTailor (2022)                   | SewPCT (2025)              | This project                                    |
| ------------------------------- | ------------------------------ | ------------------------------------- | -------------------------- | ----------------------------------------------- |
| **Input**                 | Point cloud (any object)       | Body mesh to point cloud              | Body mesh to point cloud  | 3D body scan cloud                              |
| **Output**                | Class label or per-point label | 2D sewing pattern panels              | 2D panels + stitch labels  | BodyParams levels, then pattern via adjuster   |
| **Architecture**          | Shared MLP + max pool          | PointNet encoder + panel decoder      | Point Cloud Transformer    | PointNet encoder + regression head              |
| **Training data**         | ModelNet (synthetic shapes)    | Synthetic clothed bodies (simulation) | Synthetic body meshes      | None yet — currently heuristic                 |
| **Pose requirement**      | None (general objects)         | T-pose or A-pose                      | T-pose                     | Standing canonical pose (current)               |
| **Disability body types** | Not addressed                  | Not addressed                         | Not addressed              | **Core design requirement**               |
| **Output format**         | Labels                         | SVG/JSON panels                       | SVG/JSON panels + stitches | DXF (Gerber/Lectra/CLO compatible)              |
| **XAI / explainability**  | None                           | None                                  | None                       | explain() layer traces every mm delta           |
| **Privacy model**         | N/A                            | N/A                                   | N/A                        | Level abstraction — no raw measurements stored |

### Key architectural difference from NeuralTailor

NeuralTailor is an **end-to-end reconstruction** system: it takes a point cloud and directly outputs a complete sewing pattern from scratch. It must learn everything  from examples — body shape, garment topology, seam placement.

This project takes a **decomposed** approach:


`body-pattern-research` extracts *body shape parameters* from the scan (not garment shape) into level abstractions.
(Further, the private repo `diverse-body-pattern-adjuster` takes these abstractions to work with special diverse body types.)

This is how a professional tailor would work: Measuring the person, then altering a standard block pattern rather than drafting from scratch. It requires less training data (adjustments are smaller and more structured than full reconstruction), and more explainable (each adjustment has a named cause), and integrates with existing pattern CAD workflows.

### What we borrow from each paper

**From PointNet:**

- The Conv1d shared-MLP architecture (encoder.py)
- The global max-pooling symmetry trick
- Unit-sphere normalisation
- The concept of a global shape descriptor for downstream tasks

**From PointNet++:**

- The hierarchical encoder structure in `PointNetPlusPlusEncoder`
- FPS + ball-query grouping for local feature extraction

**From NeuralTailor:**

- The idea that 3D body scan to 2D pattern is a learnable mapping
- Proof that synthetic body data (simulation) can train real-world transferable models
- The 2000-point downsampling convention
- Awareness that point clouds need canonical pose alignment before processing

**From SewPCT:**

- Evidence that transformer attention on point features outperforms pure max-pooling for fine-grained panel boundary prediction (motivates future PointNet++ or transformer upgrade)

---

## 6. PointNet vs NeuralTailor — a direct comparison

```
PointNet                              NeuralTailor
────────────────────────────────      ────────────────────────────────────────
PURPOSE: general shape recognition    PURPOSE: body-specific pattern generation
INPUT:   any point cloud              INPUT:   body point cloud (T-pose assumed)
OUTPUT:  shape class or point labels  OUTPUT:  2D sewing panels (full garment)
TRAINED ON: ModelNet40 (CAD objects)  TRAINED ON: synthetic sim data (SMPLicit)
PANEL KNOWLEDGE: none                 PANEL KNOWLEDGE: explicit panel decoder
SCALE: 1k points typical              SCALE: 2k points typical
STITCHING: not modelled               STITCHING: stitch graph predicted
```

**PointNet** is a *generic* point cloud encoder. It does not know what a sewing pattern is — it just learns to map point sets to labels. A PointNet encoder could similarly be used to classify chairs or aeroplanes.

**NeuralTailor** is a *domain-specific* system. Its decoder is purpose-built to output garment panels — it knows that a trouser has a front and back leg piece, an inseam and outseam, and so on. The architecture is more complex because the output space is more structured - which similarly adds complexity.

**This project's LandmarkNet** is closer to PointNet in philosophy — a general encoder that regresses scalar measurements — but uses those measurements in a *structured adjustment* framework (the adjuster), rather than trying to reconstruct a full pattern from scratch. Think of it as PointNet doing anthropometry, not pattern making.

---

## 7. Usage of public datasets (Kaggle, CAESAR, etc.)

### Available datasets

| Dataset                                    | Source                                                        | Format                       | Notes                                                                            |
| ------------------------------------------ | ------------------------------------------------------------- | ---------------------------- | -------------------------------------------------------------------------------- |
| **CAESAR**                           | Civilian American and European Surface Anthropometry Resource | PLY/OBJ meshes               | ~4400 bodies, standing T-pose, labelled landmarks — best match for this project |
| **SMPL/SMPL-X**                      | Max Planck Institute                                          | Parametric mesh (not a scan) | Synthetic; controllable shape parameters; good for training data generation      |
| **FAUST**                            | MPI / Bosphorus                                               | OBJ meshes                   | 300 scans × 10 poses; useful for pose robustness testing                        |
| **Kaggle: 3D Human Pose Estimation** | Various                                                       | CSV / NPY                    | Lower resolution; useful for landmark detection prototyping                      |
| **THuman2.0**                        | Tsinghua                                                      | OBJ + texture                | 526 high-quality bodies; good for fine-grained fitting                           |

### Quick start with any PLY/OBJ file

```python
from research.pointcloud.loader import load_pointcloud
from research.pointcloud.preprocess import PreprocessPipeline
from research.pointcloud.landmarks import LandmarkDetector
from research.mapping.scan_to_params import ScanToParams

# Load — supports .ply, .obj, .npy, .npz, .xyz
pts = load_pointcloud("path/to/your/scan.ply")

# Preprocess (outlier removal -> downsample -> normalise -> align)
pts_clean, info = PreprocessPipeline(n_points=2048)(pts)

# Extract heuristic landmarks
landmarks = LandmarkDetector(scale_factor=1800.0).detect(pts_clean)
# scale_factor should be approximate height in mm
# 1800.0 = 180 cm is a reasonable default for adult standing scans

# Map to BodyParams levels
params = ScanToParams().from_landmarks(landmarks, height_m=info.get("est_height_m", 1.75))

print(params)
```

### Using Kaggle SMPL synthetic data

SMPL meshes can be converted to point clouds by sampling from the mesh surface:

```python
import numpy as np

# If you have vertex positions from an OBJ/SMPL mesh:
def mesh_to_pointcloud(vertices, n_points=2048):
    """Random sample from mesh vertices to get a point cloud."""
    idx = np.random.choice(len(vertices), size=n_points, replace=len(vertices) < n_points)
    return vertices[idx].astype(np.float32)

# Then proceed with the normal pipeline:
pts = mesh_to_pointcloud(smpl_vertices)
pts_clean, _ = PreprocessPipeline(n_points=2048)(pts)
```

### Scale factor note

After normalisation, all clouds are in the range [-1, 1]. The `LandmarkDetector.scale_factor` converts back to millimetres. For a standing person of ~175 cm:

```
scale_factor ≈ real_height_mm / normalised_height_extent
             ≈ 1750 / 1.0  (unit sphere: extent ≈ 1.0 in Y)
             = 1750
```

If you do not know the real height, use 1750.0 as a default and pass `height_m=None` to `ScanToParams.from_landmarks()` — it will skip height-dependent normalisation and use relative proportions only.

### Threshold calibration warning

The current `THRESHOLDS` in `scan_to_params.py` were calibrated against **synthetic ellipsoidal bodies**, not real scans. Real scan data will have different proportional relationships. Until `LandmarkNet` is trained on labelled real-world data:

- Expect most real-scan bodies to map to `Level.HIGH` or `Level.MAX` on several axes (the thresholds are too tight)
- You can override thresholds by subclassing `ScanToParams`:

```python
class MyCalibrated(ScanToParams):
    THRESHOLDS = {
        "hip_width_mm":      [180, 220, 260, 300],   # your calibrated values
        "shoulder_width_mm": [160, 200, 240, 280],
        # ...
    }
```

- Track calibration data in `data/calibration/` (gitignored by default)
