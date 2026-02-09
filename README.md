# Monocular Visual Odometry (ORB)

This project implements a **monocular visual odometry pipeline** using classical computer vision techniques.  
It estimates a camera trajectory from a sequence of RGB images using only visual information.

The project was developed as part of *Navigation Mapping and Localization – Homework 2*.

---

## Problem Setup

- Input: monocular RGB image sequence with timestamps  
- Output: estimated camera trajectory (up to scale)  
- Constraints:
  - No depth information
  - No IMU or additional sensors
  - No camera intrinsics given (estimated from image size)
  - Unknown global scale
  - Frame-to-frame motion only

---

## Pipeline Overview

1. Load image sequence
2. Convert to grayscale and apply CLAHE
3. Extract ORB keypoints and descriptors
4. Match features using KNN + Lowe’s ratio test
5. Estimate Essential matrix using USAC_MAGSAC
6. Recover relative camera pose
7. Apply sanity checks to reject unstable poses
8. Chain poses into a world trajectory
9. Export trajectory in TUM format
10. Smooth translations in post-processing

---

## Feature Extraction

- **ORB** is used for keypoint detection and description  
- **CLAHE** is applied before feature extraction to improve contrast  
- Other filters were tested and removed since they reduced inlier count such as Gaussian Blur or Bilateral Filtering

---

## Feature Matching

- KNN matching with `k = 2`
- Lowe’s ratio test
- Frames with too few matches are skipped

No symmetric matching or optical flow is used to keep the pipeline focused on two-view geometry.

---

## Pose Estimation

- Essential matrix estimated with **USAC_MAGSAC**
- Pose recovered using `cv2.recoverPose`
- Sanity checks applied:
  - Rotation matrix determinant check
  - Finite value check
  - Rotation angle limit (flip guard)

If Essential matrix recovery fails, a **Fundamental matrix fallback** is used and converted to an Essential matrix.

---

## Trajectory Construction

Each valid relative pose is converted to a homogeneous transform and chained:

T_world_curr = T_world_prev × T_prev_curr


The resulting trajectory is in arbitrary scale, as expected for monocular VO.

---

## Visualization

- 2D visualization of keypoints using OpenCV  
- Real-time 3D trajectory visualization using **VisPy**
  - World axes
  - Camera position
  - Camera orientation axes

Visualization is for inspection only and does not affect estimation.

---

## Trajectory Smoothing

- Savitzky–Golay filtering applied **only to translations**
- Rotations are kept unchanged
- Smoothing is done in post-processing

Two trajectories are saved:
- Raw trajectory
- Smoothed trajectory

---

## Output Files

- `traj_estimation.txt` – raw trajectory (TUM format)  
- `traj_estimation_smooth.txt` – smoothed translation trajectory  

Format: timestamp tx ty tz qx qy qz qw

---

## Evaluation

Trajectories were evaluated using the **evo** toolkit with:
- SIM(3) alignment
- Scale correction enabled
- APE and RPE metrics

Rotation safeguards significantly reduced large errors, while smoothing mainly improved translation stability and visual consistency.

---

## How to Run

```bash
python main.py --data_dir path/to/image_sequence


