import os
import glob
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from vispy import scene
from vispy.scene import visuals
from vispy.visuals.transforms import MatrixTransform

# Helpers

def build_k_from_image_size(width: int, height: int) -> np.ndarray:
    """
    Estimate the camera intrinsics matrix K from image size:
    - fx = fy = focal ~ max(width, height)
    - cx, cy at image center
    """
    f = float(max(width, height))
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype=np.float64)
    return K

def to_homogeneous_transform(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Build a 4x4 homogeneous transform from rotation (3x3) and translation (3,).
    """
    t = t.reshape(3, 1)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = r
    T[:3, 3:] = t
    return T

def rotation_matrix_to_quat_xyzw(R: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Convert a 3x3 rotation matrix to quaternion (x, y, z, w).
    """
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22

    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S

    return float(qx), float(qy), float(qz), float(qw)


# Frame class
@dataclass
class Frame:
    frame_id: int
    image_bgr: np.ndarray
    timestamp: float
    keypoints: Optional[List[cv2.KeyPoint]] = None
    descriptors: Optional[np.ndarray] = None
    pose: Optional[np.ndarray] = None  # 4x4, world_T_cam

    def extract_features(self, detector) -> None:
        gray = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2GRAY)

        # CLAHE preprocessing
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )
        gray = clahe.apply(gray)

        kps, desc = detector.detectAndCompute(gray, None)
        self.keypoints = kps
        self.descriptors = desc




# Visual Odometry class
class VisualOdometry:
    def __init__(self, k: np.ndarray):
        self.k = k.astype(np.float64)

        # Pose: world_T_cam (we start at identity)
        self.world_T_cam = np.eye(4, dtype=np.float64)

        # Keep previous frame
        self.prev_frame: Optional[Frame] = None

        self.detector = cv2.ORB_create(
            nfeatures = 1500,
            scaleFactor = 1.2,
            nlevels = 8,
            edgeThreshold = 31,
            firstLevel = 0,
            WTA_K = 2,
            patchSize = 31,
            fastThreshold = 20
        )
        
        # Create the BFMatcher with Hamming distance for ORB
        # Cross-check is False because we'll do Lowe's ratio test instead
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Trajectory history
        self.traj_points = [(0.0, 0.0, 0.0)]  # (x, y, z) in arbitrary scale
        
        # TUM trajectory log: list of (timestamp, 4x4 world_T_cam)
        self.tum_poses: List[Tuple[float, np.ndarray]] = []

        # Tunable Parameters
        self.ratio_thresh = 0.7
        self.min_matches = 12
        self.min_inliers = 10
        
        self.ransac_threshold = 1.0  # pixels
        self.ransac_prob = 0.999

    def process(self, frame: Frame) -> Tuple[bool, dict]:
        """
        Orchestrates VO steps, but each step is encapsulated in a method.
        Returns: (success, debug_info)
        """
        debug = self._init_debug(frame)

        # Step 1: Extract features
        self._extract_features(frame, debug)

        # First frame init
        if self.prev_frame is None:
            self._init_first_frame(frame, debug)
            return True, debug

        # Guard: descriptors must exist
        if not self._has_descriptors(self.prev_frame, frame, debug):
            self.prev_frame = frame
            return False, debug

        # Step 2: Match descriptors (Lowe ratio)
        good_matches = self._match(self.prev_frame, frame, debug)
        if good_matches is None:
            self.prev_frame = frame
            return False, debug

        # Step 3: Build matched point arrays
        pts_prev, pts_curr = self._matched_points(self.prev_frame, frame, good_matches)

        # Step 4: Estimate Essential matrix (MAGSAC)
        E, mask = self._estimate_essential(pts_prev, pts_curr, debug)
        if E is None or mask is None:
            self.prev_frame = frame
            return False, debug

        # Step 5: Recover pose and build relative transform prev->curr
        T_prev_to_curr = self._recover_T_prev_to_curr(E, pts_prev, pts_curr, mask, debug)
        if T_prev_to_curr is None:
            self.prev_frame = frame
            return False, debug

        # Step 6: Accumulate world pose
        self._accumulate_pose(frame, T_prev_to_curr)

        # Step 7: Update trajectory
        self._append_traj_point()

        self.prev_frame = frame
        debug["status"] = "ok"
        return True, debug
    
    
    # Encapsulated Steps
    
    def _init_debug(self, frame: Frame) -> dict:
        return {
            "frame_id": frame.frame_id,
            "timestamp": frame.timestamp,
            "num_keypoints": 0,
            "num_matches": 0,
            "num_inliers": 0,
            "status": "init"
        }
        
    # Step 1: Extract features using ORB
    def _extract_features(self, frame: Frame, debug: dict) -> None:
        frame.extract_features(self.detector)
        debug["num_keypoints"] = 0 if frame.keypoints is None else len(frame.keypoints)
        
    # Step 1b: Wrapper to call from process()
    def _init_first_frame(self, frame: Frame, debug: dict) -> None:
        frame.pose = self.world_T_cam.copy()
        self.tum_poses.append((frame.timestamp, frame.pose.copy()))
        self.prev_frame = frame
        debug["status"] = "first_frame"

    # Step 1c: Check if descriptors exist before matching
    def _has_descriptors(self, prev: Frame, curr: Frame, debug: dict) -> bool:
        if prev.descriptors is None or curr.descriptors is None:
            debug["status"] = "no_descriptors"
            return False
        return True

    # Step 2: Match descriptors using Lowe's ratio test
    def _match(self, prev: Frame, curr: Frame, debug: dict) -> Optional[List[cv2.DMatch]]:
        knn = self.matcher.knnMatch(prev.descriptors, curr.descriptors, k=2)

        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio_thresh * n.distance:
                good.append(m)

        debug["num_matches"] = len(good)

        if len(good) < self.min_matches:
            debug["status"] = "too_few_matches"
            return None

        return good

    # Step 3: Build matched point arrays for Essential matrix estimation
    def _matched_points(self, prev: Frame, curr: Frame, matches: List[cv2.DMatch]) -> Tuple[np.ndarray, np.ndarray]:
        pts_prev = np.float64([prev.keypoints[m.queryIdx].pt for m in matches])
        pts_curr = np.float64([curr.keypoints[m.trainIdx].pt for m in matches])
        return pts_prev, pts_curr

    # Step 4: Estimate Essential matrix using MAGSAC
    
    def _estimate_essential(self, pts_prev: np.ndarray, pts_curr: np.ndarray, debug: dict):
        E, mask = cv2.findEssentialMat(
            pts_prev, pts_curr,
            self.k,
            method=cv2.USAC_MAGSAC,
            prob=self.ransac_prob,
            threshold=self.ransac_threshold
        )

        if E is None or mask is None:
            debug["status"] = "E_failed"
            return None, None

        inlier_count = int(mask.sum())
        debug["num_inliers"] = inlier_count

        if inlier_count < self.min_inliers:
            debug["status"] = "too_few_inliers"
            return None, None

        return E, mask

    # Step 5: Recover pose from Essential matrix and build T_prev_to_curr
    # We handle multiple candidates if E has more than 3 rows
    # and we also have a fallback to Fundamental matrix if Essential fails.
    def _recover_T_prev_to_curr(self, E, pts_prev, pts_curr, mask, debug) -> Optional[np.ndarray]:
        def split_E_candidates(E_mat: np.ndarray) -> List[np.ndarray]:
            if E_mat is None:
                return []
            E_mat = np.asarray(E_mat)
            if E_mat.shape == (3, 3):
                return [E_mat]
            if E_mat.ndim == 2 and E_mat.shape[1] == 3 and (E_mat.shape[0] % 3 == 0):
                return [E_mat[i:i+3, :] for i in range(0, E_mat.shape[0], 3)]
            return []

        def try_pose(E_try, mask_try, tag, cand_idx):
            try:
                cheir_inliers, R, t, _ = cv2.recoverPose(
                    E_try, pts_prev, pts_curr, self.k, mask=mask_try
                )
            except cv2.error:
                return None

            if R is None or t is None or (not np.isfinite(R).all()) or (not np.isfinite(t).all()):
                return None

            detR = float(np.linalg.det(R))
            if abs(detR - 1.0) > 0.1:
                return None

            # NOTE: keep your current formula if you want, but this is the correct degrees form:
            angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))

            debug[f"{tag}{cand_idx}_detR"] = detR
            debug[f"{tag}{cand_idx}_rot_angle_deg"] = angle
            debug[f"{tag}{cand_idx}_cheir_inliers"] = int(cheir_inliers)

            if angle > 60.0:
                return None

            T = to_homogeneous_transform(R, t)
            return {
                "T": T,
                "inliers": int(cheir_inliers),
                "detR": detR,
                "angle": angle,
                "t_norm": float(np.linalg.norm(t)),
            }

        # Attempt 1: Essential
        
        best = None
        best_idx = -1
        for i, E_i in enumerate(split_E_candidates(E)):
            res = try_pose(E_i, mask, "E", i)
            if res is None:
                continue
            if (best is None) or (res["inliers"] > best["inliers"]):
                best = res
                best_idx = i

        if best is not None:
            debug["E_selected_idx"] = best_idx
            debug["detR"] = best["detR"]
            debug["rot_angle_deg"] = best["angle"]
            debug["t_norm"] = best["t_norm"]
            debug["status"] = "ok_E_selected"
            return best["T"]

        # Attempt 2: Fundamental fallback
        F, maskF = cv2.findFundamentalMat(
            pts_prev, pts_curr,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=1.0,
            confidence=0.999
        )
        if F is None or maskF is None or F.shape != (3, 3):
            debug["status"] = "pose_rejected_no_F_fallback"
            return None

        E2 = self.k.T @ F @ self.k

        best2 = None
        best2_idx = -1
        for i, E_i in enumerate(split_E_candidates(E2)):
            res = try_pose(E_i, maskF, "F", i)
            if res is None:
                continue
            if (best2 is None) or (res["inliers"] > best2["inliers"]):
                best2 = res
                best2_idx = i

        if best2 is not None:
            debug["F_selected_idx"] = best2_idx
            debug["detR"] = best2["detR"]
            debug["rot_angle_deg"] = best2["angle"]
            debug["t_norm"] = best2["t_norm"]
            debug["status"] = "ok_F_selected"
            return best2["T"]

        debug["status"] = "pose_rejected"
        return None

    # Step 6: Accumulate world pose
    def _accumulate_pose(self, frame: Frame, T_prev_to_curr: np.ndarray) -> None:
        self.world_T_cam = self.world_T_cam @ T_prev_to_curr
        frame.pose = self.world_T_cam.copy()
        self.tum_poses.append((frame.timestamp, frame.pose.copy()))

    # Step 7: Update trajectory
    def _append_traj_point(self) -> None:
        x = float(self.world_T_cam[0, 3])
        y = float(self.world_T_cam[1, 3])
        z = float(self.world_T_cam[2, 3])
        self.traj_points.append((x, y, z))

# visPy Visualization
class VispyTrajectoryViewer3D:
    def __init__(self, title="VO Trajectory (3D)", size=(1100, 800)):
        self.canvas = scene.SceneCanvas(keys="interactive", title=title, size=size, show=True)
        self.view = self.canvas.central_widget.add_view()

        # Turntable camera: allows rotation, zoom, pan with mouse
        self.view.camera = scene.TurntableCamera(fov=60, distance=3.0)

        # Axes at world origin
        self.axes = visuals.XYZAxis(parent=self.view.scene)

        # Trajectory
        self.line = visuals.Line(
            pos=np.zeros((1, 3), dtype=np.float32),
            parent=self.view.scene,
            method="gl"
        )

        # Current camera position marker
        self.marker = visuals.Markers(parent=self.view.scene)
        self.marker.set_data(np.zeros((1, 3), dtype=np.float32), size=10)

        # Camera orientation axes (3 colored lines)
        self.cam_axes = visuals.XYZAxis(parent=self.view.scene)

        # Camera axes breadcrumbs
        self.breadcrumb_stride = 20
        self._breadcrumbs = []

    def _set_cam_axes_transform(self, world_T_cam: np.ndarray):
        """
        Place the camera axes in the world at the current pose.
        VisPy visuals can be transformed using a scene transform.
        """
        tr = MatrixTransform()
        tr.matrix = world_T_cam.astype(np.float32).T  
        self.cam_axes.transform = tr

    def update(self, traj_points_xyz, world_T_cam=None, frame_id=None):
        if traj_points_xyz is None or len(traj_points_xyz) < 1:
            return

        pts = np.asarray(traj_points_xyz, dtype=np.float32)

        self.line.set_data(pos=pts)
        self.marker.set_data(pts[-1:].copy(), size=10)

        if world_T_cam is not None:
            self._set_cam_axes_transform(world_T_cam)

            # Camera axes breadcrumbs
            if frame_id is not None and frame_id % self.breadcrumb_stride == 0:
                bc = visuals.XYZAxis(parent=self.view.scene)
                self._set_any_axes_transform(bc, world_T_cam)
                self._breadcrumbs.append(bc)

        self.canvas.update()

    def _set_any_axes_transform(self, axes_obj, world_T_cam: np.ndarray):
        tr = MatrixTransform()
        tr.matrix = world_T_cam.astype(np.float32).T
        axes_obj.transform = tr

    def process_events(self):
        self.canvas.app.process_events()



# Helpers

# Helper 1: Save TUM trajectory to file
def save_tum_trajectory(path: str, tum_poses: List[Tuple[float, np.ndarray]]) -> None:
    """
    TUM format:
    timestamp tx ty tz qx qy qz qw
    """
    with open(path, "w") as f:
        for ts, T in tum_poses:
            R = T[:3, :3]
            t = T[:3, 3]
            qx, qy, qz, qw = rotation_matrix_to_quat_xyzw(R)
            f.write(f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                    f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
            
# Helper 2: Save TUM trajectory with smoothed translations
def save_tum_trajectory_smoothed(
    path: str,
    tum_poses: List[Tuple[float, np.ndarray]],
    window_length: int = 21,
    polyorder: int = 3
) -> None:
    """
    Save TUM trajectory, but with SMOOTHED translations (tx,ty,tz).
    Rotations are kept as-is.
    """
    if not tum_poses:
        return

    # Extract raw positions
    positions = np.array([T[:3, 3] for _, T in tum_poses], dtype=np.float64)

    # Smooth positions
    positions_s = smooth_trajectory_savgol(
        positions, window_length=window_length, polyorder=polyorder
    )

    # Write with same rotations, smoothed translations
    with open(path, "w") as f:
        for (ts, T), p in zip(tum_poses, positions_s):
            R = T[:3, :3]
            qx, qy, qz, qw = rotation_matrix_to_quat_xyzw(R)
            f.write(f"{ts:.6f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")


# Helper 3: Savitzky-Golay smoothing for 3D trajectories
def smooth_trajectory_savgol(positions, window_length=21, polyorder=3):
    """
    Smooth a 3D trajectory using Savitzky-Golay filter.
    positions: (N,3) array-like of [x,y,z] positions.
    Returns: (N,3) np.ndarray smoothed positions.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must be shape (N,3)")

    N = positions.shape[0]
    if N < 5:
        # Too short to smooth meaningfully
        return positions.copy()

    # Ensure window_length is odd and less than N
    wl = int(window_length)
    if wl % 2 == 0:
        wl += 1
    wl = min(wl, N if N % 2 == 1 else N - 1)
    wl = max(wl, polyorder + 2 + ((polyorder + 2) % 2 == 0))  # ensure odd and > polyorder

    if wl > N:
        return positions.copy()

    try:
        from scipy.signal import savgol_filter
    except ImportError as e:
        raise ImportError("scipy is required for Savitzky-Golay smoothing: pip install scipy") from e

    smoothed = positions.copy()
    for i in range(3):
        smoothed[:, i] = savgol_filter(positions[:, i], window_length=wl, polyorder=polyorder, mode="interp")

    return smoothed

# Helper 4: Load image paths from a folder, sorted
def load_image_paths(folder: str) -> List[str]:
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    paths.sort()
    return paths


def main():
    parser = argparse.ArgumentParser(description="Visual Odometry with ORB features")
    parser.add_argument("--data_dir", type=str, default="Dataset_VO", help="Folder containing the image sequence")

    args = parser.parse_args()

    image_paths = load_image_paths(args.data_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in: {args.data_dir}")

    first = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read: {image_paths[0]}")
    h, w = first.shape[:2]
    K = build_k_from_image_size(w, h)

    vo = VisualOdometry(K)
    
    viewer3d = VispyTrajectoryViewer3D()

    cv2.namedWindow("keypoints", cv2.WINDOW_NORMAL)
    
    try:
        for i, p in enumerate(image_paths):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                print(f"[WARN] failed to read {p}")
                continue

            try:
                timestamp = float(os.path.splitext(os.path.basename(p))[0])
            except ValueError:
                timestamp = float(i)

            frame = Frame(frame_id=i, image_bgr=img, timestamp=timestamp)

            ok, dbg = vo.process(frame)

            if frame.keypoints is not None:
                vis = cv2.drawKeypoints(
                    img, frame.keypoints, None,
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                )
            else:
                vis = img

            detR = dbg.get("detR", 0.0)
            tn   = dbg.get("t_norm", 0.0)
            tz   = dbg.get("t_cam_z", 0.0)

            text = (f"id={i} t={frame.timestamp:.6f} "
                    f"kp={dbg['num_keypoints']} m={dbg['num_matches']} "
                    f"inl={dbg['num_inliers']} detR={detR:.3f} |t|={tn:.2f} tz={tz:+.2f} "
                    f"st={dbg['status']}")

            cv2.putText(vis, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)

            
    
            # Draw trajectory in VisPy (with smoothing once we have enough points)

            if not hasattr(vo, "traj_smooth_cache"):
                vo.traj_smooth_cache = vo.traj_points  # Start as raw

            # Update smooth cache once we have enough point
            if len(vo.traj_points) >= 25:
                pts = np.asarray(vo.traj_points, dtype=np.float64)
                pts_s = smooth_trajectory_savgol(pts, window_length=21, polyorder=3)
                vo.traj_smooth_cache = [tuple(p) for p in pts_s] 

            # always display cached smooth once available
            traj_to_show = vo.traj_smooth_cache if len(vo.traj_points) >= 25 else vo.traj_points

            if frame.pose is not None:
                viewer3d.update(traj_to_show, world_T_cam=frame.pose, frame_id=i)

            viewer3d.process_events()

            cv2.imshow("keypoints", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    finally:
        # TUM Trajectory log (raw, unsmoothed)
        save_tum_trajectory("traj_estimation.txt", vo.tum_poses)

        # Smoothed TUM Trajectory log (only translations smoothed)
        save_tum_trajectory_smoothed("traj_estimation_smooth.txt", vo.tum_poses, window_length=21, polyorder=3)

        print("TUM trajectory saved to traj_estimation.txt and traj_estimation_smooth.txt")

        while True:
            viewer3d.process_events()  # keep VisPy alive
            if cv2.waitKey(30) & 0xFF in (27, ord('q')):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()