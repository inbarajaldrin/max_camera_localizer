import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.spatial.transform import Rotation as R
from max_camera_localizer.aruco_pose_bridge import ArucoPoseBridge
from max_camera_localizer.geometric_functions import rvec_to_quat, quat_to_rvec, transform_orientation_cam_to_world, transform_point_cam_to_world, slerp_quat
import threading
import rclpy


c_width = 1280 # pix
c_hfov = 69.4 # deg
fx = c_width / (2 * np.tan(np.deg2rad(c_hfov / 2)))
print(f"Calculated fx as {fx}")

c_height = 720 # pix
c_vfov = 42.5 # deg
fy = c_height / (2 * np.tan(np.deg2rad(c_vfov / 2)))
print(f"Calculated fy as {fy}")

CAMERA_MATRIX = np.array([[fx, 0, c_width / 2],
                          [0, fy, c_height / 2],
                          [0, 0, 1]], dtype=np.float32)
DIST_COEFFS = np.zeros((5, 1), dtype=np.float32) # datasheet says <= 1.5%
MARKER_SIZE = 0.019  # meters
HALF_SIZE = MARKER_SIZE / 2
ARUCO_DICTS = {
    "DICT_4X4_250": aruco.DICT_4X4_250,
    "DICT_5X5_250": aruco.DICT_5X5_250
}
trackers = {}

class QuaternionKalman:
    def __init__(self):
        # 10 states: [x, y, z, qx, qy, qz, qw, vx, vy, vz]
        self.kf = cv2.KalmanFilter(10, 7)

        dt = 1

        # A: Transition matrix (10x10)
        self.kf.transitionMatrix = np.eye(10, dtype=np.float32)
        for i in range(3):  # x += vx*dt
            self.kf.transitionMatrix[i, i+7] = dt

        # H: Measurement matrix (7x10)
        self.kf.measurementMatrix = np.zeros((7, 10), dtype=np.float32)
        self.kf.measurementMatrix[0:7, 0:7] = np.eye(7)

        # Process noise covariance (Q). Lower Values = More Inertia
        self.kf.processNoiseCov = np.eye(10, dtype=np.float32) * 1e-6
        for i in range(3):   # x, y, z
            self.kf.processNoiseCov[i, i] = 1e-4
        for i in range(3, 7):  # quaternion x, y, z, w
            self.kf.processNoiseCov[i, i] = 1e-3
        for i in range(7, 10):  # vx, vy, vz
            self.kf.processNoiseCov[i, i] = 1e-4
        

        # Measurement noise covariance (R). Lower Values = More Trust = You have good cameras
        self.kf.measurementNoiseCov = np.eye(7, dtype=np.float32)
        for i in range(3):   # position
            self.kf.measurementNoiseCov[i, i] = 1e-4
        for i in range(3, 7):  # quaternion
            self.kf.measurementNoiseCov[i, i] = 1e-4
        
        self.kf.errorCovPost = np.eye(10, dtype=np.float32)

        # Initial state
        self.kf.statePost = np.zeros((10, 1), dtype=np.float32)
        self.kf.statePost[3:7] = np.array([[0], [0], [0], [1]], dtype=np.float32)  # Identity quaternion

    def correct(self, tvec, rvec):
        quat = rvec_to_quat(rvec)
        measurement = np.vstack((tvec.reshape(3, 1), np.array(quat).reshape(4, 1))).astype(np.float32)
        self.kf.correct(measurement)

    def predict(self):
        pred = self.kf.predict()
        pred_tvec = pred[0:3].flatten()
        pred_quat = pred[3:7].flatten()
        # Normalize quaternion to prevent drift
        pred_quat /= np.linalg.norm(pred_quat)
        pred_rvec = quat_to_rvec(pred_quat).flatten()
        return pred_tvec, pred_rvec

def detect_available_cameras(max_cams=15):
    """Try to open camera IDs and return a list of working ones."""
    available = []
    for i in range(max_cams):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available

def select_camera(available_ids):
    """Let user preview and select from available cameras."""
    print("Available camera IDs:", available_ids)
    for cam_id in available_ids:
        cap = cv2.VideoCapture(cam_id)
        print(f"Showing preview for camera ID {cam_id} (press any key to continue, or ESC to select this one)...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow(f"Camera ID {cam_id}", frame)
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                return cam_id
            elif key != -1:
                break
        cap.release()
        cv2.destroyAllWindows()
    return available_ids[0] if available_ids else None

def detect_markers(frame, gray, aruco_dicts, parameters):
    all_corners, all_ids = [], []
    for dict_id in aruco_dicts.values():
        aruco_dict = aruco.getPredefinedDictionary(dict_id)
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None:
            all_corners.extend(corners)
            all_ids.extend(ids.flatten())
            aruco.drawDetectedMarkers(frame, corners, ids)
    return all_corners, all_ids

def estimate_pose(frame, corners, ids, camera_matrix, dist_coeffs, marker_size,
                  kalman_filters, marker_stabilities, last_seen_frames, current_frame, cam_pos, cam_quat):
    max_movement = 0.05  # meters
    hold_required = 3    # frames it must persist


    if corners and ids:
        for corner, marker_id in zip(corners, ids):
            marker_id = int(marker_id)

            # Initialize tracking state if this is a new marker
            if marker_id not in kalman_filters:
                kalman_filters[marker_id] = QuaternionKalman()
                marker_stabilities[marker_id] = {
                    "last_tvec": None,
                    "last_frame": -1,
                    "confirmed": False,
                    "hold_counter": 0
                }
                last_seen_frames[marker_id] = 0

            kalman = kalman_filters[marker_id]
            stability = marker_stabilities[marker_id]

            image_points = corner[0].reshape(-1, 2)
            object_points = np.array([
                [-HALF_SIZE,  HALF_SIZE, 0],
                [ HALF_SIZE,  HALF_SIZE, 0],
                [ HALF_SIZE, -HALF_SIZE, 0],
                [-HALF_SIZE, -HALF_SIZE, 0]
            ], dtype=np.float32)

            success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
            if success:
                tvec_flat = tvec.flatten()
                distance = np.linalg.norm(tvec_flat - stability["last_tvec"]) if stability["last_tvec"] is not None else 0
                movement_ok = distance < max_movement

                if movement_ok:
                    stability["hold_counter"] += 1
                else:
                    stability["hold_counter"] = 0

                stability["last_tvec"] = tvec_flat
                stability["last_frame"] = current_frame

                if stability["hold_counter"] >= hold_required:
                    stability["confirmed"] = True

                    measured_quat = rvec_to_quat(rvec)
                    pred_tvec, pred_rvec = kalman.predict()
                    pred_quat = rvec_to_quat(pred_rvec)
                    blend_factor = 0.99
                    blended_quat = slerp_quat(pred_quat, measured_quat, blend=blend_factor)
                    blended_rvec = quat_to_rvec(blended_quat)
                    blended_tvec = blend_factor * tvec_flat + (1 - blend_factor) * pred_tvec
                    kalman.correct(blended_tvec, blended_rvec)
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, blended_rvec, blended_tvec, marker_size * 0.5)
                    print(f"[{marker_id}] Confirmed: t={tvec_flat}, r={rvec.flatten()}")
                    last_seen_frames[marker_id] = current_frame
                    # Convert to world frame
                    marker_pos_world = transform_point_cam_to_world(blended_tvec, cam_pos, cam_quat)
                    marker_quat_world = transform_orientation_cam_to_world(blended_quat, cam_quat)
                    print(f"[{marker_id}] WORLD Pose:\n  Pos: {marker_pos_world}\n  Quat: {marker_quat_world}")
                else:
                    print(f"[{marker_id}] Holding: t={tvec_flat}, hold={stability['hold_counter']}")




    for marker_id, kalman in kalman_filters.items():
        stability = marker_stabilities[marker_id]
        last_seen = last_seen_frames[marker_id]
        if not stability["confirmed"]:
            continue

        if current_frame - last_seen < 15:
            pred_tvec, pred_rvec = kalman.predict()
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                              pred_rvec.reshape(3, 1), pred_tvec.reshape(3, 1), marker_size * 0.5)
            if not current_frame == last_seen:
                print(f"[{marker_id}] Ghost: t={pred_tvec}, r={pred_rvec}")
                # Convert to world frame
                pred_quat = rvec_to_quat(pred_rvec)
                marker_pos_world = transform_point_cam_to_world(pred_tvec, cam_pos, cam_quat)
                marker_quat_world = transform_orientation_cam_to_world(pred_quat, cam_quat)
                print(f"[{marker_id}] GHOST WORLD Pose:\n  Pos: {marker_pos_world}\n  Quat: {marker_quat_world}")
        else:
            stability["confirmed"] = False


def draw_overlay(frame, cam_pos, cam_quat, marker_data, frame_idx):
    # Convert quaternion to Euler for readability
    cam_rot = R.from_quat(cam_quat)
    cam_rot = np.rad2deg(cam_rot.as_rotvec())
    # cam_euler = np.rad2deg(cam_rot.as_euler('xyz'))

    text_lines = [
        f"Frame: {frame_idx}",
        f"Camera Position: x={1000*cam_pos[0]:.3f}mm, y={1000*cam_pos[1]:.3f}mm, z={1000*cam_pos[2]:.3f}mm",
        f"Camera Euler: rx={cam_rot[0]:.1f}deg, ry={cam_rot[1]:.1f}deg, rz={cam_rot[2]:.1f}deg"
    ]

    for i, line in enumerate(text_lines):
        cv2.putText(frame, line, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    for i, (marker_id, (world_pos, world_quat)) in enumerate(marker_data.items()):
        line = f"Marker {marker_id} pos: x={1000*world_pos[0]:.3f}mm, y={1000*world_pos[1]:.3f}mm, z={1000*world_pos[2]:.3f}mm"
        cv2.putText(frame, line, (10, 120 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        world_rot = R.from_quat(world_quat)
        world_rot = np.rad2deg(world_rot.as_rotvec())
        # world_euler = np.rad2deg(world_rot.as_euler('xyz'))
        line = f"Marker {marker_id} rot: rx={world_rot[0]:.1f}deg, ry={world_rot[1]:.1f}deg, rz={world_rot[2]:.1f}deg"
        cv2.putText(frame, line, (10, 140 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)



def start_ros_node():
    rclpy.init()
    node = ArucoPoseBridge()
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    return node


def main():
    bridge_node = start_ros_node()

    kalman_filters = {}
    marker_stabilities = {}
    last_seen_frames = {}
    frame_idx = 0
    available = detect_available_cameras()
    if not available:
        return

    # Defaults to selector, but hardcoding camera id is faster.
    # cam_id = 8
    cam_id = select_camera(available)
    if cam_id is None:
        return

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        return

    parameters = aruco.DetectorParameters()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, c_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c_height)
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = detect_markers(frame, gray, ARUCO_DICTS, parameters)

        marker_data = {}
        cam_pos, cam_quat = bridge_node.get_camera_pose()
        # print("Camera Pose:", cam_pos)
        # print("Camera Quat:", cam_quat)

        corners, ids = detect_markers(frame, gray, ARUCO_DICTS, parameters)
        estimate_pose(frame, corners, ids, CAMERA_MATRIX, DIST_COEFFS, MARKER_SIZE,
                    kalman_filters, marker_stabilities, last_seen_frames, frame_idx, cam_pos, cam_quat)


        # After estimating pose, collect marker world positions
        for marker_id in kalman_filters:
            if marker_stabilities[marker_id]["confirmed"]:
                tvec, rvec = kalman_filters[marker_id].predict()
                rquat = rvec_to_quat(rvec)
                world_pos = transform_point_cam_to_world(tvec, cam_pos, cam_quat)
                world_rot = transform_orientation_cam_to_world(rquat, cam_quat)
                marker_data[marker_id] = (world_pos, world_rot)

        bridge_node.publish_camera_pose(cam_pos, cam_quat)
        bridge_node.publish_marker_poses(marker_data)
        draw_overlay(frame, cam_pos, cam_quat, marker_data, frame_idx)

        cv2.imshow("ArUco Detection", frame)
        print(frame_idx)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
