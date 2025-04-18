import cv2
# import cv2.aruco as aruco
import numpy as np
from itertools import combinations
from scipy.spatial.transform import Rotation as R
from max_camera_localizer.camera_selection import detect_available_cameras, select_camera
from max_camera_localizer.object_frame_definitions import define_body_frame_allen_key, define_body_frame_pliers
from max_camera_localizer.aruco_pose_bridge import ArucoPoseBridge
from max_camera_localizer.geometric_functions import rvec_to_quat, quat_to_rvec, transform_orientation_cam_to_world, transform_point_cam_to_world, slerp_quat, transform_point_world_to_cam
import threading
import rclpy
import argparse

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
# MARKER_SIZE = 0.019  # meters
# HALF_SIZE = MARKER_SIZE / 2
# ARUCO_DICTS = {
#     "DICT_4X4_250": aruco.DICT_4X4_250,
#     "DICT_5X5_250": aruco.DICT_5X5_250
# }
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

def detect_blue_object_positions(frame, camera_matrix, cam_pos, cam_quat, table_height=0.01, min_area=120):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define blue range in HSV
    lower_blue = np.array([100, 75, 50])
    upper_blue = np.array([140, 255, 255])
    
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    world_points = []
    image_points = []

    if contours:
        for cnt in contours:
            M = cv2.moments(cnt)
            area = cv2.contourArea(cnt)            
            if area < min_area:
                continue  # skip tiny blobs
            if M["m00"] > 0:
                cv2.drawContours(frame, [cnt], 0, (255, 255, 255), 1)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                image_points.append((cx, cy))
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

                # Step 1: Ray in camera frame
                pixel = np.array([cx, cy, 1.0])
                ray_cam = np.linalg.inv(camera_matrix) @ pixel

                # Step 2: Transform ray to world frame
                R_wc = R.from_quat(cam_quat).as_matrix()
                ray_world = R_wc @ ray_cam
                cam_origin_world = np.array(cam_pos)

                # Step 3: Ray-plane intersection with z = table_height
                t = (table_height - cam_origin_world[2]) / ray_world[2]
                point_world = cam_origin_world + t * ray_world
                world_points.append(point_world)

    return world_points, image_points

def identify_objects_from_blobs(world_points, tolerance=5.0):
    known_triangles = {
        "allen_key": [40, 145, 170],
        "pliers": [37, 70, 70]
    }

    identified_objects = []

    for tri_pts in combinations(world_points, 3):
        p1, p2, p3 = np.array(tri_pts[0]), np.array(tri_pts[1]), np.array(tri_pts[2])
        sides = sorted([
            1000 * np.linalg.norm(p1 - p2),
            1000 * np.linalg.norm(p2 - p3),
            1000 * np.linalg.norm(p3 - p1)
        ])

        for name, template in known_triangles.items():
            expected = sorted(template)
            diffs = [abs(a - b) for a, b in zip(sides, expected)]
            if all(d < tolerance for d in diffs):
                if name == "allen_key":
                    pos, quat = define_body_frame_allen_key(p1, p2, p3)
                elif name == "pliers":
                    pos, quat = define_body_frame_pliers(p1, p2, p3)

                identified_objects.append({
                    "name": name,
                    "points": (p1, p2, p3),
                    "sides_mm": sides,
                    "position": pos,
                    "quaternion": quat
                })
                break  # One match per triangle

    return identified_objects

# def estimate_pose(frame, corners, ids, camera_matrix, dist_coeffs, marker_size,
#                   kalman_filters, marker_stabilities, last_seen_frames, current_frame, cam_pos, cam_quat, talk=True):
#     max_movement = 0.05  # meters
#     hold_required = 3    # frames it must persist


#     if corners and ids:
#         for corner, marker_id in zip(corners, ids):
#             marker_id = int(marker_id)

#             # Initialize tracking state if this is a new marker
#             if marker_id not in kalman_filters:
#                 kalman_filters[marker_id] = QuaternionKalman()
#                 marker_stabilities[marker_id] = {
#                     "last_tvec": None,
#                     "last_frame": -1,
#                     "confirmed": False,
#                     "hold_counter": 0
#                 }
#                 last_seen_frames[marker_id] = 0

#             kalman = kalman_filters[marker_id]
#             stability = marker_stabilities[marker_id]

#             image_points = corner[0].reshape(-1, 2)
#             object_points = np.array([
#                 [-HALF_SIZE,  HALF_SIZE, 0],
#                 [ HALF_SIZE,  HALF_SIZE, 0],
#                 [ HALF_SIZE, -HALF_SIZE, 0],
#                 [-HALF_SIZE, -HALF_SIZE, 0]
#             ], dtype=np.float32)

#             success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
#             if success:
#                 tvec_flat = tvec.flatten()
#                 distance = np.linalg.norm(tvec_flat - stability["last_tvec"]) if stability["last_tvec"] is not None else 0
#                 movement_ok = distance < max_movement

#                 if movement_ok:
#                     stability["hold_counter"] += 1
#                 else:
#                     stability["hold_counter"] = 0

#                 stability["last_tvec"] = tvec_flat
#                 stability["last_frame"] = current_frame

#                 if stability["hold_counter"] >= hold_required:
#                     stability["confirmed"] = True

#                     measured_quat = rvec_to_quat(rvec)
#                     pred_tvec, pred_rvec = kalman.predict()
#                     pred_quat = rvec_to_quat(pred_rvec)
#                     blend_factor = 0.99
#                     blended_quat = slerp_quat(pred_quat, measured_quat, blend=blend_factor)
#                     blended_rvec = quat_to_rvec(blended_quat)
#                     blended_tvec = blend_factor * tvec_flat + (1 - blend_factor) * pred_tvec
#                     kalman.correct(blended_tvec, blended_rvec)
#                     cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, blended_rvec, blended_tvec, marker_size * 0.5)
#                     last_seen_frames[marker_id] = current_frame
#                     # Convert to world frame
#                     marker_pos_world = transform_point_cam_to_world(blended_tvec, cam_pos, cam_quat)
#                     marker_quat_world = transform_orientation_cam_to_world(blended_quat, cam_quat)
#                     if talk:
#                         print(f"[{marker_id}] Confirmed: t={tvec_flat}, r={rvec.flatten()}")
#                         print(f"[{marker_id}] WORLD Pose:\n  Pos: {marker_pos_world}\n  Quat: {marker_quat_world}")
#                 elif talk:
#                     print(f"[{marker_id}] Holding: t={tvec_flat}, hold={stability['hold_counter']}")




#     for marker_id, kalman in kalman_filters.items():
#         stability = marker_stabilities[marker_id]
#         last_seen = last_seen_frames[marker_id]
#         if not stability["confirmed"]:
#             continue

#         if current_frame - last_seen < 15:
#             pred_tvec, pred_rvec = kalman.predict()
#             cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
#                               pred_rvec.reshape(3, 1), pred_tvec.reshape(3, 1), marker_size * 0.5)
#             if not current_frame == last_seen:
#                 # Convert to world frame
#                 pred_quat = rvec_to_quat(pred_rvec)
#                 marker_pos_world = transform_point_cam_to_world(pred_tvec, cam_pos, cam_quat)
#                 marker_quat_world = transform_orientation_cam_to_world(pred_quat, cam_quat)
#                 if talk:
#                     print(f"[{marker_id}] Ghost: t={pred_tvec}, r={pred_rvec}")
#                     print(f"[{marker_id}] GHOST WORLD Pose:\n  Pos: {marker_pos_world}\n  Quat: {marker_quat_world}")
#         else:
#             stability["confirmed"] = False


def draw_overlay(frame, cam_pos, cam_quat, object_data, frame_idx, ee_pos, ee_quat):
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = 20
    x0 = 10
    y = 30

    def put_line(text, color=(255, 255, 255)):
        nonlocal y
        cv2.putText(frame, text, (x0, y), font, 0.6, color, 1)
        y += line_height

    # Frame Index
    put_line(f"Frame: {frame_idx}", (200, 200, 200))

    # End Effector
    ee_rotvec = R.from_quat(ee_quat).as_rotvec(degrees=True)
    ee_euler = R.from_quat(ee_quat).as_euler('xyz', degrees=True)
    put_line(f"EE Pos: x={1000*ee_pos[0]:.1f} mm, y={1000*ee_pos[1]:.1f} mm, z={1000*ee_pos[2]:.1f} mm")
    put_line(f"EE Rot: rx={ee_rotvec[0]: 5.1f} deg, ry={ee_rotvec[1]: 5.1f} deg, rz={ee_rotvec[2]: 5.1f} deg")
    put_line(f"EE Euler: r={ee_euler[0]: 5.1f} deg, p={ee_euler[1]: 5.1f} deg, y={ee_euler[2]: 5.1f} deg")

    y += 10  # small gap

    # Camera Info
    cam_euler = R.from_quat(cam_quat).as_euler('xyz', degrees=True)
    put_line(f"Camera Pos: x={1000*cam_pos[0]:.1f} mm, y={1000*cam_pos[1]:.1f} mm, z={1000*cam_pos[2]:.1f} mm", (255, 255, 0))
    put_line(f"Camera Euler: r={cam_euler[0]: 5.1f} deg, p={cam_euler[1]: 5.1f} deg, y={cam_euler[2]: 5.1f} deg", (255, 255, 0))

    y += 10  # gap before object info

    # Generic Object Info
    for obj in object_data:
        name = obj["name"]
        pos = obj["position"]
        quat = obj["quaternion"]

        rotvec = R.from_quat(quat).as_rotvec(degrees=True)
        euler = R.from_quat(quat).as_euler('xyz', degrees=True)
        put_line(f"{name} Pos: x={1000*pos[0]:.1f} mm, y={1000*pos[1]:.1f} mm, z={1000*pos[2]:.1f} mm", (0, 255, 0))
        put_line(f"{name} Euler: r={euler[0]: 5.1f} deg, p={euler[1]: 5.1f} deg, y={euler[2]: 5.1f} deg", (0, 255, 0))
        y += 5


def draw_identified_triangles(frame, camera_matrix, cam_pos, cam_quat, identified_objects):
    color_map = {
        "allen_key": (0, 255, 0),   # Green
        "pliers": (0, 0, 255),      # Red
    }

    for obj in identified_objects:
        name = obj["name"]
        world_pts = obj["points"]
        color = color_map.get(name, (255, 255, 255))  # default to white

        image_pts = []
        for pt in world_pts:
            cam_pt = transform_point_world_to_cam(pt, cam_pos, cam_quat)
            if cam_pt[2] <= 0.01:
                continue  # skip points behind the camera or too close
            u = int(camera_matrix[0, 0] * cam_pt[0] / cam_pt[2] + camera_matrix[0, 2])
            v = int(camera_matrix[1, 1] * cam_pt[1] / cam_pt[2] + camera_matrix[1, 2])
            image_pts.append((u, v))

        if len(image_pts) == 3:
            # Draw triangle edges
            for i in range(3):
                pt1 = image_pts[i]
                pt2 = image_pts[(i + 1) % 3]
                cv2.line(frame, pt1, pt2, color, 2)

            # Draw label with background
            centroid = tuple(np.mean(image_pts, axis=0).astype(int))
            label = f"{name}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (centroid[0] - 2, centroid[1] - h - 4), (centroid[0] + w + 2, centroid[1] + 2), (0, 0, 0), -1)
            cv2.putText(frame, label, (centroid[0], centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw axes using the object pose
        origin = obj["position"]
        rot = R.from_quat(obj["quaternion"])
        axes_world = [
            origin,                          # origin
            origin + rot.apply([0.02, 0, 0]),  # X axis
            origin + rot.apply([0, 0.02, 0]),  # Y axis
            origin + rot.apply([0, 0, 0.02])   # Z axis
        ]

        axes_image = []
        for pt in axes_world:
            cam_pt = transform_point_world_to_cam(pt, cam_pos, cam_quat)
            if cam_pt[2] <= 0.01:
                axes_image.append(None)
                continue
            u = int(camera_matrix[0, 0] * cam_pt[0] / cam_pt[2] + camera_matrix[0, 2])
            v = int(camera_matrix[1, 1] * cam_pt[1] / cam_pt[2] + camera_matrix[1, 2])
            axes_image.append((u, v))

        o, x, y, z = axes_image
        if o and x:
            cv2.arrowedLine(frame, o, x, (0, 0, 255), 2, tipLength=0.3)  # X: Red
        if o and y:
            cv2.arrowedLine(frame, o, y, (0, 255, 0), 2, tipLength=0.3)  # Y: Green
        if o and z:
            cv2.arrowedLine(frame, o, z, (255, 0, 0), 2, tipLength=0.3)  # Z: Blue

    return frame

def start_ros_node():
    rclpy.init()
    node = ArucoPoseBridge()
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    return node

def parse_args():
    parser = argparse.ArgumentParser(description="Run ArUco pose tracker with optional camera ID.")
    parser.add_argument("--camera-id", type=int, default=None,
                        help="Camera device ID to use (e.g., 8). If not set, will scan and prompt.")
    parser.add_argument("--suppress-prints", action='store_true',
                        help="Prevents console prints. Otherwise, prints object positions in both camera frame and base frame.")
    return parser.parse_args()


def main():
    args = parse_args()
    bridge_node = start_ros_node()

    kalman_filters = {}
    marker_stabilities = {}
    last_seen_frames = {}
    frame_idx = 0

    if args.camera_id is not None:
        cam_id = args.camera_id
    else:        
        available = detect_available_cameras()
        if not available:
            return
        cam_id = select_camera(available)
        if cam_id is None:
            return

    talk = not args.suppress_prints

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        return

    # parameters = aruco.DetectorParameters()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, c_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c_height)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        marker_data = {}
        ee_pos, ee_quat = bridge_node.get_ee_pose()
        cam_pos, cam_quat = bridge_node.get_camera_pose()
        # print("Camera Pose:", cam_pos)
        # print("Camera Quat:", cam_quat)

        # estimate_pose(frame, corners, ids, CAMERA_MATRIX, DIST_COEFFS, MARKER_SIZE,
        #             kalman_filters, marker_stabilities, last_seen_frames, frame_idx, cam_pos, cam_quat, talk)

        world_points, image_points = detect_blue_object_positions(frame, CAMERA_MATRIX, cam_pos, cam_quat)
        identified_objects = identify_objects_from_blobs(world_points)
        # After estimating pose, collect marker world positions
        # for marker_id in kalman_filters:
        #     if marker_stabilities[marker_id]["confirmed"]:
        #         tvec, rvec = kalman_filters[marker_id].predict()
        #         rquat = rvec_to_quat(rvec)
        #         world_pos = transform_point_cam_to_world(tvec, cam_pos, cam_quat)
        #         world_rot = transform_orientation_cam_to_world(rquat, cam_quat)
        #         marker_data[marker_id] = (world_pos, world_rot)

        # bridge_node.publish_camera_pose(cam_pos, cam_quat)
        # bridge_node.publish_marker_poses(marker_data)
        draw_overlay(frame, cam_pos, cam_quat, identified_objects, frame_idx, ee_pos, ee_quat)
        
        draw_identified_triangles(frame, CAMERA_MATRIX, cam_pos, cam_quat, identified_objects)

        cv2.imshow("Blue Object Detection", frame)
        if talk:
            print(frame_idx)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
