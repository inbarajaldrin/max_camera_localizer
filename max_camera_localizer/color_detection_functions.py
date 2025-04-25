import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from max_camera_localizer.geometric_functions import transform_points_world_to_img

def detect_blue_object_positions(frame, camera_matrix, cam_pos, cam_quat, table_height=0.01, min_area=120, merge_threshold=0.02):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define blue range in HSV
    lower_blue = np.array([100, 80, 50])
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

    # Step 4: Merge nearby points in world frame
    merged_world_points = []
    used = set()
    for i, pt in enumerate(world_points):
        if i in used:
            continue
        cluster = [pt]
        used.add(i)
        for j in range(i + 1, len(world_points)):
            if j in used:
                continue
            if np.linalg.norm(world_points[j] - pt) < merge_threshold:
                cluster.append(world_points[j])
                used.add(j)
        cluster_avg = np.mean(cluster, axis=0)
        merged_world_points.append(cluster_avg)
    image_points = transform_points_world_to_img(merged_world_points, cam_pos, cam_quat, camera_matrix)
    for (u,v) in image_points:
        cv2.circle(frame, (u, v), 5, (255, 0, 0), -1)
    return merged_world_points, image_points