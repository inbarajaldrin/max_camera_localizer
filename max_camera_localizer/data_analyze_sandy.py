import sqlite3
import os
import csv
import yaml
import rclpy
from bisect import bisect_left
import copy
import rclpy.serialization
from rosidl_runtime_py.utilities import get_message

from geometry_msgs.msg import PoseStamped
from max_camera_msgs.msg import PusherInfo
from scipy.spatial.transform import Rotation as R

# Config
bag_dir = "/media/max/OS/Users/maxlg/Desktop/==RISELABS/Pusher Data Batch 2/jenga"
pose_topic = "/object_poses/jenga_6"
db_files = [f for f in os.listdir(bag_dir) if f.endswith(".db3")]
assert len(db_files) == 1, f"Expected 1 .db3 file in {bag_dir}, found {len(db_files)}"
bag_name = db_files[0]
green_topic = "/pusher_data_green"
yellow_topic = "/pusher_data_yellow"
topics_to_export = [pose_topic, green_topic, yellow_topic]
output_dir = bag_dir + "/csv_output"
TOLERANCE = 0.002  # seconds
JUMP_THRESHOLD = 5
WINDOW_SIZE = 10
DELETED_PRECENTAGE = 0.95

# Paths
pose_file = os.path.join(output_dir, pose_topic.replace("/", "_") + ".csv")
green_file = os.path.join(output_dir, green_topic.replace("/", "_") + ".csv")
yellow_file = os.path.join(output_dir, yellow_topic.replace("/", "_") + ".csv")
output_file = os.path.join(output_dir, "merged_pushers_with_pose.csv")

os.makedirs(output_dir, exist_ok=True)

# Load topic metadata
topic_map = {
    entry['topic_metadata']['name']: entry['topic_metadata']['type']
    for entry in yaml.safe_load(open(os.path.join(bag_dir, "metadata.yaml")))
    ['rosbag2_bagfile_information']['topics_with_message_count']
}

# Connect and query SQLite
conn = sqlite3.connect(os.path.join(bag_dir, bag_name))
cursor = conn.cursor()
cursor.execute("SELECT id, name, type FROM topics")
topic_id_map = {name: (id, type_) for id, name, type_ in cursor.fetchall()}

# Export each topic
for topic in topics_to_export:
    if topic not in topic_id_map:
        print(f"Topic {topic} not found.")
        continue

    topic_id, type_str = topic_id_map[topic]
    try:
        msg_type = get_message(type_str)
    except Exception as e:
        print(f"Could not load message type for {type_str}: {e}")
        continue

    cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = ?", (topic_id,))
    messages = cursor.fetchall()

    out_path = os.path.join(output_dir, topic.replace("/", "_") + ".csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)

        if type_str == "geometry_msgs/msg/PoseStamped":
            writer.writerow(["sec", "nanosec", "position_x", "position_y", "position_z", "orientation_r", "orientation_p", "orientation_y"])
            for ts, raw in messages:
                msg = rclpy.serialization.deserialize_message(raw, PoseStamped)
                p, o = msg.pose.position, msg.pose.orientation
                euler = R.from_quat([o.x, o.y, o.z, o.w]).as_euler('xyz')
                writer.writerow([msg.header.stamp.sec, msg.header.stamp.nanosec, p.x, p.y, p.z, euler[0], euler[1], euler[2]])

        elif type_str.endswith("PusherInfo"):
            writer.writerow(["sec", "nanosec", "frame_num", "pusher_name", "color_r", "color_g", "color_b", "color_a", "loc_x", "loc_y", "loc_z", "nearest_x", "nearest_y", "nearest_z", "kappa", "object_index", "local_contour_index"])
            for ts, raw in messages:
                msg = rclpy.serialization.deserialize_message(raw, PusherInfo)
                writer.writerow([
                    msg.header.stamp.sec, msg.header.stamp.nanosec, msg.frame_num, msg.pusher_name,
                    msg.color.r, msg.color.g, msg.color.b, msg.color.a,
                    msg.pusher_location.x, msg.pusher_location.y, msg.pusher_location.z,
                    msg.nearest_point.x, msg.nearest_point.y, msg.nearest_point.z,
                    msg.kappa, msg.object_index, msg.local_contour_index
                ])
        else:
            print(f"Unsupported message type: {type_str}")

    print(f"Exported decoded messages from {topic} to {out_path}")

conn.close()

# Helpers
def load_csv_with_timestamps(path):
    data = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ts = float(row["sec"]) + float(row["nanosec"]) * 1e-9
            data[ts] = row
    return data

def load_pose_data(path):
    poses = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            t = float(row["sec"]) + float(row["nanosec"]) * 1e-9
            poses.append((t, row))
    poses.sort(key=lambda x: x[0])
    return poses

def find_closest_pose(ts, poses, pose_times):
    idx = bisect_left(pose_times, ts)
    best, best_dt = None, float("inf")
    for offset in [-1, 0, 1]:
        i = idx + offset
        if 0 <= i < len(poses):
            dt = abs(poses[i][0] - ts)
            if dt < best_dt:
                best_dt = dt
                best = poses[i][1]
    return best if best_dt <= TOLERANCE else None

def clean_contour_data(rows, jump_threshold=JUMP_THRESHOLD, window_size=WINDOW_SIZE):
    i = 0
    while i < len(rows) - (window_size + 1):
        try:
            cur = int(rows[i]["green_index"] or rows[i]["yellow_index"])
            nxt = int(rows[i + 1]["green_index"] or rows[i + 1]["yellow_index"])
        except:
            i += 1
            continue
        if abs(nxt - cur) > jump_threshold:
            all_jump = True
            for j in range(2, window_size + 2):
                try:
                    val = int(rows[i + j]["green_index"] or rows[i + j]["yellow_index"])
                    if abs(val - cur) <= jump_threshold:
                        all_jump = False
                        break
                except:
                    all_jump = False
                    break
            if not all_jump:
                del rows[i + 1]
            else:
                i += 1
        else:
            i += 1
    return rows

def group_and_trim(rows, threshold=JUMP_THRESHOLD):
    grouped = []
    group = [rows[0]]
    group_id = 0
    rows[0]["group"] = group_id
    for i in range(1, len(rows)):
        try:
            prev = int(rows[i - 1]["green_index"] or rows[i - 1]["yellow_index"])
            curr = int(rows[i]["green_index"] or rows[i]["yellow_index"])
        except:
            group.append(rows[i])
            rows[i]["group"] = group_id
            continue
        if abs(curr - prev) <= threshold:
            group.append(rows[i])
            rows[i]["group"] = group_id
        else:
            grouped.append(group)
            group_id += 1
            group = [rows[i]]
            rows[i]["group"] = group_id
    if group:
        grouped.append(group)

    trimmed = []
    for g in grouped:
        cutoff = max(1, int(len(g) * DELETED_PRECENTAGE))
        trimmed.extend(g[:cutoff])
    return trimmed

# Load data
green_data = load_csv_with_timestamps(green_file)
yellow_data = load_csv_with_timestamps(yellow_file)
poses = load_pose_data(pose_file)
pose_times = [p[0] for p in poses]

data_times = sorted(set(green_data.keys()) | set(yellow_data.keys()))
rows = []
for ts in data_times:
    if not (green_data.get(ts) or yellow_data.get(ts)):
        continue
    row = {"timestamp": f"{ts:.9f}"}
    if (g := green_data.get(ts)):
        row.update({"green_frame": g["frame_num"], "green_index": g["local_contour_index"]})
    if (y := yellow_data.get(ts)):
        row.update({"yellow_frame": y["frame_num"], "yellow_index": y["local_contour_index"]})
    if (p := find_closest_pose(ts, poses, pose_times)):
        row.update({"pose_x": p["position_x"], "pose_y": p["position_y"], "ori_y": p["orientation_y"]})
    rows.append(row)

rows = clean_contour_data(rows)
rows = group_and_trim(rows)

# Additional Jenga-specific duplication
if "jenga" in pose_topic:
    duplicated = []
    for row in rows:
        if "ori_y" not in row:
            continue
        angle = float(row["ori_y"])
        if -3.14159 <= angle < 0:
            new_row = copy.deepcopy(row)
            new_row["ori_y"] = str(angle + 3.14159)
            duplicated.append(new_row)
        elif 0 <= angle < 3.14159:
            new_row = copy.deepcopy(row)
            new_row["ori_y"] = str(angle - 3.14159)
            duplicated.append(new_row)
    rows.extend(duplicated)

with open(output_file, "w", newline="") as f_out:
    fieldnames = ["timestamp", "green_frame", "green_index", "yellow_frame", "yellow_index", "pose_x", "pose_y", "ori_y", "group"]
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"\u2705 Merged alignment saved to: {output_file}")
