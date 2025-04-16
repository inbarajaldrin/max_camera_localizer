# camera_pose_bridge.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy.spatial.transform import Rotation as R
import threading

class CameraPose:
    def __init__(self):
        self.position = np.zeros(3)
        self.quaternion = np.array([0, 0, 0, 1])
        self.lock = threading.Lock()

    def update(self, pos, quat):
        with self.lock:
            self.position = pos
            self.quaternion = quat

    def get(self):
        with self.lock:
            return self.position.copy(), self.quaternion.copy()

# This will be shared externally
camera_pose = CameraPose()

class TCPSubscriber(Node):
    def __init__(self):
        super().__init__('tcp_subscriber')
        
        # Offset of camera from EE (in EE frame)
        self.cam_offset_position = np.array([-0.012, -0.03, -0.01])  # meters
        self.cam_offset_quat = np.array([0.0, 0.0, 0.0, 1.0])  # identity quaternion

        self.subscription = self.create_subscription(
            PoseStamped,
            '/tcp_pose_broadcaster/pose',
            self.listener_callback,
            10)
        self.get_logger().info("TCPSubscriber node started.")

    def listener_callback(self, msg):
        ee_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        ee_quat = np.array([msg.pose.orientation.x,
                            msg.pose.orientation.y,
                            msg.pose.orientation.z,
                            msg.pose.orientation.w])

        r_ee = R.from_quat(ee_quat)
        r_cam_offset = R.from_quat(self.cam_offset_quat)

        cam_pos_world = ee_pos + r_ee.apply(self.cam_offset_position)
        cam_quat_world = (r_ee * r_cam_offset).as_quat()

        camera_pose.update(cam_pos_world, cam_quat_world)


def start_ros_listener():
    import threading
    rclpy.init()
    node = TCPSubscriber()
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
