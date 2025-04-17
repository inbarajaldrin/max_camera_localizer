# geometric_function.py
import cv2
import scipy.spatial.transform
from scipy.spatial.transform import Rotation as R


def rvec_to_quat(rvec):
    """Convert OpenCV rotation vector to quaternion [x, y, z, w]"""
    rot, _ = cv2.Rodrigues(rvec)
    return R.from_matrix(rot).as_quat()  # returns [x, y, z, w]

def quat_to_rvec(quat):
    """Convert quaternion [x, y, z, w] to OpenCV rotation vector"""
    rot = R.from_quat(quat).as_matrix()
    rvec, _ = cv2.Rodrigues(rot)
    return rvec

def transform_point_cam_to_world(point_cam, cam_pos_world, cam_quat_world):
    r_cam_world = R.from_quat(cam_quat_world)
    return cam_pos_world + r_cam_world.apply(point_cam)

def transform_orientation_cam_to_world(marker_quat_cam, cam_quat_world):
    r_marker_cam = R.from_quat(marker_quat_cam)
    r_cam_world = R.from_quat(cam_quat_world)
    r_marker_world = r_cam_world * r_marker_cam
    return r_marker_world.as_quat()

def slerp_quat(q1, q2, blend=0.5):
    """Spherical linear interpolation between two quaternions"""
    rot1 = R.from_quat(q1)
    rot2 = R.from_quat(q2)
    rots = R.concatenate([rot1, rot2])
    slerp = scipy.spatial.transform.Slerp([0, 1], rots)
    return slerp(blend).as_quat()