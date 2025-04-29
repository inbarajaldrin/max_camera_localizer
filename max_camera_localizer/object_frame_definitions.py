from scipy.spatial.transform import Rotation as R
import numpy as np

def define_body_frame_allen_key(p1, p2, p3, width=0.005):
    "Returns Origin, Quat, {Contact Points}"
    "Contact points as (idx, pos, normvec)"
    # Identify the side lengths
    dists = [
        (np.linalg.norm(p1 - p2), p1, p2),
        (np.linalg.norm(p2 - p3), p2, p3),
        (np.linalg.norm(p3 - p1), p3, p1)
    ]
    dists.sort(key=lambda x: x[0])  # Sort by length

    short, mid, long = dists  # short: ~4cm (A and C), mid: ~14.5cm (A and B), long: ~17cm (B and C)

    A = [pt for pt in [p1, p2, p3] if (pt is short[1] or pt is short[2]) and (pt is mid[1] or pt is mid[2])][0]
    B = [pt for pt in [p1, p2, p3] if (pt is mid[1] or pt is mid[2]) and (pt is long[1] or pt is long[2])][0]
    C = [pt for pt in [p1, p2, p3] if pt is not A and pt is not B][0]  # Third point

    origin = A
    x_axis = B - A
    x_axis /= np.linalg.norm(x_axis)

    to_c = C - origin
    y_axis = to_c - np.dot(to_c, x_axis) * x_axis
    y_axis /= np.linalg.norm(y_axis)

    z_axis = np.cross(x_axis, y_axis)

    rot_matrix = np.column_stack((x_axis, y_axis, z_axis))
    quat = R.from_matrix(rot_matrix).as_quat()

    # Contact #s 1 and 2 are +-Y side of point A
    # Contact #s 3 and 4 are +-Y side of point B
    # Contact #s 5 and 6 are +-X side of point C
    normvecs = np.array([[0, -1, 0], [0,  1, 0],
                         [0, -1, 0], [0,  1, 0],
                         [-1, 0, 0], [ 1, 0, 0]])
    normvecs = [rot_matrix @ normvec for normvec in normvecs]
    positions = [A, A, B, B, C, C]
    positions = [position - width * normvec for position, normvec in zip(positions, normvecs)]
    contact_points = [(i, position, normvec) for i, (position, normvec) in enumerate(zip(positions, normvecs))]
    return origin, quat, contact_points


def define_body_frame_pliers(p1, p2, p3, widths=[0.005, 0.005, 0.012, 0.012]):
    # Identify the 3.7cm side
    dists = [
        (np.linalg.norm(p1 - p2), p1, p2),
        (np.linalg.norm(p2 - p3), p2, p3),
        (np.linalg.norm(p3 - p1), p3, p1)
    ]
    dists.sort(key=lambda x: x[0])

    short, long1, long2 = dists
    A, B = short[1], short[2]  # Shortest side
    C = [pt for pt in [p1, p2, p3] if pt is not A and pt is not B][0]

    origin = C
    mid_ab = (A + B) / 2
    y_axis = mid_ab - origin
    y_axis /= np.linalg.norm(y_axis)

    # Z Always up
    z_axis = np.array([0, 0, 1])

    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    rot_matrix = np.column_stack((x_axis, y_axis, z_axis))
    quat = R.from_matrix(rot_matrix).as_quat()

    # Reshuffle A and B so that A has the lower x value
    xfactor_a = np.dot(A, x_axis)
    xfactor_b = np.dot(B, x_axis)
    if xfactor_b < xfactor_a:
        A, B = B, A  # Swap so that A has the lower x value

    # Contact #s 1 and 2 are -+X side of point A, B
    # Contact #s 3 and 4 are +-X side of point C
    normvecs = np.array([[ 1, 0, 0], [-1, 0, 0],
                         [-1, 0, 0], [ 1, 0, 0]])
    normvecs = [rot_matrix @ normvec for normvec in normvecs]
    positions = [A, B, C, C]
    positions = [position - width * normvec for position, normvec, width in zip(positions, normvecs, widths)]
    contact_points = [(i, position, normvec) for i, (position, normvec) in enumerate(zip(positions, normvecs))]
    return origin, quat, contact_points