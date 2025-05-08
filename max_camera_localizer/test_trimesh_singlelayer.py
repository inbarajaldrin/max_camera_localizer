import trimesh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import os
from scipy.interpolate import splprep, splev, interp1d

mesh_key = trimesh.load_mesh(os.path.dirname(os.path.realpath(__file__))+"/STL/Allen Key.STL")
origin_key = [28.33, 0, 37] # STL file doesn't preserve origin... probably should refactor other origin-setting.
mesh_wrench = trimesh.load_mesh(os.path.dirname(os.path.realpath(__file__))+"/STL/Wrench.STL")
origin_wrench = [49, 0, 26]

def get_path(mesh, origin):
    direction = np.array([0, 1, 0])  # y is up/down for meshes
    direction = direction / np.linalg.norm(direction)

    # Get the centroid of the mesh
    plane_y = mesh.center_mass[1]
    plane_origin = [origin[0], plane_y, origin[2]]

    # Intersect mesh with all planes
    all_paths, to3D, _ = trimesh.intersections.mesh_multiplane(mesh, 
                                                    plane_origin=plane_origin,
                                                    plane_normal=direction,
                                                    heights=[0])
    return all_paths[0]

def group_segments_into_loops(segments, tol=1e-5):
    used = np.zeros(len(segments), dtype=bool)
    loops = []

    def next_loop():
        for i, seg in enumerate(segments):
            if not used[i]:
                return i
        return None

    while (start := next_loop()) is not None:
        loop_segments = []
        queue = [start]
        used[start] = True
        loop_segments.append(segments[start])

        # build connected loop
        end_pt = segments[start][1]
        while True:
            found = False
            for i, seg in enumerate(segments):
                if used[i]:
                    continue
                if np.allclose(seg[0], end_pt, atol=tol):
                    loop_segments.append(seg)
                    end_pt = seg[1]
                    used[i] = True
                    found = True
                    break
                elif np.allclose(seg[1], end_pt, atol=tol):
                    loop_segments.append(seg[::-1])  # reverse
                    end_pt = seg[0]
                    used[i] = True
                    found = True
                    break
            if not found:
                break
            if np.allclose(loop_segments[0][0], end_pt, atol=tol):
                break  # closed
        loops.append(np.array(loop_segments))
    return loops

def segments_to_polyline(loop_segments):
    polyline = [loop_segments[0][0], loop_segments[0][1]]
    for seg in loop_segments[1:]:
        if np.allclose(seg[0], polyline[-1]):
            polyline.append(seg[1])
        elif np.allclose(seg[1], polyline[-1]):
            polyline.append(seg[0])
        else:
            raise ValueError("Segment discontinuity")
    return np.array(polyline)

def resample_polyline_equally(polyline, n_points=200):
    # Step 1: Compute arc lengths
    diffs = np.diff(polyline, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.insert(np.cumsum(seg_lengths), 0, 0)

    # Step 2: Interpolation functions
    fx = interp1d(cumulative, polyline[:, 0])
    fy = interp1d(cumulative, polyline[:, 1])

    # Step 3: Evenly spaced distances
    total_length = cumulative[-1]
    new_lengths = np.linspace(0, total_length, n_points)

    # Step 4: Sample new points
    x_new = fx(new_lengths)
    y_new = fy(new_lengths)
    return np.stack((x_new, y_new), axis=1)

def polygon_area(polygon):
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def is_contained(inner, outer):
    # Use center point of inner
    centroid = np.mean(inner, axis=0)
    return Path(outer).contains_point(centroid)

def fit_spline_arclength_weighted(polyline, s=1.0, per=1, pow=1):
    # Compute arc-length-based parameterization
    dists = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    u = np.zeros(len(polyline))
    u[1:] = np.cumsum(dists**pow)
    # u[1:] = dists
    u /= u[-1]  # Normalize to [0, 1]

    # Fit spline using manual parameterization
    x, y = polyline[:, 0], polyline[:, 1]
    tck, _ = splprep([x, y], u=u, s=s, per=per)
    return tck

def compute_curvature(tck, u_eval):
    """
    Compute curvature κ over the spline defined by tck and u_eval.
    Units: curvature is in 1/mm (inverse millimeters), assuming (x, y) in mm.
    """
    dx, dy = splev(u_eval, tck, der=1)
    ddx, ddy = splev(u_eval, tck, der=2)

    # Curvature κ = (x'·y'' - y'·x'') / (x'^2 + y'^2)^(3/2)
    num = dx * ddy - dy * ddx
    denom = (dx**2 + dy**2)**1.5
    kappa = np.divide(num, denom, out=np.zeros_like(num), where=denom > 1e-8)

    # Arc length (normalized)
    delta = np.sqrt(dx**2 + dy**2)
    arc_length = np.cumsum(delta)
    arc_length /= arc_length[-1]

    return arc_length, kappa

def get_curve(mesh, origin, ax_geom, ax_curv):
    path = get_path(mesh, origin)
    all_loops = group_segments_into_loops(path)
    polylines = [segments_to_polyline(loop) for loop in all_loops if len(loop) > 2]

    # Filter only closed and valid loops
    polylines = [pline for pline in polylines if np.allclose(pline[0], pline[-1], atol=1e-5)]

    # Compute containment relationships
    contained_by = [set() for _ in polylines]
    for i, outer in enumerate(polylines):
        for j, inner in enumerate(polylines):
            if i != j and is_contained(inner, outer):
                contained_by[j].add(i)

    outermost_indices = [i for i, container in enumerate(contained_by) if len(container) == 0]
    outermost_loop = max(outermost_indices, key=lambda i: polygon_area(polylines[i]))

    # Resample and fit
    resampled = resample_polyline_equally(polylines[outermost_loop], n_points=1000)
    ax_geom.scatter(resampled[:, 0], resampled[:, 1], alpha=0.3)

    # Fit spline
    tck = fit_spline_arclength_weighted(resampled, s=50.0, per=1)
    u_eval = np.linspace(0, 1, 1000)
    x_smooth, y_smooth = splev(u_eval, tck)
    ax_geom.plot(x_smooth, y_smooth, 'r')
    ax_geom.plot(x_smooth[0], y_smooth[0], 'o', color='green', markersize=8, label='Start')
    arrow_dx = x_smooth[1] - x_smooth[0]
    arrow_dy = y_smooth[1] - y_smooth[0]
    ax_geom.arrow(x_smooth[0], y_smooth[0], 50*arrow_dx, 50*arrow_dy,
                head_width=2.0, head_length=3.0, fc='green', ec='green', length_includes_head=True)

    # Draw original polylines
    for pline in polylines:
        ax_geom.plot(pline[:, 0], pline[:, 1], 'k--', alpha=0.3)

    ax_geom.set_aspect('equal')
    ax_geom.set_title("Smoothed Contour")

    arc_length, kappa = compute_curvature(tck, u_eval)

    # Curvature plot
    ax_curv.plot(arc_length, kappa, label='Curvature (1/mm)')
    ax_curv.set_xlabel("Normalized Arc Length")
    ax_curv.set_ylabel("Curvature κ")
    ax_curv.set_title("Curvature vs Arc Length")
    ax_curv.grid(True)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
# Run for both meshes
get_curve(mesh_key, origin_key, axs[0, 1], axs[0, 0])
get_curve(mesh_wrench, origin_wrench, axs[1, 1], axs[1, 0])

plt.tight_layout()
plt.show()