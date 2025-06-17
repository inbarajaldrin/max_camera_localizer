# Aruco-Based Localization for Camera mounted on UR End-Effector

This is a ROS 2 Humble Package that outputs aruco tag position and orientation *in the robot base frame* using camera inputs and ROS 2 communication with the UR driver. 

## Setup
Usage of this package will require the UR ROS 2 driver and cv2. 

To install, place this repository in the src directory of your workspace. Also install the repository `max_camera_msgs` in the same location. Then, build your workspace. 

To run, first make sure a UR driver is running. Then, run `ros2 run max_camera_localizer localize` inside your workspace. 

A preview window will appear, picking one camera accessible to the system. Press ESC to select this camera, or press any other key to move onto the next camera. Once you select your camera, a more detailed window will appear marking Aruco position and annotating details in the top left corner. If you know your camera ID and would like to skip this process, pass the additional arg `--camera-id [YOUR_ID]`. 

While running, this script will publish estimated camera pose (in the UR frame) in `/camera_pose` as well as poses of any detected objects in `/marker_poses/marker_[MARKER-ID]`. Each marker gets its own subtopic. Poses are given as `PoseStamped`.  

This script will doubly print its findings to console. If you wish, you may pass the argument `--suppress-prints` to not see this.

This script assumes a fixed translation from the end effector frame origin and the focal point of the camera. If the camera is ever repositioned, please measure or calibrate, and update the translation or rotation in `aruco_pose_bridge.py`

## Example Command:

`ros2 run max_camera_localizer localize --camera-id 8 --suppress-prints`

## Workspace Structure:

```
YOUR_ROS2_WORKSPACE/
├── src/
│   └── max_camera_localizer/
│       ├── max_camera_localizer/
│       │   ├── trash/
│       │   │   └── [unused python files]
│       │   ├── aruco_pose_bridge.py
│       │   ├── camera_selection.py
│       │   ├── detection_functions.py
│       │   ├── geometric_functions.py
│       │   ├── kalman_functions.py
│       │   ├── merged_localization.py
│       │   ├── object_frame_definitions
│       │   └── process_stl.py
│       ├── STL/
│       │   ├── Allen Key.STL
│       │   └── Wrench.STL
│       ├── package.xml
│       ├── README.md
│       ├── setup.cfg
│       └── setup.py
```