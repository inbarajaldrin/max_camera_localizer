# Aruco-Based Localization for Camera mounted on UR End-Effector

This is a ROS 2 Humble Package that outputs aruco tag position and orientation *in the robot base frame* using camera inputs and ROS 2 communication with the UR driver. 

## Setup
Usage of this package will require the UR ROS 2 driver and cv2. 

To install, place this repository in the src directory of your workspace. Then, build your workspace. 

To run, first make sure a UR driver is running. Then, run `ros2 run max_camera_localizer localize` inside your workspace. 
If you wish to test your driver connection, run `ros2 run max_camera_localizer test_ros` instead.

A preview window will appear, picking one camera accessible to the system. Press ESC to select this camera, or press any other key to move onto the next camera. Once you select your camera, a more detailed window will appear marking Aruco position and annotating details in the top left corner. 
