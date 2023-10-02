## YOLO3D ROS Node

This repo contains a Monocular 3D detection Ros node. Base on https://github.com/Owen-Liuyuxuan/visualDet3D.

**Update (2023-10-2)**: This repo now also supports inference from [FSNet/VisionFactory](https://github.com/Owen-Liuyuxuan/visionfactory). Use the ```mono3d_node.py``` and the ```mono3d.launch``` instead. Existing models from visualDet3D are still well-supported by ```yolo3d_node.py/yolo3d.launch```

We have also provided a [cookbook](https://owen-liuyuxuan.github.io/papers_reading_sharing.github.io/3dDetection/my_cookbook/#synthetic-cookbook-for-usingtestingdemonstrating-visualdet3d-in-ros) on how to utilize the open-source tool chains. 

All parameters are exposed in the launch file.

![image](doc/yolo3d_ros_realtime.gif)

**Notice**: This node only takes image and camera_info as input, the outputs are bounding boxes. 


### Setup

Install ROS, tested on Ubuntu 18.04, ROS melodic.

Enable rospy in Python3 (this should not affect Python2), but it **does not** enable **tf** in Python3.
```bash
sudo apt-get install python3-catkin-pkg-modules
sudo apt-get install python3-rospkg-modules
```

Clone this repo into a ROS workspace and run
```bash
catkin_make
source devel/setup.bash
```
under the workspace folder.

Also modify the launch file.

### Subscribed Topics

image_raw ([sensor_msgs/Image](http://docs.ros.org/en/api/sensor_msgs/html/msg/Image.html))

A stream of rectifiled image to be predicted using monodepth.

camera_info ([sensor_msgs/CameraInfo](http://docs.ros.org/en/api/sensor_msgs/html/msg/CameraInfo.html))

Camera calibration information of the rectified image.

### Published Topics

bboxes ([visualization_msgs/MarkerArray](http://docs.ros.org/en/noetic/api/visualization_msgs/html/msg/MarkerArray.html))

Predicted bboxes 3D objects.
