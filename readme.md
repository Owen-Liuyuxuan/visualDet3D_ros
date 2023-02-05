## YOLO3D ROS Node

This repo contains a Monocular 3D detection Ros node. Base on https://github.com/Owen-Liuyuxuan/visualDet3D.

We have also provided a [cookbook](https://owen-liuyuxuan.github.io/papers_reading_sharing.github.io/3dDetection/my_cookbook/#synthetic-cookbook-for-usingtestingdemonstrating-visualdet3d-in-ros) on how to utilize the open-source tool chains. 

All parameters are exposed in the launch file.

![image](doc/yolo3d_ros_realtime.gif)

**Notice**: This node only takes image and camera_info as input, the outputs are bounding boxes. 

### Subscribed Topics

image_raw ([sensor_msgs/Image](http://docs.ros.org/en/api/sensor_msgs/html/msg/Image.html))

A stream of rectifiled image to be predicted using monodepth.

camera_info ([sensor_msgs/CameraInfo](http://docs.ros.org/en/api/sensor_msgs/html/msg/CameraInfo.html))

Camera calibration information of the rectified image.

### Published Topics

bboxes ([visualization_msgs/MarkerArray](http://docs.ros.org/en/noetic/api/visualization_msgs/html/msg/MarkerArray.html))

Predicted bboxes 3D objects.
