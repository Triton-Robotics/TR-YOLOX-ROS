# YOLOX-ROS

## TR setup 

### Installing TensorRT

Go to https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html, and install with option 2: Debian Package Installation. Ensure that your TensorRT is compatible with your CUDA version.

#### Aside: CUDA compatibility for older devices
For the few with older GPUs, note that while the CUDA driver is backwards compatible with most older GPUs, modules like PyTorch and TensorRT are not. In such a case, you must identify what your Compute Capability (CC) is (https://developer.nvidia.com/cuda/gpus/legacy) and then find a version (preferably the newest) of your desired package that is compatible with your CC. Using the wayback machine, you can find the version for TensorRT that supports your CC.

For example, suppose you have an Quadro M1000M. Checking the CC table linked shows that our CC is 5.0. Now, using the wayback machine (since this archive page is no longer active), we can go through the documentation for different versions of TensorRT (https://web.archive.org/web/20260116063126/https://docs.nvidia.com/deeplearning/tensorrt/archives/index.html), and check the `support matrix` tab for each version. As it turns out, 8.4.3 is the latest version of TensorRT that supports CC 5.0, so you should install TensorRT 8.4.3.

If you cannot find the associated debian package, you can use the .tar installation method instead.

If you have just installed CUDA, beware that you might end up breaking your torch and torchvision modules. You must also install the corresponding cudnn library demanded by TensorRT if such an error occurs.

### before you build 
`sudo apt install ros-humble-generate-parameter-library`

### Converting the model
Navigate to `.../TR-Autonomy/src/TR-YOLOX-ROS/weights/tensorrt`. Then run the command
```
$ ./convert.bash armor_tiny
```

### building
colcon build --cmake-args -DYOLOX_USE_TENSORRT=ON

### running
`ros2 launch yolox_ros_cpp yolox_tensorrt.launch.py`


![](https://img.shields.io/github/stars/Ar-Ray-code/YOLOX-ROS)

[![iron](https://github.com/Ar-Ray-code/YOLOX-ROS/actions/workflows/ci_iron.yml/badge.svg?branch=iron)](https://github.com/Ar-Ray-code/YOLOX-ROS/actions/workflows/ci_iron.yml)


[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) + ROS2 Iron demo

![yolox_s_result](https://github.com/Ar-Ray-code/RenderTexture2ROS2Image/blob/main/images_for_readme/unity-demo.gif?raw=true)

<div align="center">🔼 Unity + YOLOX-ROS Demo</div>

## Supported List

| Base            | ROS2 C++ |
| --------------- | -------- |
| TensorRT (CUDA) |  ✅       |
| OpenVINO        |  ✅       |
| ONNX Runtime    |  ✅       |
| TFLite          |  ✅       |


## Installation & Demo (C++)

Check [this URL](./yolox_ros_cpp/README.md).

<br>

## Topic

### Subscribe

- image_raw (`sensor_msgs/Image`)

### Publish

<!-- - yolox/image_raw : Resized image (`sensor_msgs/Image`) -->

- bounding_boxes (`bboxes_ex_msgs/BoundingBoxes` or `vision_msgs/Detection2DArray`)
  - `bboxes_ex_msgs/BoundingBoxes`: Output BoundingBoxes like darknet_ros_msgs
  - ※ If you want to use `darknet_ros_msgs` , replace `bboxes_ex_msgs` with `darknet_ros_msgs`.

<!-- ![yolox_topic](images_for_readme/yolox_topic.png) -->

<br>

##

## Reference

![](https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/main/assets/logo.png)

- [YOLOX (GitHub)](https://github.com/Megvii-BaseDetection/YOLOX)

```
@article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```

<br>

## Contributors

<a href="https://github.com/Ar-Ray-code/YOLOX-ROS/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Ar-Ray-code/YOLOX-ROS" />
</a>

<br>
