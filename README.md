# uwds3_perception
A lightweight, fast and robust perception stack for HRI research

This package is designed to be part of the Underwrolds stack but can be used alone.

## How to install

```
cd uwds3_ws/src
git clone https://github.com/uwds3/uwds3_msgs.git
git clone https://github.com/uwds3/uwds3_perception.git
./install_dependencies.sh
./download_models.sh
cd ..
catkin build
```

## How to use

To launch the perception layer, do:

```
roslaunch uwds3_perception underworlds_perception.launch
```

### Parameters

* `rgb_image_topic`: The input image topic
* `rgb_camera_info_topic`: The rgb camera info topic
* `depth_image_topic`: The depth image topic
* `depth_camera_info_topic`: The depth camera info topic
* `global_frame_id`: The global frame (default `map`)
* `detector_model_filename`: The dnn detector model (default SSD/MobilenetV2 mscoco)
* `detector_weights_filename`: The dnn detector weights (default SSD/MobilenetV2 mscoco)

### Subscribers


### Publishers
