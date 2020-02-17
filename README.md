# Underworlds perception
A lightweight, fast and robust perception stack for HRI research that run on **CPU only**. Despite it's relative simplicity this perception layer can achieve relativelly good performances to be used in research scenarios.

This package is designed to be part of the Underworlds stack but can be used alone.

This perception stack is an example of provider for the Underworlds architecture. To achieve realtime performances, single shot detectors optimized for embedded devices are used in conjunction with medianflow/kalman trackers. Each track is assigned using two assignement (iou then features or centroid based). Thanks to dlib face alignement the face landmarks and head pose are computed. Dominent color extraction and 3d shape estimation based on simple primitives are also provided to create a 3D model of the scene for the robot. Facial recognition is provided thanks to openface.

Weakness in the object detector are compensated by using single object trackers, thus the objects need to be detected at least n_init times before being correcly tracker.

Note: The head pose is provided to be able to compute the human perspective for HRI scenario and thus is in z-forward convention (for scene rendering).

## How to install

```
cd uwds3_ws/src
git clone https://github.com/uwds3/uwds3_msgs.git
git clone https://github.com/uwds3/uwds3.git
git clone https://github.com/uwds3/uwds3_perception.git
cd uwds3_perception
./install_dependencies.sh
./download_models.sh
cd ..
catkin build
```

## How to use

To launch the perception layer, simply do:

```
roslaunch uwds3_perception underworlds_perception.launch
```

To run the demo, you can use your laptop camera by using this command you will upload a fake robot that allow you to test the perception layer:

```
roslaunch uwds3_perception camera_publisher.launch
```

### Parameters

* `rgb_image_topic`: The input image topic
* `rgb_camera_info_topic`: The rgb camera info topic
* `depth_image_topic`: The depth image topic
* `depth_camera_info_topic`: The depth camera info topic
* `global_frame_id`: The global frame (default `map`)
* `detector_model_filename`: The detector model (default SSD/MobilenetV2 mscoco)
* `detector_weights_filename`: The detector weights (default SSD/MobilenetV2 mscoco)
* `detector_config_filename`: The detector yaml config file
* `face_detector_model_filename`: The face detector model (default opencv face SSD)
* `face_detector_weights_filename`: The face detector weights (default opencv face SSD)
* `face_detector_config_filename`: The face detector yaml config file
* `shape_predictor_config_filename`: The dlib shape predictor file
* `face_3d_model_filename`: The 3D face model used for face fitting
* `facial_features_model_filename`: The facial features extraction model (default openface)
* `n_init`: The number of observation before considering a track confirmed
* `n_frame`: The N frames (default 2)
* `max_iou_distance`: The maximum iou distance used for matching
* `max_color_distance`: The maximum color distance used for matching
* `max_face_distance`: The maximum face distance used for matching
* `max_centroid_distance`: The maximum centroid distance used for matching
* `max_disappeared`: The number of missed observation before considering occluded
* `max_age`: The maximum age of occluded tracks
* `publish_visualization_image`: If true, publish the debug image
* `publish_tf`: If true, publish the `\tf` frames for each located confirmed track
* `use_depth`: If true, use the depth image to provide shape and depth estimation

### Subscribers

* `rgb_image_topic` of type `Image`: the input rgb image
* `depth_image_topic` of type `Image`: the input depth image if `use_depth` is true.


### Publishers

* `tracks` of type `SceneChangesStamped` : the ouput world state

## How to contribute

To contribute to this project, fork it and make a pull request.

#### TODO list
* [ ] enchance single object tracker by matching keypoints for robust tracking
* [ ] provide a saliency detector for one-shot tasks
* [ ] provide a salient keypoints detector (like kadir-brady) for tracking and recognition

#### Known issues

* Weird roll behavior of head pose estimation in some singular positions (due to the kalman motion on rotations)
* The single object tracker sometimes tracks the background

## References
