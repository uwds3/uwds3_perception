import cv2
import rospy
import numpy as np
import geometry_msgs
from sensor_msgs.msg import Image, CameraInfo
import message_filters
from uwds3_msgs.msg import SceneNodeArrayStamped, SceneChangesStamped, SceneNode, Feature
from cv_bridge import CvBridge
from tf import transformations as tfm
import tf2_ros
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from .detection.opencv_dnn_detector import OpenCVDNNDetector
from .tracking.tracker import Tracker
from .tracking.linear_assignment import iou_distance
from .estimation.shape_estimator import ShapeEstimator

class HumanVisualModel(object):
    FOV = 60.0 # human field of view
    WIDTH = 90 # image width resolution for rendering
    HEIGHT = 68  # image height resolution for rendering
    CLIPNEAR = 0.3 # clipnear
    CLIPFAR = 1e+3 # clipfar
    ASPECT = 1.333 # aspect ratio for rendering
    SACCADE_THRESHOLD = 0.01 # angular variation in rad/s
    SACCADE_ESPILON = 0.005 # error in angular variation
    FOCUS_DISTANCE_FIXATION = 0.1 # focus distance when performing a fixation
    FOCUS_DISTANCE_SACCADE = 0.5 # focus distance when performing a saccade

    def get_camera_info(self):
        camera_info = CameraInfo()
        width = HumanVisualModel.WIDTH
        height = HumanVisualModel.HEIGHT
        camera_info.width = width
        camera_info.height = height
        focal_length = height
        center = (height/2, width/2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                 [0, focal_length, center[1]],
                                 [0, 0, 1]], dtype="double")
        P_matrix = np.array([[focal_length, 0, center[0], 0],
                            [0, focal_length, center[1], 0],
                            [0, 0, 1, 0]], dtype="double")

        dist_coeffs = np.zeros((4, 1))
        camera_info.distortion_model = "blob"
        camera_info.D = list(dist_coeffs)
        camera_info.K = list(camera_matrix.flatten())
        camera_info.P = list(P_matrix.flatten())
        return camera_info

class Uwds3Perception(object):
    def __init__(self):
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)
        self.tf_broadcaster = TransformBroadcaster()

        self.rgb_image_topic = rospy.get_param("~rgb_image_topic", "/camera/rgb/image_raw")
        self.rgb_camera_info_topic = rospy.get_param("~rgb_camera_info_topic", "/camera/rgb/camera_info")

        self.depth_image_topic = rospy.get_param("~depth_image_topic", "/camera/depth/image_raw")
        self.depth_camera_info_topic = rospy.get_param("~depth_camera_info_topic", "/camera/depth/camera_info")

        self.base_frame_id = rospy.get_param("~base_frame_id", "base_link")
        self.global_frame_id = rospy.get_param("~global_frame_id", "odom")

        self.bridge = CvBridge()

        rospy.loginfo("[perception] Subscribing to '/{}' topic...".format(self.depth_camera_info_topic))
        self.camera_info = None
        self.camera_frame_id = None
        self.camera_info_subscriber = rospy.Subscriber(self.depth_camera_info_topic, CameraInfo, self.camera_info_callback)

        self.detector_model_filename = rospy.get_param("~detector_model_filename", "")
        self.detector_weights_filename = rospy.get_param("~detector_weights_filename", "")
        self.detector_config_filename = rospy.get_param("~detector_config_filename", "")

        self.face_detector_model_filename = rospy.get_param("~face_detector_model_filename", "")
        self.face_detector_weights_filename = rospy.get_param("~face_detector_weights_filename", "")
        self.face_detector_config_filename = rospy.get_param("~face_detector_config_filename", "")

        self.body_parts = ["person", "face", "right_hand", "left_hand"]

        self.shape_estimator = ShapeEstimator()

        self.detector = OpenCVDNNDetector(self.detector_model_filename,
                                          self.detector_weights_filename,
                                          self.detector_config_filename,
                                          300)

        self.use_faces = rospy.get_param("~use_faces", True)
        if self.use_faces is True:
            self.face_detector = OpenCVDNNDetector(self.face_detector_model_filename,
                                                   self.face_detector_weights_filename,
                                                   self.face_detector_config_filename,
                                                   300)

        self.n_frame = rospy.get_param("~n_frame", 2)
        self.frame_count = 0

        self.only_human = rospy.get_param("~only_human", False)

        self.use_depth = rospy.get_param("~use_depth", False)

        self.publish_tf = rospy.get_param("~publish_tf", True)

        self.as_provider = rospy.get_param("~as_provider", True)

        self.use_retina_filter = rospy.get_param("~use_retina_filter", False)
        self.retina_filter_config_filename = rospy.get_param("~retina_filter_config_filename", "")
        if self.use_retina_filter is True:
            from .preprocessing.retina_filter import RetinaFilter
            self.retina_filter = RetinaFilter(self.retina_filter_config_filename)

        self.shape_predictor_config_filename = rospy.get_param("~shape_predictor_config_filename", "")

        self.n_init = rospy.get_param("~n_init", 6)
        self.min_iou_distance = rospy.get_param("~min_iou_distance", 0.8)
        self.max_disappeared = rospy.get_param("~max_disappeared", 30)
        self.max_age = rospy.get_param("~max_age", 60)

        self.tracker = Tracker(iou_distance, n_init=self.n_init, min_distance=self.min_iou_distance, max_disappeared=7, max_age=15)

        if self.as_provider is False:
            self.tracks_publisher = rospy.Publisher("tracks", SceneNodeArrayStamped, queue_size=1)
        else:
            self.tracks_publisher = rospy.Publisher("tracks", SceneChangesStamped, queue_size=1)

        self.visualization_publisher = rospy.Publisher("tracks_image", Image, queue_size=1)

        if self.use_depth is True:
            rospy.loginfo("[perception] Subscribing to '/{}' topic...".format(self.rgb_image_topic))
            self.rgb_image_sub = message_filters.Subscriber(self.rgb_image_topic, Image)
            rospy.loginfo("[perception] Subscribing to '/{}' topic...".format(self.depth_image_topic))
            self.depth_image_sub = message_filters.Subscriber(self.depth_image_topic, Image)

            self.sync = message_filters.ApproximateTimeSynchronizer([self.rgb_image_sub, self.depth_image_sub], 10, 0.1, allow_headerless=True)
            self.sync.registerCallback(self.observation_callback)

        else:
            rospy.loginfo("[perception] Subscribing to '/{}' topic...".format(self.rgb_image_topic))
            self.rgb_image_sub = rospy.Subscriber(self.rgb_image_topic, Image, self.observation_callback, queue_size=1)

        self.previous_camera_pose = None
        self.camera_motion = None

    def camera_info_callback(self, msg):
        if self.camera_info is None:
            rospy.loginfo("[perception] Camera info received !")
        self.camera_info = msg
        self.camera_frame_id = msg.header.frame_id
        self.camera_matrix = np.array(msg.K).reshape((3, 3))
        self.dist_coeffs = np.array(msg.D)

    def extract_features(self, tracks):
        pass

    def estimate_depth(self, tracks):
        pass

    def observation_callback(self, bgr_image_msg, depth_image_msg=None):
        if self.camera_info is not None:
            perception_timer = cv2.getTickCount()
            bgr_image = self.bridge.imgmsg_to_cv2(bgr_image_msg, "bgr8")
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            if depth_image_msg is not None:
                depth_image = self.bridge.imgmsg_to_cv2(depth_image_msg)
            else:
                depth_image = None
            viz_frame = bgr_image

            detection_timer = cv2.getTickCount()
            detections = []
            if self.use_faces is True:
                if self.frame_count % self.n_frame == 0:
                    detections = self.detector.detect(rgb_image)
                if self.frame_count % self.n_frame == 1:
                    if self.use_retina_filter is True:
                        bgr_image_filtered = self.retina_filter.filter(bgr_image)
                        rgb_image_filtered = cv2.cvtColor(bgr_image_filtered, cv2.COLOR_BGR2RGB)
                        detections = self.face_detector.detect(rgb_image_filtered)
                    detections = self.face_detector.detect(rgb_image)
            else:
                if self.frame_count % self.n_frame == 0:
                    detections = self.detector.detect(rgb_image)
                else:
                    detections = []
            self.frame_count += 1
            detection_fps = cv2.getTickFrequency() / (cv2.getTickCount() - detection_timer)

            tracking_timer = cv2.getTickCount()
            if self.only_human is False:
                tracks = self.tracker.update(rgb_image, detections, self.camera_matrix, self.dist_coeffs)
            else:
                detections = [d for d in detections if d.class_label in self.body_parts]
            tracks = self.tracker.update(rgb_image, detections, self.camera_matrix, self.dist_coeffs, depth_image=depth_image)

            tracking_fps = cv2.getTickFrequency() / (cv2.getTickCount() - tracking_timer)
            perception_fps = cv2.getTickFrequency() / (cv2.getTickCount() - perception_timer)
            detection_fps_str = "Detection fps : {:0.4f}hz".format(detection_fps)
            tracking_fps_str = "Tracking and pose estimation fps : {:0.4f}hz".format(tracking_fps)
            perception_fps_str = "Perception fps : {:0.4f}hz".format(perception_fps)

            cv2.putText(viz_frame, "Nb detections/tracks : {}/{}".format(len(detections), len(tracks)), (5, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(viz_frame, detection_fps_str, (5, 45),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(viz_frame, tracking_fps_str, (5, 65),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(viz_frame, perception_fps_str, (5, 85),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            if self.as_provider is not False:
                scene_changes = SceneChangesStamped()
                scene_changes.header = bgr_image_msg.header
            else:
                entity_array = SceneNodeArrayStamped()
                entity_array.header = bgr_image_msg.header

            for track in tracks:
                self.draw_track(viz_frame, track, self.camera_matrix, self.dist_coeffs)
                if track.is_confirmed() or track.is_occluded():
                    success, t, q = self.get_transform_from_tf2(self.global_frame_id, self.camera_frame_id, time=bgr_image_msg.header.stamp)
                    if success:
                        if track.rotation is not None and track.translation is not None:
                            tf_track = np.dot(tfm.translation_matrix(track.translation), tfm.euler_matrix(track.rotation[0], track.rotation[1], track.rotation[2], "rxyz"))
                            tf_sensor = np.dot(tfm.translation_matrix(t), tfm.quaternion_matrix(q))
                            tf_track_global = np.dot(tf_sensor, tf_track)
                            q_final = tfm.quaternion_from_matrix(tf_track_global)
                            t_final = tfm.translation_from_matrix(tf_track_global)
                            transform = geometry_msgs.msg.TransformStamped()
                            transform.header = bgr_image_msg.header
                            transform.header.frame_id = self.global_frame_id
                            transform.child_frame_id = track.class_label+"_"+track.uuid.replace("-", "")
                            transform.transform.translation.x = t_final[0]
                            transform.transform.translation.y = t_final[1]
                            transform.transform.translation.z = t_final[2]
                            if track.class_label == "face":
                                transform.transform.rotation.x = q_final[0]
                                transform.transform.rotation.y = q_final[1]
                                transform.transform.rotation.z = q_final[2]
                                transform.transform.rotation.w = q_final[3]
                            else:
                                transform.transform.rotation.x = 0.0
                                transform.transform.rotation.y = 0.0
                                transform.transform.rotation.z = 0.0
                                transform.transform.rotation.w = 1.0
                            self.tf_broadcaster.sendTransform(transform)

                        entity = SceneNode()
                        entity.label = track.class_label
                        entity.id = track.class_label+"_"+track.uuid.replace("-", "")
                        if track.translation is not None and track.rotation is not None:
                            entity.is_located = True
                            entity.position_with_cov.header = bgr_image_msg.header
                            entity.position_with_cov.header.frame_id = self.global_frame_id
                            entity.position_with_cov.pose.pose.position.x = t_final[0]
                            entity.position_with_cov.pose.pose.position.y = t_final[1]
                            entity.position_with_cov.pose.pose.position.z = t_final[2]
                            if track.class_label == "face":
                                entity.position_with_cov.pose.pose.orientation.x = q_final[0]
                                entity.position_with_cov.pose.pose.orientation.y = q_final[1]
                                entity.position_with_cov.pose.pose.orientation.z = q_final[2]
                                entity.position_with_cov.pose.pose.orientation.w = q_final[3]
                            else:
                                entity.position_with_cov.pose.pose.orientation.x = 0.0
                                entity.position_with_cov.pose.pose.orientation.y = 0.0
                                entity.position_with_cov.pose.pose.orientation.z = 0.0
                                entity.position_with_cov.pose.pose.orientation.w = 1.0
                        else:
                            entity.is_located = False

                        success, shape = self.shape_estimator.estimate(track, self.camera_matrix, self.dist_coeffs)
                        entity.has_shape = success
                        if success is True:
                            if track.class_label == "person":
                                new_dim = []
                                shape_height = shape.dimensions[2]
                                center_height = t_final[2]
                                height = center_height + shape_height/2
                                new_dim.append(shape.dimensions[0])
                                new_dim.append(shape.dimensions[1])
                                new_dim.append(height)
                                shape.dimensions = new_dim
                                shape.pose.position.z = - (center_height - height/2)
                            entity.shape = shape

                        if "facial_landmarks" in track.properties:
                            feature = []
                            #TODO remove this for loop
                            #track.properties["facial_landmarks"][0]
                            for (x, y) in track.properties["facial_landmarks"]:
                                feature.append(float(x)/rgb_image.shape[0])
                                feature.append(float(y)//rgb_image.shape[1])
                            entity.features.append(Feature(name="facial_landmarks", data=feature))

                        if entity.label == "face":
                            entity.has_camera = True
                            entity.camera = HumanVisualModel().get_camera_info()
                            entity.camera.header.frame_id = track.class_label+"_"+track.uuid[:6]
                        entity.last_update = bgr_image_msg.header.stamp
                        entity.expiration_time = bgr_image_msg.header.stamp + rospy.Duration(3.0)
                        if self.as_provider is not False:
                            scene_changes.changes.nodes.append(entity)
                        else:
                            entity_array.nodes.append(entity)

            viz_img_msg = self.bridge.cv2_to_imgmsg(viz_frame)
            if self.as_provider is not False:
                self.tracks_publisher.publish(scene_changes)
            else:
                self.tracks_publisher.publish(entity_array)
            self.visualization_publisher.publish(viz_img_msg)

    def get_transform_from_tf2(self, source_frame, target_frame, time=None):
        try:
            if time is not None:
                trans = self.tf_buffer.lookup_transform(source_frame, target_frame, time)
            else:
                trans = self.tf_buffer.lookup_transform(source_frame, target_frame, rospy.Time(0))
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            z = trans.transform.translation.z

            rx = trans.transform.rotation.x
            ry = trans.transform.rotation.y
            rz = trans.transform.rotation.z
            rw = trans.transform.rotation.w

            return True, [x, y, z], [rx, ry, rz, rw]
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("[perception] Exception occured: {}".format(e))
            return False, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]

    def draw_track(self, rgb_image, track, camera_matrix, dist_coeffs):
        tl_corner = (int(track.bbox.left()), int(track.bbox.top()))
        br_corner = (int(track.bbox.right()), int(track.bbox.bottom()))
        if track.rotation is not None and track.translation is not None:
            cv2.drawFrameAxes(rgb_image, camera_matrix, dist_coeffs, np.array(track.rotation).reshape((3,1)), np.array(track.translation).reshape(3,1), 0.03)
        if track.class_label == "face":
            if "facial_landmarks" in track.properties:
                previous_point = None
                for pxd, (x, y) in enumerate(track.properties["facial_landmarks"]):
                    if previous_point is not None:
                        cv2.line(rgb_image, previous_point, (x, y), (0, 255, 255), thickness=1)
                    if pxd == 16 or pxd == 21 or pxd == 26 or pxd == 30 or pxd == 35 or pxd == 41 or pxd == 47:
                        previous_point = None
                    else:
                        previous_point = x, y

        if track.is_confirmed() is True:
            cv2.putText(rgb_image, track.uuid[:6], (tl_corner[0]+5, tl_corner[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 250, 250), 1)
            cv2.putText(rgb_image, track.class_label, (tl_corner[0]+5, tl_corner[1]+45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 250, 250), 1)
            cv2.rectangle(rgb_image, tl_corner, br_corner, (0, 255, 0), 1)
        elif track.is_occluded() is True:
            cv2.rectangle(rgb_image, tl_corner, br_corner, (0, 0, 250), 1)
        else:
            cv2.rectangle(rgb_image, tl_corner, br_corner, (200, 200, 0), 1)
