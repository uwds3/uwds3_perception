import cv2
import rospy
import math
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
import message_filters
from uwds3_msgs.msg import SceneNodeArrayStamped, SceneChangesStamped
from cv_bridge import CvBridge
import tf2_ros
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from .detection.opencv_dnn_detector import OpenCVDNNDetector
from .tracking.multi_object_tracker import MultiObjectTracker, iou_cost, face_cost, color_cost
from .estimation.shape_estimator import ShapeEstimator
from .estimation.head_pose_estimator import HeadPoseEstimator
from .estimation.facial_landmarks_estimator import FacialLandmarksEstimator
from .estimation.facial_features_extractor import FacialFeaturesExtractor
from .estimation.appearance_features_extractor import AppearanceFeaturesExtractor
from .estimation.color_features_extractor import ColorFeaturesExtractor
from .types.vector import Vector6D


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

        detector_model_filename = rospy.get_param("~detector_model_filename", "")
        detector_weights_filename = rospy.get_param("~detector_weights_filename", "")
        detector_config_filename = rospy.get_param("~detector_config_filename", "")

        face_detector_model_filename = rospy.get_param("~face_detector_model_filename", "")
        face_detector_weights_filename = rospy.get_param("~face_detector_weights_filename", "")
        face_detector_config_filename = rospy.get_param("~face_detector_config_filename", "")

        self.shape_estimator = ShapeEstimator()

        self.detector = OpenCVDNNDetector(detector_model_filename,
                                          detector_weights_filename,
                                          detector_config_filename,
                                          300)

        self.use_faces = rospy.get_param("~use_faces", True)
        if self.use_faces is True:
            self.face_detector = OpenCVDNNDetector(face_detector_model_filename,
                                                   face_detector_weights_filename,
                                                   face_detector_config_filename,
                                                   300)
            shape_predictor_config_filename = rospy.get_param("~shape_predictor_config_filename", "")
            self.facial_landmarks_estimator = FacialLandmarksEstimator(shape_predictor_config_filename)
            self.head_pose_estimator = HeadPoseEstimator()
            self.face_of_interest_uuid = None

            facial_features_model_filename = rospy.get_param("~facial_features_model_filename", "")
            self.facial_features_extractor = FacialFeaturesExtractor(facial_features_model_filename)

        self.appearance_features_extractor = AppearanceFeaturesExtractor(model_type="MobileNetV2")

        self.color_features_extractor = ColorFeaturesExtractor()

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
        self.min_color_distance = rospy.get_param("~min_color_distance", 0.8)
        self.min_face_distance = rospy.get_param("~min_face_distance", 0.8)
        self.max_disappeared = rospy.get_param("~max_disappeared", 30)
        self.max_age = rospy.get_param("~max_age", 60)

        self.object_tracker = MultiObjectTracker(iou_cost,
                                                 color_cost,
                                                 self.min_iou_distance,
                                                 self.min_color_distance,
                                                 self.n_init,
                                                 self.max_disappeared,
                                                 self.max_age,
                                                 tracker_type=None)

        self.face_tracker = MultiObjectTracker(iou_cost,
                                               face_cost,
                                               self.min_iou_distance,
                                               self.min_face_distance,
                                               self.n_init,
                                               self.max_disappeared,
                                               self.max_age,
                                               tracker_type=None)

        self.person_tracker = MultiObjectTracker(iou_cost,
                                                 color_cost,
                                                 self.min_iou_distance,
                                                 self.min_color_distance,
                                                 self.n_init,
                                                 self.max_disappeared,
                                                 self.max_age,
                                                 tracker_type=None)

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

            _, image_height, image_width = bgr_image.shape

            view = self.get_pose_from_tf2(self.global_frame_id, self.camera_frame_id)
            ######################################################
            # Detection
            ######################################################
            detection_timer = cv2.getTickCount()

            detections = []
            if self.use_faces is True:
                if self.frame_count % self.n_frame == 0:
                    detections = self.detector.detect(rgb_image)
                if self.frame_count % self.n_frame == 1:
                    if self.use_retina_filter is True:
                        bgr_image_filtered = self.retina_filter.filter(bgr_image)
                        rgb_image_filtered = cv2.cvtColor(bgr_image_filtered,
                                                          cv2.COLOR_BGR2RGB)
                        detections = self.face_detector.detect(rgb_image_filtered)
                    detections = self.face_detector.detect(rgb_image)
            else:
                if self.frame_count % self.n_frame == 0:
                    detections = self.detector.detect(rgb_image)
                else:
                    detections = []

            detection_fps = cv2.getTickFrequency() / (cv2.getTickCount() - detection_timer)
            ####################################################################
            # Features extraction
            ####################################################################
            features_timer = cv2.getTickCount()

            if self.frame_count % self.n_frame == 1:
                self.facial_features_extractor.extract(rgb_image, detections)
            else:
                self.color_features_extractor.extract(rgb_image, detections)

            features_fps = cv2.getTickFrequency() / (cv2.getTickCount() - features_timer)
            ######################################################
            # Tracking
            ######################################################
            tracking_timer = cv2.getTickCount()

            if self.frame_count % self.n_frame == 1:
                face_tracks = self.face_tracker.update(rgb_image, detections, view, self.camera_matrix, self.dist_coeffs)
                object_tracks = self.object_tracker.update(rgb_image, [], view, self.camera_matrix, self.dist_coeffs)
                person_tracks = self.person_tracker.update(rgb_image, [], view, self.camera_matrix, self.dist_coeffs)
            else:
                object_detections = [d for d in detections if d.label != "person"]
                person_detections = [d for d in detections if d.label == "person"]
                face_tracks = self.face_tracker.update(rgb_image, [], view, self.camera_matrix, self.dist_coeffs)
                object_tracks = self.object_tracker.update(rgb_image, object_detections, view, self.camera_matrix, self.dist_coeffs)
                person_tracks = self.person_tracker.update(rgb_image, person_detections, view, self.camera_matrix, self.dist_coeffs)
            tracks = face_tracks + object_tracks + person_tracks

            tracking_fps = cv2.getTickFrequency() / (cv2.getTickCount() - tracking_timer)

            ########################################################
            # Head pose estimation
            ########################################################
            head_pose_timer = cv2.getTickCount()


            face_tracks = [t for t in tracks if t.label=="face" and t.is_confirmed()]

            self.facial_features_extractor.extract(rgb_image, face_tracks)

            face_of_interest = None
            face_of_interest_uuid = None

            cx = image_width/2
            cy = image_height/2
            min_dist = 10000

            for face in face_tracks:
                if self.face_of_interest_uuid is not None:
                    if self.face_of_interest_uuid != face.uuid:
                        if "facial_landmarks" in face.features:
                            del face.features["facial_landmarks"]
                            face.translation = None
                            face.rotation = None
                face_x = face.bbox.center().x
                face_y = face.bbox.center().y
                distance_from_center = math.sqrt(pow(cx-face_x, 2)+pow(cy-face_y, 2))

                if min_dist > distance_from_center:
                    face_of_interest_uuid = face.uuid
                    face_of_interest = face
                    min_dist = distance_from_center

            if face_of_interest is not None:
                landmarks = self.facial_landmarks_estimator.estimate(rgb_image, face)

                # if face_of_interest.is_located() is False:
                #     success, trans, rot = self.head_pose_estimator.estimate(landmarks, self.camera_matrix, self.dist_coeffs)
                # else:
                #     success, trans, rot = self.head_pose_estimator.estimate(landmarks, self.camera_matrix, self.dist_coeffs, previous_head_pose=face_of_interest.pose)
                # if success is True:
                #     if depth_image is not None:
                #         success, trans_depth = self.translation_estimator.estimate(face.bbox, depth_image, self.camera_matrix, self.dist_coeffs)
                #         if success:
                #             face_of_interest.update_pose(trans_depth, rot)
                #         else:
                #             face_of_interest.update_pose(trans, rot)
                #     else:
                #         face_of_interest.update_pose(trans, rot)
                face_of_interest.features["facial_landmarks"] = landmarks

                self.face_of_interest_uuid = face_of_interest_uuid

            head_pose_fps = cv2.getTickFrequency() / (cv2.getTickCount() - head_pose_timer)

            ######################################################
            # Depth & Shape estimation
            ######################################################
            # depth_shape_timer = cv2.getTickCount()
            #
            # if depth_image_msg is not None:
            #     for track in tracks:
            #         if track.translation is None and track.rotation is None:
            #             success, trans = self.translation_estimator.estimate(track.bbox, depth_image, self.camera_matrix, self.dist_coeffs)
            #             if success is True:
            #                 rot = np.array([math.pi/2, 0.0, 0.0])
            #                 track.filter(rot, trans)
            #
            # depth_shape_fps = cv2.getTickFrequency() / (cv2.getTickCount() - depth_shape_timer)

            ######################################################
            # Visualization of debug image and tf publication
            ######################################################
            perception_fps = cv2.getTickFrequency() / (cv2.getTickCount() - perception_timer)
            #
            # detection_load = perception_fps / detection_fps
            # features_load = features_fps / perception_fps
            # tracking_load = perception_fps / tracking_fps
            # head_pose_load = perception_fps / head_pose_fps
            #
            # cv2.rectangle(viz_frame, (0, 0), (int(image_width*detection_load), 20), (200,0,0), -1)
            # cv2.rectangle(viz_frame, (int(image_width*detection_load), 0), (int(image_width*features_load), 20), (200,200,0), -1)
            # cv2.rectangle(viz_frame, (int(image_width*features_load), 0), (int(image_width*tracking_load), 20), (200,0,200), -1)
            # cv2.rectangle(viz_frame, (int(image_width*tracking_load), 0), (int(image_width*head_pose_load), 20), (200,150,60), -1)

            cv2.rectangle(viz_frame, (0, 0), (250, 40), (200, 200, 200), -1)
            perception_fps_str = "Perception fps : {:0.1f}hz".format(perception_fps)
            cv2.putText(viz_frame, "Nb detections/tracks : {}/{}".format(len(detections), len(tracks)), (5, 15),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(viz_frame, perception_fps_str, (5, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            if self.as_provider is not False:
                scene_changes = SceneChangesStamped()
                scene_changes.header = bgr_image_msg.header
            else:
                entity_array = SceneNodeArrayStamped()
                entity_array.header = bgr_image_msg.header

            header = bgr_image_msg.header
            for track in tracks:

                track.draw(viz_frame, (230, 0, 120, 125), 1)

                if track.pose is not None:
                    success, t, q = self.get_transform_from_tf2(self.global_frame_id,
                                                                self.camera_frame_id,
                                                                time=header.stamp)
                    if success:
                        sensor_vector = Vector6D(x=t[0], y=t[1], z=t[2]).from_quaternion(q[0], q[1], q[2], q[3])
                        header.frame_id = self.global_frame_id
                        track.pose = sensor_vector + track.pose
                    else:
                        track.pose = None

                if self.as_provider is not False:
                    header = bgr_image_msg.header
                    header.frame_id = self.global_frame_id
                    scene_changes.changes.nodes.append(track.to_msg(header))
                else:
                    entity_array.nodes.append(track.to_msg(header))

            viz_img_msg = self.bridge.cv2_to_imgmsg(viz_frame)
            if self.as_provider is not False:
                self.tracks_publisher.publish(scene_changes)
            else:
                self.tracks_publisher.publish(entity_array)
            self.visualization_publisher.publish(viz_img_msg)

            self.frame_count += 1

    def get_pose_from_tf2(self, source_frame, target_frame, time=None):
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

            return True, Vector6D(x=0.0, y=0.0, z=0.0).from_quaternion(rx, ry, rz, rw)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("[perception] Exception occured: {}".format(e))
            return False, Vector6D(x=0.0, y=0.0, z=0.0).from_quaternion(0.0, 0.0, 0.0, 1.0)
