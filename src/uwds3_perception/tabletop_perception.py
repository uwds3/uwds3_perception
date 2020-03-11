import cv2
import rospy
import math
import numpy as np
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, CameraInfo
import message_filters
from uwds3_msgs.msg import SceneChangesStamped
from cv_bridge import CvBridge
import tf2_ros
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from pyuwds3.types.detection import Detection
from pyuwds3.types.shape.box import Box
from .tracking.track import Track
from .detection.ssd_detector import SSDDetector
from .detection.foreground_detector import ForegroundDetector
from .tracking.multi_object_tracker import MultiObjectTracker, iou_cost, color_cost, centroid_cost
from .estimation.head_pose_estimator import HeadPoseEstimator
from .estimation.object_pose_estimator import ObjectPoseEstimator
from .estimation.shape_estimator import ShapeEstimator
from .estimation.facial_landmarks_estimator import FacialLandmarksEstimator
from .estimation.facial_features_estimator import FacialFeaturesEstimator
from .estimation.color_features_estimator import ColorFeaturesEstimator

from pyuwds3.types.vector.vector6d import Vector6D


class TabletopPerception(object):
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

        rospy.loginfo("[tabletop_perception] Subscribing to '/{}' topic...".format(self.depth_camera_info_topic))
        self.camera_info = None
        self.camera_frame_id = None
        self.camera_info_subscriber = rospy.Subscriber(self.rgb_camera_info_topic,
                                                       CameraInfo,
                                                       self.camera_info_callback)

        detector_model_filename = rospy.get_param("~detector_model_filename", "")
        detector_weights_filename = rospy.get_param("~detector_weights_filename", "")
        detector_config_filename = rospy.get_param("~detector_config_filename", "")

        hand_detector_model_filename = rospy.get_param("~hand_detector_model_filename", "")
        hand_detector_weights_filename = rospy.get_param("~hand_detector_weights_filename", "")
        hand_detector_config_filename = rospy.get_param("~hand_detector_config_filename", "")

        self.detector = SSDDetector(detector_model_filename,
                                    detector_weights_filename,
                                    detector_config_filename,
                                    300)

        self.hand_detector = SSDDetector(hand_detector_model_filename,
                                         hand_detector_weights_filename,
                                         hand_detector_config_filename,
                                         300)

        self.foreground_detector = ForegroundDetector(interactive_mode=False)

        self.color_features_estimator = ColorFeaturesEstimator()

        self.n_frame = rospy.get_param("~n_frame", 3)
        self.frame_count = 0

        self.publish_tf = rospy.get_param("~publish_tf", True)
        self.publish_visualization_image = rospy.get_param("~publish_visualization_image", True)

        self.n_init = rospy.get_param("~n_init", 1)
        self.max_iou_distance = rospy.get_param("~max_iou_distance", 0.8)
        self.max_color_distance = rospy.get_param("~max_color_distance", 0.2)
        self.max_centroid_distance = rospy.get_param("~max_centroid_distance", 0.8)
        self.max_disappeared = rospy.get_param("~max_disappeared", 7)
        self.max_age = rospy.get_param("~max_age", 10)

        self.object_tracker = MultiObjectTracker(iou_cost,
                                                 color_cost,
                                                 0.35,
                                                 0.3,
                                                 self.n_init,
                                                 40,
                                                 self.max_age,
                                                 use_appearance_tracker=True)

        self.person_tracker = MultiObjectTracker(iou_cost,
                                                 color_cost,
                                                 self.max_iou_distance,
                                                 self.max_color_distance,
                                                 5,
                                                 self.max_disappeared,
                                                 self.max_age,
                                                 use_appearance_tracker=True)

        self.table_track = None
        self.table_depth = None
        self.table_shape = None

        self.shape_estimator = ShapeEstimator()

        self.object_pose_estimator = ObjectPoseEstimator()

        self.tracks_publisher = rospy.Publisher("tracks",
                                                SceneChangesStamped,
                                                queue_size=1)

        self.visualization_publisher = rospy.Publisher("tracks_image",
                                                       Image,
                                                       queue_size=1)

        self.use_depth = rospy.get_param("~use_depth", False)
        if self.use_depth is True:
            rospy.loginfo("[tabletop_perception] Subscribing to '/{}' topic...".format(self.rgb_image_topic))
            self.rgb_image_sub = message_filters.Subscriber(self.rgb_image_topic, Image)
            rospy.loginfo("[tabletop_perception] Subscribing to '/{}' topic...".format(self.depth_image_topic))
            self.depth_image_sub = message_filters.Subscriber(self.depth_image_topic, Image)

            self.sync = message_filters.ApproximateTimeSynchronizer([self.rgb_image_sub, self.depth_image_sub], 10, 0.2, allow_headerless=True)
            self.sync.registerCallback(self.observation_callback)
        else:
            rospy.loginfo("[tabletop_perception] Subscribing to '/{}' topic...".format(self.rgb_image_topic))
            self.rgb_image_sub = rospy.Subscriber(self.rgb_image_topic, Image, self.observation_callback, queue_size=1)

    def camera_info_callback(self, msg):
        if self.camera_info is None:
            rospy.loginfo("[tabletop_perception] Camera info received !")
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
            if self.publish_visualization_image is True:
                viz_frame = bgr_image

            image_height, image_width, _ = bgr_image.shape

            success, view_pose = self.get_pose_from_tf2(self.global_frame_id, self.camera_frame_id)

            if success is not True:
                rospy.logwarn("[tabletop_perception] The camera sensor is not localized in world space (frame '{}'), please check if the sensor frame is published in /tf".format(self.global_frame_id))
            else:
                view_matrix = view_pose.transform()

                self.frame_count %= self.n_frame
                ######################################################
                # Detection
                ######################################################

                detections = []

                if depth_image is not None:
                    if self.table_depth is None:
                        x = int(image_width*(1/2.0))
                        y = int(image_height*(3/4.0))
                        if not math.isnan(depth_image[y, x]):
                            self.table_depth = depth_image[y, x]/1000.0
                    table_depth = self.table_depth
                else:
                    table_depth = None
                table_detection = Detection(0, int(image_height/2.0), int(image_width), image_height, "table", 1.0, table_depth)

                if depth_image is not None:
                    if self.table_shape is None:
                        fx = self.camera_matrix[0][0]
                        fy = self.camera_matrix[1][1]
                        cx = self.camera_matrix[0][2]
                        cy = self.camera_matrix[1][2]
                        x = int(image_width*(1/2.0))
                        y = int(image_height*(3/4.0))
                        if not math.isnan(depth_image[y, x]):
                            z = depth_image[y, x]/1000.0
                            x = (x - cx) * z / fx
                            y = (y - cy) * z / fy
                            table_border = Vector6D(x=x, y=y, z=z)
                            x = int(image_width*(1/2.0))
                            y = int(image_height*(1/2.0))
                            if not math.isnan(depth_image[y, x]):
                                z = depth_image[y, x]/1000.0
                                x = (x - cx) * z / fx
                                y = (y - cy) * z / fy
                                table_center = Vector6D(x=x, y=y, z=z)
                                table_length = (image_width - cx) * z / fx
                                table_width = 2.0*math.sqrt(pow(table_border.pos.x-table_center.pos.x, 2)+pow(table_border.pos.y-table_center.pos.y, 2)+pow(table_border.pos.z-table_center.pos.z, 2))
                                center_in_world = view_pose+table_center
                                real_z = center_in_world.position().z
                                print view_pose.position()
                                print center_in_world.position()
                                self.table_shape = Box(table_width, table_length, real_z)
                                self.table_shape.pose.pos.z = -real_z/2.0
                                self.table_shape.color = self.shape_estimator.compute_dominant_color(rgb_image, table_detection.bbox)

                if self.frame_count == 0:
                    person_detections = self.detector.detect(rgb_image, depth_image=depth_image)
                    detections = person_detections
                elif self.frame_count == 1:
                    object_detections = self.foreground_detector.detect(rgb_image, depth_image=depth_image)
                    detections = object_detections
                else:
                    detections = []
                ####################################################################
                # Features estimation
                ####################################################################

                self.color_features_estimator.estimate(rgb_image, detections)

                ######################################################
                # Tracking
                ######################################################

                if self.frame_count == 0:
                    object_tracks = self.object_tracker.update(rgb_image, [], depth_image=depth_image)
                    person_tracks = self.person_tracker.update(rgb_image, person_detections, depth_image=depth_image)
                elif self.frame_count == 1:
                    object_tracks = self.object_tracker.update(rgb_image, object_detections, depth_image=depth_image)
                    person_tracks = self.person_tracker.update(rgb_image, [], depth_image=depth_image)
                else:
                    object_tracks = self.object_tracker.update(rgb_image, [], depth_image=depth_image)
                    person_tracks = self.person_tracker.update(rgb_image, [], depth_image=depth_image)

                if self.table_track is None:
                    self.table_track = Track(table_detection, 1, 4, 20)
                else:
                    self.table_track.update(table_detection)
                if self.table_shape is not None:
                    self.table_track.shapes = [self.table_shape]
                table_track = [self.table_track]

                tracks = object_tracks + person_tracks + table_track

                ########################################################
                # Pose & Shape estimation
                ########################################################

                self.object_pose_estimator.estimate(object_tracks + person_tracks + table_track, view_matrix, self.camera_matrix, self.dist_coeffs)

                self.shape_estimator.estimate(rgb_image, tracks, self.camera_matrix, self.dist_coeffs)

                self.frame_count += 1
                ######################################################
                # Visualization of debug image
                ######################################################

                perception_fps = cv2.getTickFrequency() / (cv2.getTickCount() - perception_timer)

                cv2.rectangle(viz_frame, (0, 0), (250, 40), (200, 200, 200), -1)
                perception_fps_str = "Perception fps : {:0.1f}hz".format(perception_fps)
                cv2.putText(viz_frame, "Nb detections/tracks : {}/{}".format(len(detections), len(tracks)), (5, 15),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.putText(viz_frame, perception_fps_str, (5, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                scene_changes = SceneChangesStamped()
                scene_changes.world = "tracks"

                header = bgr_image_msg.header
                header.frame_id = self.global_frame_id
                scene_changes.header = header

                for track in tracks:
                    if self.publish_visualization_image is True:
                        track.draw(viz_frame, (230, 0, 120, 125), 1, view_matrix, self.camera_matrix, self.dist_coeffs)
                    if track.is_confirmed():
                        scene_node = track.to_msg(header)
                        if self.publish_tf is True:
                            if track.is_located() is True:
                                self.publish_tf_pose(scene_node.pose_stamped, self.global_frame_id, scene_node.id)
                        scene_changes.changes.nodes.append(scene_node)
                if self.publish_visualization_image is True:
                    viz_img_msg = self.bridge.cv2_to_imgmsg(viz_frame)
                    self.visualization_publisher.publish(viz_img_msg)

                self.tracks_publisher.publish(scene_changes)

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
            pose = Vector6D(x=x, y=y, z=z).from_quaternion(rx, ry, rz, rw)
            return True, pose
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("[perception] Exception occured: {}".format(e))
            return False, None

    def publish_tf_pose(self, pose_stamped, source_frame, target_frame):
        transform = TransformStamped()
        transform.child_frame_id = target_frame
        transform.header.frame_id = source_frame
        transform.header.stamp = pose_stamped.header.stamp
        transform.transform.translation = pose_stamped.pose.pose.position
        transform.transform.rotation = pose_stamped.pose.pose.orientation
        self.tf_broadcaster.sendTransform(transform)
