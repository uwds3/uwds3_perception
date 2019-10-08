import cv2
import rospy
import numpy as np
import geometry_msgs
import sensor_msgs
import message_filters
import uwds3_msgs
from cv_bridge import CvBridge
from tf import transformations
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from .detection.opencv_dnn_detector import OpenCVDNNDetector
from .detection.hog_face_detector import HOGFaceDetector
from .estimation.facial_landmarks_estimator import FacialLandmarksEstimator, NOSE, POINT_OF_SIGHT
from .estimation.head_pose_estimator import HeadPoseEstimator
from .tracking.human_tracker import HumanTracker
from .estimation.depth_estimator import DepthEstimator
from .tracking.linear_assignment import iou_distance


class Uwds3Perception(object):
    def __init__(self):

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)
        self.tf_broadcaster = TransformBroadcaster()

        self.rgb_image_topic = rospy.get_param("~rgb_image_topic", "/camera/rgb/image_raw")
        self.depth_image_topic = rospy.get_param("~depth_image_topic", "/camera/depth/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/rgb/camera_info")

        self.base_frame_id = rospy.get_param("~base_frame_id", "base_link")
        self.global_frame_id = rospy.get_param("~global_frame_id", "map")

        self.bridge = CvBridge()

        rospy.loginfo("Subscribing to /{} topic...".format(self.camera_info_topic))
        self.camera_info = None
        self.camera_frame_id = None
        self.camera_info_subscriber = rospy.Subscriber(self.camera_info_topic, sensor_msgs.msg.CameraInfo, self.camera_info_callback)

        self.detector_model_filename = rospy.get_param("~detector_model_filename", "")
        self.detector_weights_filename = rospy.get_param("~detector_weights_filename", "")
        self.detector_config_filename = rospy.get_param("~detector_config_filename", "")

        self.body_parts = ["person", "face", "right_hand", "left_hand"]

        self.detector = OpenCVDNNDetector(self.detector_model_filename,
                                          self.detector_weights_filename,
                                          self.detector_config_filename,
                                          300)

        self.face_detector = HOGFaceDetector()

        self.n_frame = rospy.get_param("~n_frame", 2)
        self.frame_count = 0

        self.only_human = rospy.get_param("~only_human", True)

        self.use_depth = rospy.get_param("~use_depth", False)

        self.shape_predictor_config_filename = rospy.get_param("~shape_predictor_config_filename", "")

        self.human_tracker = HumanTracker(iou_distance, min_distance=0.7, max_disappeared=8, max_age=10)

        #self.tracks_publisher = rospy.Publisher("uwds3_perception/human_tracks", uwds3_msgs.msg.EntityArray, queue_size=1)

        self.visualization_publisher = rospy.Publisher("uwds3_perception/visualization", sensor_msgs.msg.Image, queue_size=1)

        if self.use_depth is True:
            self.rgb_image_sub = message_filters.Subscriber(self.rgb_image_topic, sensor_msgs.msg.Image)
            self.depth_image_sub = message_filters.Subscriber(self.depth_image_topic, sensor_msgs.msg.Image)

            self.sync = message_filters.TimeSynchronizer([self.rgb_image_sub, self.depth_image_sub], 10)
            self.depth_estimator = DepthEstimator()
            self.sync.registerCallback(self.observation_callback)

        else:
            self.rgb_image_sub = rospy.Subscriber(self.rgb_image_topic, sensor_msgs.msg.Image, self.observation_callback, queue_size=1)


    def camera_info_callback(self, msg):
        if self.camera_info is None:
            rospy.loginfo("Camera info received !")
        self.camera_info = msg
        self.camera_frame_id = msg.header.frame_id
        self.camera_matrix = np.array(msg.K).reshape((3, 3))
        self.dist_coeffs = np.array(msg.D)

    def observation_callback(self, rgb_image_msg, depth_image_msg=None):
        if self.camera_info is not None:
            bgr_image = self.bridge.imgmsg_to_cv2(rgb_image_msg)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            viz_frame = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

            detection_timer = cv2.getTickCount()
            detections = []
            if self.frame_count % self.n_frame == 0:
                detections = self.face_detector.detect(rgb_image)
            if self.frame_count % self.n_frame == 1:
                detections = self.detector.detect(rgb_image)
            self.frame_count += 1
            detection_fps = cv2.getTickFrequency() / (cv2.getTickCount() - detection_timer)

            human_detections = [d for d in detections if d.class_label in self.body_parts]
            object_detections = [d for d in detections if d.class_label not in self.body_parts]

            tracking_timer = cv2.getTickCount()
            if self.only_human is False:
                object_tracks = self.object_tracker.update(rgb_image, object_detections)
            human_tracks = self.human_tracker.update(rgb_image, human_detections, self.camera_matrix, self.dist_coeffs)
            tracking_fps = cv2.getTickFrequency() / (cv2.getTickCount() - tracking_timer)

            detection_fps_str = "Detection fps : %0.4fhz" % detection_fps
            tracking_fps_str = "Tracking and pose estimation fps : %0.4fhz" % tracking_fps

            cv2.putText(viz_frame, detection_fps_str, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(viz_frame, tracking_fps_str, (5, 45),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if self.only_human is False:
                tracks = human_tracks + object_tracks
            else:
                tracks = human_tracks

            for track in tracks:
                draw_track(viz_frame, track, self.camera_matrix, self.dist_coeffs)
                if track.rotation is not None and track.translation is not None:
                    transform = geometry_msgs.msg.TransformStamped()
                    transform.header.stamp = rospy.Time.now()
                    transform.header.frame_id = self.camera_frame_id
                    transform.child_frame_id = track.class_label+"_"+track.uuid[:6]
                    transform.transform.translation.x = track.translation[0]
                    transform.transform.translation.y = track.translation[1]
                    transform.transform.translation.z = track.translation[2]
                    q_rot = transformations.quaternion_from_euler(track.rotation[0], track.rotation[1], track.rotation[2], "rxyz")
                    transform.transform.rotation.x = q_rot[0]
                    transform.transform.rotation.y = q_rot[1]
                    transform.transform.rotation.z = q_rot[2]
                    transform.transform.rotation.w = q_rot[3]
                    self.tf_broadcaster.sendTransform(transform)

            viz_img_msg = self.bridge.cv2_to_imgmsg(viz_frame)
            self.visualization_publisher.publish(viz_img_msg)


def draw_track(opencv_image, track, camera_matrix, dist_coeffs):
    if track.is_confirmed():
        tl_corner = (int(track.bbox.left()), int(track.bbox.top()))
        br_corner = (int(track.bbox.right()), int(track.bbox.bottom()))
        rot = track.rotation
        trans = track.translation
        if rot is not None and trans is not None:
            cv2.drawFrameAxes(opencv_image, camera_matrix, dist_coeffs, np.array(rot).reshape((3,1)), np.array(trans).reshape(3,1), 0.03)
            if track.class_label == "face":
                if "facial_landmarks" in track.properties:
                    for (x, y) in track.properties["facial_landmarks"]:
                        cv2.circle(opencv_image, (x, y), 1, (0, 255, 0), -1)
        cv2.putText(opencv_image, track.uuid[:6], (tl_corner[0]+5, tl_corner[1]+25),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 0, 0), 2)
        cv2.putText(opencv_image, track.class_label, (tl_corner[0]+5, tl_corner[1]+45),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 0, 0), 2)
        cv2.rectangle(opencv_image, tl_corner, br_corner, (255, 255, 0), 2)


def get_last_transform_from_tf2(self, source_frame, target_frame):
        try:
            trans = self.tf_buffer.lookup_transform(source_frame, target_frame, rospy.Time(0))
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            z = trans.transform.translation.z

            rx = trans.transform.rotation.x
            ry = trans.transform.rotation.y
            rz = trans.transform.rotation.z
            rw = trans.transform.rotation.w

            return True, [x, y, z], [rx, ry, rz, rw]
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return False, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]
