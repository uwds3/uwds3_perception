import rospy
import yaml
import numpy as np
import uwds3_msgs.msg
import ar_track_alvar_msgs.msg
import sensor_msgs.msg
import tf2_ros
from tf.transformations import quaternion_from_euler
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from pyuwds3.types.vector.vector6d import Vector6D


class ARObjectsPerception(object):
    def __init__(self):
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)
        self.tf_broadcaster = TransformBroadcaster()

        self.camera_info_topic = rospy.get_param("~camera_info_topic", "")
        self.global_frame_id = rospy.get_param("~global_frame_id", "odom")

        self.resource_directory = rospy.get_param("~resource_directory", "")
        if self.resource_directory == "":
            raise ValueError("Need to specify the '~resource_directory' parameter")

        self.ar_tags_config = rospy.get_param("~ar_tags_config", "")
        if self.ar_tags_config == "":
            raise ValueError("Need to specify the '~ar_tags_config' parameter")

        self.load_config(self.ar_tags_config)

        self.tracks = {}

        self.camera_info = None

        for entity_id, entity in self.config.items():
            track = uwds3_msgs.msg.SceneNode()
            track.label = self.config[entity_id]["label"]
            track.description = self.config[entity_id]["description"]
            track.is_located = True
            track.has_shape = True
            shape = uwds3_msgs.msg.PrimitiveShape()
            shape.type = uwds3_msgs.msg.PrimitiveShape.MESH
            position = self.config[entity_id]["position"]
            orientation = self.config[entity_id]["orientation"]
            q = quaternion_from_euler(orientation["x"], orientation["y"], orientation["z"], "rxyz")
            shape.pose.position.x = position["x"]
            shape.pose.position.y = position["y"]
            shape.pose.position.z = position["z"]
            shape.pose.orientation.x = q[0]
            shape.pose.orientation.y = q[1]
            shape.pose.orientation.z = q[2]
            shape.pose.orientation.w = q[3]
            shape.mesh_resource = "file://"+self.resource_directory+self.config[entity_id]["mesh_resource"]
            track.shapes = [shape]
            self.tracks[entity_id] = track

        self.camera_info_subscriber = rospy.Subscriber(self.camera_info_topic,
                                                       sensor_msgs.msg.CameraInfo,
                                                       self.camera_info_callback)

        self.ar_track_subscriber = rospy.Subscriber("ar_pose_marker",
                                                    ar_track_alvar_msgs.msg.AlvarMarkers,
                                                    self.observation_callback)
        self.tracks_publisher = rospy.Publisher("tracks",
                                                uwds3_msgs.msg.SceneChangesStamped,
                                                queue_size=1)

    def load_config(self, config_file_path):
        with open(config_file_path, "r")as file:
            self.config = yaml.load(file)

    def camera_info_callback(self, msg):
        if self.camera_info is None:
            rospy.loginfo("[ar_perception] Camera info received !")
        self.camera_info = msg
        self.camera_frame_id = msg.header.frame_id
        self.camera_matrix = np.array(msg.K).reshape((3, 3))
        self.dist_coeffs = np.array(msg.D)

    def observation_callback(self, ar_track_msg):
        scene_changes = uwds3_msgs.msg.SceneChangesStamped()
        scene_changes.world = "tracks"
        scene_changes.header = ar_track_msg.header
        scene_changes.header.stamp = rospy.Time().now()
        scene_changes.header.frame_id = self.global_frame_id
        if self.camera_info is not None:
            if len(ar_track_msg.markers) > 0:
                scene_changes.header.stamp = ar_track_msg.header.stamp
            for object in ar_track_msg.markers:
                success, view_pose = self.get_pose_from_tf2(self.global_frame_id, object.header.frame_id)#, ar_track_msg.header.stamp)
                if success is True:
                    position = object.pose.pose.position
                    orientation = object.pose.pose.orientation
                    obj_sensor_pose = Vector6D(x=position.x,
                                               y=position.y,
                                               z=position.z).from_quaternion(orientation.x,
                                                                             orientation.y,
                                                                             orientation.z,
                                                                             orientation.w)
                    world_pose = view_pose + obj_sensor_pose
                    self.tracks[object.id].pose_stamped.header = scene_changes.header
                    self.tracks[object.id].pose_stamped.pose.pose.position.x = world_pose.position().x
                    self.tracks[object.id].pose_stamped.pose.pose.position.y = world_pose.position().y
                    self.tracks[object.id].pose_stamped.pose.pose.position.z = world_pose.position().z
                    q = world_pose.quaternion()
                    self.tracks[object.id].pose_stamped.pose.pose.orientation.x = q[0]
                    self.tracks[object.id].pose_stamped.pose.pose.orientation.y = q[1]
                    self.tracks[object.id].pose_stamped.pose.pose.orientation.z = q[2]
                    self.tracks[object.id].pose_stamped.pose.pose.orientation.w = q[3]
                    scene_changes.changes.nodes.append(self.tracks[object.id])
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
