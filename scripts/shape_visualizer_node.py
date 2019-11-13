#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import rospy
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from uwds3_msgs.msg import SceneNode, PrimitiveShape, SceneNodeArrayStamped
from geometry_msgs.msg import Vector3
from std_msgs.msg import ColorRGBA

class ShapeVisualizerNode(object):
    def __init__(self):
        self.tracks_topic = rospy.get_param("~tracks_topic", "tracks")
        self.marker_id_map = {}
        self.tracks = {}
        self.track_marker = {}
        self.track_color = {}
        self.last_marker_id = 0
        self.markers_publisher = rospy.Publisher(self.tracks_topic+"_viz", MarkerArray, queue_size=1)
        self.track_subscriber = rospy.Subscriber(self.tracks_topic, SceneNodeArrayStamped, self.observation_callback, queue_size=1)
        rospy.loginfo("[visualizer] Shape visualizer ready !")

    def observation_callback(self, tracks_msg):
        markers_msg = MarkerArray()
        perceived_tracks = []
        for track in tracks_msg.nodes:
            if track.id in perceived_tracks:
                rospy.logwarn("[visualizer] Error occured: Shape for node <{}> already created".format(track.id))
                continue
            perceived_tracks.append(track.id)
            if track.has_shape is True:
                if track.id in self.track_marker:
                    marker = self.track_marker[track.id]
                    marker.action = Marker.MODIFY
                else:
                    marker = Marker()
                    self.last_marker_id += 1
                    marker.id = self.last_marker_id
                    marker.action = Marker.ADD
                    frame = track.id
                    marker.header.frame_id = frame
                    self.track_marker[track.id] = marker
                marker.header.stamp = tracks_msg.header.stamp

                if track.shape.type == PrimitiveShape.CYLINDER:
                    marker.type = Marker.CYLINDER
                elif track.shape.type == PrimitiveShape.BOX:
                    marker.type = Marker.CUBE
                if track.label == "face":
                    marker.scale = Vector3(x=track.shape.dimensions[0],
                                           y=track.shape.dimensions[1],
                                           z=track.shape.dimensions[0])
                else:
                    marker.scale = Vector3(x=track.shape.dimensions[0],
                                           y=track.shape.dimensions[1],
                                           z=track.shape.dimensions[2])
                if track.id in self.track_color:
                    color = self.track_color[track.id]
                else:
                    color = ColorRGBA()
                    color.a = 0.65
                    color.r = np.random.random_sample()
                    color.g = np.random.random_sample()
                    color.b = np.random.random_sample()
                    self.track_color[track.id] = color
                marker.color = color
                marker.pose = track.shape.pose
                self.track_marker[track.id] = marker
                markers_msg.markers.append(marker)
            if track.has_camera is True:
                pass
        for track_id in self.track_marker.keys():
            if track_id not in perceived_tracks:
                self.track_marker[track_id].action = Marker.DELETE
                markers_msg.markers.append(self.track_marker[track_id])
                del self.track_marker[track_id]
        self.markers_publisher.publish(markers_msg)

    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()

if __name__ == '__main__':
    rospy.init_node("shape_visualizer", anonymous=False)
    ros_node = ShapeVisualizerNode().run()
