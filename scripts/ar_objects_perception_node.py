#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import rospy
from uwds3_perception.ar_objects_perception import ARObjectsPerception


class ARObjectsPerceptionNode(object):
    def __init__(self):
        rospy.loginfo("[ar_perception] Starting ARObjects perception...")
        self.perception = ARObjectsPerception()
        rospy.loginfo("[ar_perception] ARObjects perception ready !")

    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()


if __name__ == '__main__':
    rospy.init_node("uwds3_ar_objects")
    core = ARObjectsPerceptionNode().run()
