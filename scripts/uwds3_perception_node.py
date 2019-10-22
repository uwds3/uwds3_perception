#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import rospy
from uwds3_perception.uwds3_perception import Uwds3Perception


class Uwds3PerceptionNode(object):
    def __init__(self):
        rospy.loginfo("[perception] Starting Underworlds perception...")
        self.underworlds_core = Uwds3Perception()
        rospy.loginfo("[perception] Underworlds perception ready !")

    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()

if __name__ == '__main__':
    rospy.init_node("uwds3_perception")
    core = Uwds3PerceptionNode().run()
