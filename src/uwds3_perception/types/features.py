import uwds3_msgs
import numpy as np


class Features(object):
    """Represent a features vector with a confidence"""

    def __init__(self, name, data, confidence):
        """Features constructor"""
        self.name = name
        self.data = data
        self.confidence = confidence

    def to_msg(self):
        """Converts into ROS message"""
        return uwds3_msgs.msg.Features(name=self.name,
                                       data=list(np.array(self.data).flatten()),
                                       confidence=self.confidence)
