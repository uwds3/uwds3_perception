import numpy as np
import cv2
from .bbox import BoundingBox


class Detection(object):
    """Represents a 2D detection associated with a label and a confidence"""

    def __init__(self, xmin, ymin, xmax, ymax, label, confidence, depth=None):
        """Detection constructor"""
        self.label = label
        self.confidence = confidence
        self.bbox = BoundingBox(xmin, ymin, xmax, ymax, depth=None)
        self.features = {}

    def draw(self, image, color, thickness):
        """Draws the detection"""
        self.bbox.draw(image, color, thickness)
        cv2.putText(image,
                    "{}".format(self.confidence),
                    (self.bbox.xmin+5, self.bbox.ymax+5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    thickness)
        cv2.putText(image,
                    self.label,
                    (self.bbox.xmin+35, self.bbox.ymax+5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    thickness)
