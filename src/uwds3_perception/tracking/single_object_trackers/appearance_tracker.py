import cv2
import numpy as np
import math
from pyuwds3.types.detection import Detection


class AppearanceTracker(object):
    """Single object tracker based on opencv appearance tracker"""
    def __init__(self, tracker_type="MEDIANFLOW"):
        """ """
        tracker_types = ["MOSSE", "MEDIANFLOW", "KCF", "MIL"]
        if tracker_type not in tracker_types:
            raise ValueError("Invalid tracker")
        self.tracker = None
        self.object_label = None
        self.tracker_type = tracker_type

    def update(self, rgb_image, detection, depth_image=None):
        """ """
        if self.tracker_type == "MOSSE":
            self.tracker = cv2.TrackerMOSSE_create()
        elif self.tracker_type == "MEDIANFLOW":
            self.tracker = cv2.TrackerMedianFlow_create()
        elif self.tracker_type == "KCF":
            self.tracker = cv2.TrackerKCF_create()
        elif self.tracker_type == "MIL":
            self.tracker = cv2.TrackerMIL_create()
        else:
            raise NotImplementedError("Tracker not available")
        self.object_label = detection.label
        xmin = detection.bbox.xmin
        ymin = detection.bbox.ymin
        w = detection.bbox.width()
        h = detection.bbox.height()
        bbox = (xmin, ymin, w, h)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        self.tracker.init(bgr_image, bbox)

    def predict(self, rgb_image, depth_image=None):
        """ """
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        success, bbox = self.tracker.update(bgr_image)
        xmin, ymin, w, h = bbox
        xmin = 0 if xmin < 0 else xmin
        ymin = 0 if ymin < 0 else ymin
        x = int(xmin + w/2.0)
        y = int(ymin + h/2.0)
        if depth_image is not None:
            x = depth_image.shape[1]-1 if x > depth_image.shape[1] else x
            y = depth_image.shape[0]-1 if y > depth_image.shape[0] else y
            depth = depth_image[int(y)][int(x)]/1000.0
            if math.isnan(depth) or depth == 0.0:
                depth = None
        else:
            depth = None
        prediction = Detection(xmin, ymin, xmin+w, ymin+h, self.object_label, 1.0, depth=depth)
        return success, prediction
