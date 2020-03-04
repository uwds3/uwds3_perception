import cv2
import numpy as np
import math
from pyuwds3.types.detection import Detection


class MedianflowTracker(object):
    """Single object tracker based on opencv MEDIANFLOW tracker"""
    def __init__(self):
        """ """
        self.medianflow_tracker = None
        self.object_label = None

    def update(self, rgb_image, detection, depth_image=None):
        """ """
        #if self.medianflow_tracker is None:
        self.medianflow_tracker = cv2.TrackerMedianFlow_create()
        self.object_label = detection.label
        xmin = detection.bbox.xmin
        ymin = detection.bbox.ymin
        w = detection.bbox.width()
        h = detection.bbox.height()
        bbox = (xmin, ymin, w, h)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        self.medianflow_tracker.init(bgr_image, bbox)

    def predict(self, rgb_image, depth_image=None):
        """ """
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        success, bbox = self.medianflow_tracker.update(bgr_image)
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
        prediction = Detection(xmin, ymin, xmin+w, ymin+h, self.object_label, 1.0, depth=None)
        return success, prediction
