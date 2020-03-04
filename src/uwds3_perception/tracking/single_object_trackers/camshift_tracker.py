import numpy as np
import cv2
import math
from pyuwds3.types.detection import Detection
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class CamshiftTracker(object):
    def __init__(self):
        """ See: Computer Vision Face Tracking for Use in a Perceptual User Interface (1998)"""
        self.object_label = None
        self.histogram = None
        self.bbox = None
        self.bridge = CvBridge()
        self.pub = rospy.Publisher("test", Image, queue_size=1)

    def update(self, rgb_image, detection, depth_image=None):
        """ """
        self.bbox = detection.bbox
        self.object_label = detection.label
        xmin = detection.bbox.xmin
        ymin = detection.bbox.ymin
        w = detection.bbox.width()
        h = detection.bbox.height()

        hsv_roi = cv2.cvtColor(rgb_image[ymin:ymin+h, xmin:xmin+w].copy(), cv2.COLOR_RGB2HSV)
        if self.histogram is None:
            self.histogram = cv2.calcHist([hsv_roi], [0], detection.mask, [180], [0,180])
        else:
            histogram = cv2.calcHist([hsv_roi], [0], detection.mask, [180], [0,180])
            self.histogram += histogram
        cv2.normalize(self.histogram, self.histogram, 0, 255, cv2.NORM_MINMAX)

    def predict(self, rgb_image, depth_image=None):
        """ """
        xmin = self.bbox.xmin
        ymin = self.bbox.ymin
        xmax = self.bbox.xmax
        ymax = self.bbox.ymax
        w = self.bbox.width()
        h = self.bbox.height()

        new_xmin = int(xmin - 2.0*w) if int(xmin - 2.0*w) > 0 else 0
        new_ymin = int(ymin - 2.0*h) if int(xmin - 2.0*h) > 0 else 0
        new_xmax = int(xmax + 2.0*w) if int(xmax + 2.0*w) < rgb_image.shape[1] else rgb_image.shape[1]
        new_ymax = int(ymax + 2.0*h) if int(ymax + 2.0*h) < rgb_image.shape[0] else rgb_image.shape[0]

        hsv_image = cv2.cvtColor(rgb_image[new_ymin:new_ymax, new_xmin:new_xmax], cv2.COLOR_RGB2HSV)
        dst = cv2.calcBackProject([hsv_image], [0], self.histogram, [0, 180], 1)

        track_window = xmin - new_xmin, ymin - new_ymin, w, h
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1)
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        points = cv2.boxPoints(ret)
        points = np.int0(points)

        xmin, ymin, w, h = cv2.boundingRect(points)
        if h < 7 or w < 7:
            return False, None
        if xmin < 0 or ymin < 0:
            return False, None
        _, thresh_map = cv2.threshold(dst[ymin:ymin+h, xmin:xmin+w], 20, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
        closing = cv2.morphologyEx(thresh_map, cv2.MORPH_CLOSE, kernel_big)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_small)

        mask = opening

        if depth_image is not None:
            x = depth_image.shape[1]-1 if x > depth_image.shape[1] else x
            y = depth_image.shape[0]-1 if y > depth_image.shape[0] else y
            depth = depth_image[int(y)][int(x)]/1000.0
            if math.isnan(depth) or depth == 0.0:
                depth = None
        else:
            depth = None

        pred = Detection(int(xmin), int(ymin), int(xmin+w), int(ymin+h), self.object_label, 0.6, depth=depth, mask=mask)
        #
        # cv2.normalize(dst, dst, 0, 255, cv2.NORM_MINMAX)
        # _, thresh_map = cv2.threshold(dst[ymin:ymin+h, xmin:xmin+w], 20, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
        # closing = cv2.morphologyEx(thresh_map, cv2.MORPH_CLOSE, kernel_big)
        # opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_small)
        # refined_xmin, refined_ymin, w, h = cv2.boundingRect(opening)
        #
        # xmin += new_xmin + refined_xmin
        # ymin += new_ymin + refined_ymin
        # if w < 30 or h < 30:
        #     return False, None
        # else:
        #     x = int(xmin + w/2.0)
        #     y = int(ymin + h/2.0)
        #     if depth_image is not None:
        #         x = depth_image.shape[1]-1 if x > depth_image.shape[1] else x
        #         y = depth_image.shape[0]-1 if y > depth_image.shape[0] else y
        #         depth = depth_image[int(y)][int(x)]/1000.0
        #         if math.isnan(depth) or depth == 0.0:
        #             depth = None
        #     else:
        #         depth = None
        #     if w < 30 or h < 30:
        #         return False, None
        #     mask = opening[int(ymin):int(ymin+h), int(xmin):int(xmin+w)]
        #     pred = Detection(int(xmin), int(ymin), int(xmin+w), int(ymin+h), self.object_label, 0.6, depth=depth, mask=mask)
        #     print pred.bbox
        return True, pred
