import cv2
import numpy as np
from pyuwds3.types.detection import Detection
from scipy.spatial.distance import cosine


class SingleObjectTracker(object):
    def __init__(self, max_hsv_distance=.4):
        self.mediaflow_tracker = cv2.TrackerMedianFlow_create()
        self.object_label = None
        self.hsv_hist = None
        self.max_hsv_distance = max_hsv_distance

    def update(self, rgb_image, object):
        self.mediaflow_tracker = cv2.TrackerMedianFlow_create()
        self.object_label = object.label
        xmin = object.bbox.xmin
        ymin = object.bbox.ymin
        w = object.bbox.width()
        h = object.bbox.height()
        self.hsv_hist = self.__compute_histogram(rgb_image, object)
        bbox = (xmin, ymin, w, h)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        self.mediaflow_tracker.init(bgr_image, bbox)

    def predict(self, rgb_image):
        success, bbox = self.mediaflow_tracker.update(rgb_image)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        xmin, ymin, width, height = bbox
        xmin = 0 if xmin < 0 else xmin
        ymin = 0 if ymin < 0 else ymin
        if width < 10 or height < 10:
            return False, None
        det = Detection(xmin, ymin, xmin+width, ymin+height, self.object_label, 1.0)
        hsv_hist = self.__compute_histogram(bgr_image, det)
        distance = 1 - cosine(self.hsv_hist, hsv_hist)
        if distance > self.max_hsv_distance:
            return False, None
        else:
            det.confidence = 1 - distance
            return True, det

    def __compute_histogram(self, rgb_image, object):
        xmin = object.bbox.xmin
        ymin = object.bbox.ymin
        w = object.bbox.width()
        h = object.bbox.height()
        assert h > 10 and w > 10
        cropped_image = rgb_image[ymin:ymin+h, xmin:xmin+w]
        cropped_hsv = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(cropped_hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        hsv_hist = cv2.calcHist([cropped_hsv], [0], mask, [180], [0, 180])
        cv2.normalize(hsv_hist, hsv_hist, 0, 255, cv2.NORM_MINMAX)
        return hsv_hist
