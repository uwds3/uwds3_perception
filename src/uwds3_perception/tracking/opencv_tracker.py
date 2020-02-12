import cv2
from pyuwds3.types.detection import Detection


class OpenCVTracker(object):
    def __init__(self, tracker_type="MEDIANFLOW"):
        self.object_label = None
        self.tracker_type = tracker_type

    def init_tracking(self, rgb_image, object):
        self.object_label = object.label
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        xmin = object.bbox.xmin
        ymin = object.bbox.ymin
        width = object.bbox.width()
        height = object.bbox.height()
        if self.tracker_type == 'BOOSTING':
            self.tracker = cv2.TrackerBoosting_create()
        if self.tracker_type == 'MIL':
            self.tracker = cv2.TrackerMIL_create()
        if self.tracker_type == 'KCF':
            self.tracker = cv2.TrackerKCF_create()
        if self.tracker_type == 'TLD':
            self.tracker = cv2.TrackerTLD_create()
        if self.tracker_type == 'MEDIANFLOW':
            self.tracker = cv2.TrackerMedianFlow_create()
        if self.tracker_type == 'GOTURN':
            self.tracker = cv2.TrackerGOTURN_create()
        if self.tracker_type == 'MOSSE':
            self.tracker = cv2.TrackerMOSSE_create()
        if self.tracker_type == "CSRT":
            self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(bgr_image, (xmin, ymin, width, height))

    def update(self, rgb_image, object):
        self.init_tracking(rgb_image, object)

    def predict(self, rgb_image):
        success, bbox = self.tracker.update(rgb_image)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        self.tracker.init(bgr_image, bbox)
        xmin, ymin, width, height = bbox
        det = Detection(xmin, ymin, xmin+width, ymin+height, 1.0, self.object_label)
        return success, det
