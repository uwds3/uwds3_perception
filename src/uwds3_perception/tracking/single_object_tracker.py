import cv2
import dlib
from pyuwds3.types.bbox import BoundingBox


class SingleObjectTracker(object):
    def __init__(self, tracker_type):
        self.tracker_types = ["Dlib", "BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "MOSSE", "GOTURN", "CSRT", "BOOSTING"]
        if tracker_type not in self.tracker_types:
            raise ValueError("Invalid tracker type")

        if tracker_type == 'BOOSTING':
            self.tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            self.tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            self.tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            self.tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            self.tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            self.tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            self.tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            self.tracker = cv2.TrackerCSRT_create()
        if tracker_type == "Dlib":
            self.tracker = dlib.correlation_tracker()

        self.tracker_type = tracker_type

        self.tracker_started = False

    def update(self, rgb_image, bbox):
        if self.tracker_type == "Dlib":
            rect = dlib.rectangle(int(bbox.xmin),
                                  int(bbox.ymin),
                                  int(bbox.xmin),
                                  int(bbox.ymax))
            if self.tracker_started is False:
                self.tracker.start_track(rgb_image, rect)
            else:
                self.tracker.update(rgb_image, rect)
                self.tracker_started = True
        else:
            w = bbox.width()
            h = bbox.height()
            x = bbox.center().x
            y = bbox.center().y
            self.tracker.init(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB),
                                          (x, y, w, h))

    def predict(self, rgb_image):
        try:
            if self.tracker_type == "Dlib":
                self.tracker.update(rgb_image)
                pos = self.tracker.get_position()
                xmin = pos.left()
                ymin = pos.top()
                xmax = pos.right()
                ymax = pos.bottom()
                return True, BoundingBox(xmin, ymin, xmax, ymax)
            else:
                ok, bbox = self.tracker.update(cv2.cvtColor(rgb_image,
                                                            cv2.COLOR_BGR2RGB))
                x, y, w, h = bbox
                xmin = x - w/2.0
                ymin = y - h/2.0
                xmax = x + w/2.0
                ymax = y + h/2.0
                return True, BoundingBox(xmin, ymin, xmax, ymax)
        except Exception:
            return False, None
