import cv2
import dlib
from scipy.spatial.distance import cosine, euclidean
from pyuwds3.types.detection import Detection


class DlibTracker(object):
    def __init__(self, max_age):
        self.tracker = dlib.correlation_tracker()
        self.track = None
        self.max_age = max_age
        self.age = 0
        self.started = False

    def update(self, rgb_image, track):
        self.track = track
        self.age = 0
        rect = dlib.rectangle(int(track.bbox.xmin),
                              int(track.bbox.ymin),
                              int(track.bbox.xmax),
                              int(track.bbox.ymax))
        self.tracker = dlib.correlation_tracker()
        self.tracker.start_track(rgb_image, rect)
        # if self.started is False:
        #     self.tracker.start_track(rgb_image, rect)
        #     self.started = True
        # else:
        #     self.tracker.update(rgb_image, rect)

    def predict(self, rgb_image):
        try:
            self.tracker.update(rgb_image)
            pos = self.tracker.get_position()
            self.tracker.update(rgb_image, pos)
            xmin = pos.left()
            ymin = pos.top()
            xmax = pos.right()
            ymax = pos.bottom()
            det = Detection(xmin, ymin, xmax, ymax, 1.0, self.track.label)
            self.age += 1
            if self.age > self.max_age:
                return False, None
            return True, det
        except Exception as e:
            print("{}".format(e))
            return False, None
