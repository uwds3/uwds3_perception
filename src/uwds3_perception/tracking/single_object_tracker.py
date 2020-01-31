import cv2
import dlib
from scipy.spatial.distance import cosine, euclidean
from pyuwds3.types.detection import Detection


class SingleObjectTracker(object):
    def __init__(self, track, similarity_metric, max_dist, features_extractor):
        self.tracker = dlib.correlation_tracker()
        self.features_extractor = features_extractor
        self.max_dist = max_dist
        self.similarity_metric = similarity_metric
        self.track = track
        self.started = False

    def update(self, rgb_image, track):
        self.track = track
        rect = dlib.rectangle(int(track.bbox.xmin),
                              int(track.bbox.ymin),
                              int(track.bbox.xmax),
                              int(track.bbox.ymax))
        if self.started is False:
            self.tracker.start_track(rgb_image, rect)
            self.started = True
        else:
            self.tracker.update(rgb_image, rect)

    def predict(self, rgb_image):
        try:
            self.tracker.update(rgb_image)
            pos = self.tracker.get_position()
            xmin = pos.left()
            ymin = pos.top()
            xmax = pos.right()
            ymax = pos.bottom()
            det = Detection(xmin, ymin, xmax, ymax, 1.0, self.track.label)
            self.features_extractor.extract(rgb_image, [det])
            dist = self.similarity_metric(det, self.track)
            if dist > self.max_dist:
                return False, None
            else:
                det.confidence = dist
                return True, det
        except Exception:
            return False, None
