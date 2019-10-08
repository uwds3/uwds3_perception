import dlib
import uuid
import multiprocessing
from .kalman_stabilizer import Stabilizer


class TrackState:
    TENTATIVE = 1
    CONFIRMED = 2
    OCCLUDED = 3
    DELETED = 4


def run_tracker(rgb_image, bbox, input_queue, output_queue):
    tracker = dlib.correlation_tracker()
    tracker.start_track(rgb_image, bbox)
    while True:
        rgb, bbox = input_queue.get()
        if rgb is not None:
            if bbox is not None:
                tracker.update(rgb, bbox)
            else:
                tracker.update(rgb)
            bbox = tracker.get_position()
            output_queue.put(bbox)


class Track(object):
    def __init__(self,
                 detection,
                 n_init,
                 max_disappeared,
                 max_age):

        self.uuid = str(uuid.uuid4())

        self.bbox = dlib.rectangle(int(detection.bbox.left()),
                                   int(detection.bbox.top()),
                                   int(detection.bbox.right()),
                                   int(detection.bbox.bottom()))

        self.class_label = detection.class_label

        self.state = TrackState.TENTATIVE

        self.n_init = n_init
        self.max_disappeared = max_disappeared
        self.max_age = max_age

        self.translation = None
        self.rotation = None

        self.tracker = None

        self.first_kalman_update = True

        self.stabilizer = Stabilizer()

        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()

        self.age = 1
        self.hits = 1

        self.properties = {}

    def start_tracker(self, bbox, rgb_image):

        process = multiprocessing.Process(target=run_tracker,
                                          args=(rgb_image,
                                                bbox,
                                                self.input_queue,
                                                self.output_queue))
        process.daemon = True
        process.start()
        self.tracker = process

    def stop_tracker(self):
        del self.tracker

    def update(self, bbox, rgb_image=None):
        if rgb_image is not None:
            self.input_queue.put((rgb_image, bbox))
            bbox = self.output_queue.get()
        self.age = 0
        self.bbox = bbox
        self.hits += 1
        if self.state == TrackState.TENTATIVE and self.hits >= self.n_init:
            self.state = TrackState.CONFIRMED
        if self.state == TrackState.OCCLUDED:
            self.state = TrackState.CONFIRMED

    def filter(self, rotation, translation):
        pass

    def predict(self, rgb_image=None):
        self.age += 1
        if rgb_image is not None:
            self.input_queue.put((rgb_image, None))
            self.bbox = self.output_queue.get()
        else:
            pass # TODO perform kalman prediction
        if self.age > self.max_age:
            self.state = TrackState.DELETED

    def mark_missed(self):
        self.age += 1
        if self.state == TrackState.TENTATIVE:
            if self.age > self.max_disappeared:
                self.state = TrackState.DELETED
        if self.state == TrackState.CONFIRMED:
            if self.age > self.max_disappeared:
                self.state = TrackState.OCCLUDED
        elif self.state == TrackState.OCCLUDED:
            if self.age > self.max_age:
                self.state = TrackState.DELETED

    def is_perceived(self):
        if not self.is_deleted():
            return self.state != TrackState.OCCLUDED
        else:
            return False

    def is_confirmed(self):
        return self.state == TrackState.CONFIRMED

    def is_occluded(self):
        return self.state == TrackState.OCCLUDED

    def is_deleted(self):
        return self.state == TrackState.DELETED

    def __str__(self):
        return "rect : {} for class : {}".format(self.bbox, self.class_label)
