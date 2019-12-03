import dlib
import numpy as np
import uuid
from .kalman_stabilizer import Stabilizer

class TrackState:
    TENTATIVE = 1
    CONFIRMED = 2
    OCCLUDED = 3
    DELETED = 4


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

        self.translation_cov = 0.98

        self.rotation_cov = 0.1

        r_stabilizers = [Stabilizer(
                        state_num=2,
                        measure_num=1,
                        cov_process=0.1,
                        cov_measure=self.rotation_cov) for _ in range(3)]

        t_stabilizers = [Stabilizer(
                        state_num=2,
                        measure_num=1,
                        cov_process=0.1,
                        cov_measure=self.translation_cov) for _ in range(3)]

        self.stabilizers = r_stabilizers + t_stabilizers

        self.age = 1
        self.hits = 1

        self.properties = {}

    def start_tracker(self, bbox, rgb_image):
        self.tracker = dlib.correlation_tracker()
        self.tracker.start_track(rgb_image, bbox)

    def stop_tracker(self):
        if self.tracker is not None:
            self.tracker = None

    def update(self, bbox, rgb_image=None):
        if rgb_image is not None and self.tracker is not None:
            self.tracker.update(rgb_image, bbox)
            bbox = self.tracker.get_position()
        self.age = 0
        self.bbox = bbox
        self.hits += 1
        if self.state == TrackState.TENTATIVE and self.hits >= self.n_init:
            self.state = TrackState.CONFIRMED
        if self.state == TrackState.OCCLUDED:
            self.state = TrackState.CONFIRMED

    def filter(self, rotation, translation):
        stable_pose = []
        if self.first_kalman_update is True:
            self.first_kalman_update = False
            pose_np = np.array((rotation, translation)).flatten()
            for value, ps_stb in zip(pose_np, self.stabilizers):
                ps_stb.state[0] = value
                ps_stb.state[1] = 0
                stable_pose.append(value)
        else:
            stable_pose = []
            pose_np = np.array((rotation, translation)).flatten()
            for value, ps_stb in zip(pose_np, self.stabilizers):
                ps_stb.update([value])
                stable_pose.append(ps_stb.state[0])

        stable_pose = np.reshape(stable_pose, (-1, 3))
        self.rotation = stable_pose[0]
        self.translation = stable_pose[1]

    def predict(self, rgb_image=None):
        self.age += 1
        if rgb_image is not None and self.tracker is not None:
            self.tracker.update(rgb_image)
            self.bbox = self.tracker.get_position()
        else:
            pass # TODO perform kalman prediction
        if self.age > self.max_disappeared:
            self.state = TrackState.OCCLUDED
        if self.state == TrackState.OCCLUDED:
            if self.age > self.max_age:
                self.state = TrackState.DELETED

    def mark_missed(self):
        self.age += 1
        if self.state == TrackState.TENTATIVE:
            if self.age > self.n_init:
                self.state = TrackState.DELETED
        if self.state == TrackState.CONFIRMED:
            if self.age > self.max_disappeared:
                self.state = TrackState.OCCLUDED
        elif self.state == TrackState.OCCLUDED:
            if self.age > self.max_age:
                self.state = TrackState.DELETED

    def to_delete(self):
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

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_tracker()
