import numpy as np
import rospy
import math
from .linear_assignment import LinearAssignment, iou_distance, euler2d_distance
from uwds3_perception.estimation.facial_landmarks_estimator import FacialLandmarksEstimator
from uwds3_perception.estimation.head_pose_estimator import HeadPoseEstimator
from uwds3_perception.estimation.translation_estimator import TranslationEstimator
from track import Track


class Tracker(object):
    def __init__(self,
                 landmarks_prediction_model_filename,
                 min_distance=0.5,
                 n_init=6,
                 max_disappeared=5,
                 max_age=8):

        self.n_init = n_init
        self.max_disappeared = max_disappeared
        self.max_age = max_age
        self.tracks = []
        self.human_tracks = []
        self.iou_assignment = LinearAssignment(iou_distance, min_distance=min_distance)

    def update(self, rgb_image, detections, camera_matrix, dist_coeffs, depth_image=None):

        matches, unmatched_detections, unmatched_tracks = self.iou_assignment.match(self.tracks, detections)

        for detection_indice, track_indice in matches:
            if self.tracks[track_indice].class_label != "person":
                self.tracks[track_indice].update(detections[detection_indice].bbox, rgb_image=rgb_image)
            else:
                self.tracks[track_indice].update(detections[detection_indice].bbox)

        for track_indice in unmatched_tracks:
            if self.tracks[track_indice].is_confirmed():
                if self.tracks[track_indice].class_label != "person":
                    self.tracks[track_indice].predict(rgb_image)
                else:
                    self.tracks[track_indice].predict()
            else:
                self.tracks[track_indice].mark_missed()

        for detection_indice in unmatched_detections:
            self.start_track(rgb_image, detections[detection_indice])

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        return self.tracks

    def start_track(self, rgb_image, detection):
        self.tracks.append(Track(detection, self.n_init, self.max_disappeared, self.max_age))
        if detection.class_label != "person" and detection.class_label != "face":
            pass#self.tracks[len(self.tracks)-1].start_tracker(detection.bbox, rgb_image)
        return len(self.tracks)-1
