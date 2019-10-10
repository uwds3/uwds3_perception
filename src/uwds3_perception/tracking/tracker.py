import numpy as np
import rospy
import math
from .linear_assignment import LinearAssignment, iou_distance, euler_distance
from uwds3_perception.estimation.facial_landmarks_estimator import FacialLandmarksEstimator
from uwds3_perception.estimation.head_pose_estimator import HeadPoseEstimator
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
        self.euler_assignment = LinearAssignment(euler_distance)
        shape_predictor_config_filename = rospy.get_param("~shape_predictor_config_filename", "")
        self.facial_landmarks_estimator = FacialLandmarksEstimator(shape_predictor_config_filename)
        self.head_pose_estimator = HeadPoseEstimator()

    def update(self, rgb_image, detections, camera_matrix, dist_coeffs, depth_image=None):
        image_weight, image_height, _ = rgb_image.shape

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

        face_tracks = [t for t in self.tracks if t.class_label=="face"]

        x_center = image_weight/2
        y_center = image_weight/2

        min_indice = None
        min_distance = 10000
        for face_track_indice, face_track in enumerate(face_tracks):
            x_track_center = face_track.bbox.center().x
            y_track_center = face_track.bbox.center().y
            distance_from_center = math.sqrt(pow(x_center-x_track_center, 2)+pow(y_center-y_track_center, 2))
            if distance_from_center < min_distance:
                min_distance = distance_from_center
                min_indice = face_track_indice
            if "facial_landmarks" in face_track.properties:
                del face_track.properties["facial_landmarks"]
        if min_indice is not None:
            shape = self.facial_landmarks_estimator.estimate(rgb_image, face_tracks[min_indice])
            if face_track.rotation is None or face_track.translation is None:
                success, rot, trans = self.head_pose_estimator.estimate(shape, camera_matrix, dist_coeffs)
            else:
                success, rot, trans = self.head_pose_estimator.estimate(shape, camera_matrix, dist_coeffs, previous_head_pose=(face_tracks[min_indice].rotation.reshape((3,1)), face_tracks[min_indice].translation.reshape((3,1))))
            if success is True:
                final_rot = rot.reshape((3,))
                final_rot[2] += math.pi
                face_track.filter(final_rot, trans.reshape((3,)))
                face_track.properties["facial_landmarks"] = shape

        person_tracks = [t for t in self.tracks if t.class_label=="person" and t.is_confirmed()]

        matches, unmatched_faces, unmatched_humans = self.euler_assignment.match(person_tracks, face_tracks)

        for face_indice, person_indice in matches:
            face_tracks[face_indice].uuid = person_tracks[person_indice].uuid

        for face_indice in unmatched_faces:
            face_tracks[face_indice].to_delete()

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        return self.tracks

    def start_track(self, rgb_image, detection):
        self.tracks.append(Track(detection, self.n_init, self.max_disappeared, self.max_age))
        # if detection.class_label != "person":
        #     self.tracks[len(self.tracks)-1].start_tracker(detection.bbox, rgb_image)
        return len(self.tracks)-1
