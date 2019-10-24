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
        self.euler_assignment = LinearAssignment(euler2d_distance)
        shape_predictor_config_filename = rospy.get_param("~shape_predictor_config_filename", "")
        self.facial_landmarks_estimator = FacialLandmarksEstimator(shape_predictor_config_filename)
        self.head_pose_estimator = HeadPoseEstimator()
        self.translation_estimator = TranslationEstimator()


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

        face_tracks = [t for t in self.tracks if t.class_label=="face" and t.is_confirmed()]

        x_center = image_weight/2
        y_center = image_height/2

        min_indice = None
        min_uuid = None
        min_distance = 10000
        for face_track_indice, face_track in enumerate(face_tracks):
            x_track_center = face_track.bbox.center().x
            y_track_center = face_track.bbox.center().y
            distance_from_center = math.sqrt(pow(x_center-x_track_center, 2)+pow(y_center-y_track_center, 2))
            if distance_from_center < min_distance:
                min_distance = distance_from_center
                min_indice = face_track_indice
                min_uuid = face_track.uuid
            if "facial_landmarks" in face_track.properties:
                del face_track.properties["facial_landmarks"]
        if min_indice is not None:
            shape = self.facial_landmarks_estimator.estimate(rgb_image, face_tracks[min_indice])
            if face_track.rotation is None or face_track.translation is None:
                success, rot, trans = self.head_pose_estimator.estimate(shape, camera_matrix, dist_coeffs)
            else:
                success, rot, trans = self.head_pose_estimator.estimate(shape, camera_matrix, dist_coeffs, previous_head_pose=(face_tracks[min_indice].rotation, face_tracks[min_indice].translation))
            if success is True:
                if depth_image is not None:
                    success, trans_depth = self.translation_estimator.estimate(face_track.bbox, depth_image, camera_matrix, dist_coeffs)
                    if success:
                        face_track.filter(rot.reshape((3,)), trans_depth.reshape((3,)))
                    else:
                        face_track.filter(rot.reshape((3,)), trans.reshape((3,)))
                else:
                    face_track.filter(rot.reshape((3,)), trans.reshape((3,)))
                face_track.properties["facial_landmarks"] = shape

        if depth_image is not None:
            for track in self.tracks:
                if track.is_confirmed():
                    estimate_trans = False
                    if min_uuid is not None:
                        if track.uuid != min_uuid:
                            estimate_trans = True
                        else:
                            estimate_trans = False
                    else:
                        estimate_trans = True
                    if estimate_trans is True:
                        success, trans = self.translation_estimator.estimate(track.bbox, depth_image, camera_matrix, dist_coeffs)
                        if success is True:
                            rot = np.array([math.pi/2, 0.0, 0.0])
                            track.filter(rot, trans)

        person_tracks = [t for t in self.tracks if t.class_label=="person"]

        matches, unmatched_faces, unmatched_humans = self.euler_assignment.match(person_tracks, face_tracks)

        for face_indice, person_indice in matches:
            face_tracks[face_indice].uuid = person_tracks[person_indice].uuid

        return self.tracks

    def start_track(self, rgb_image, detection):
        self.tracks.append(Track(detection, self.n_init, self.max_disappeared, self.max_age))
        if detection.class_label != "person" and detection.class_label != "face":
            self.tracks[len(self.tracks)-1].start_tracker(detection.bbox, rgb_image)
        return len(self.tracks)-1
