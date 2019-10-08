import numpy as np
import rospy
from .linear_assignment import LinearAssignment, iou_distance, euler_distance
from uwds3_perception.estimation.facial_landmarks_estimator import FacialLandmarksEstimator
from uwds3_perception.estimation.head_pose_estimator import HeadPoseEstimator
from track import Track


class HumanTracker(object):
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
        #self.overlap_assignment = LinearAssignment(overlap_distance)
        shape_predictor_config_filename = rospy.get_param("~shape_predictor_config_filename", "")
        self.facial_landmarks_estimator = FacialLandmarksEstimator(shape_predictor_config_filename)
        self.head_pose_estimator = HeadPoseEstimator()

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

        face_tracks = [t for t in self.tracks if t.class_label=="face"]

        for face_track in face_tracks:
            shape = self.facial_landmarks_estimator.estimate(rgb_image, face_track)
            if face_track.rotation is None or face_track.translation is None:
                success, rot, trans = self.head_pose_estimator.estimate(shape, camera_matrix, dist_coeffs)
            else:
                success, rot, trans = self.head_pose_estimator.estimate(shape, camera_matrix, dist_coeffs, previous_head_pose=(face_track.rotation.reshape((3,1)), face_track.translation.reshape((3,1))))
            if success is True:
                face_track.rotation = rot.reshape((3,))
                face_track.translation = trans.reshape((3,))
                face_track.properties["facial_landmarks"] = shape

        # person_tracks = [t for t in self.tracks if t.class_label=="person"]

        # matches, unmatched_persons, unmatched_humans = self.iou_assignment.match(self.human_tracks, person_tracks)
        #
        # for person_indice, human_indice in matches:
        #     self.human_tracks[human_indice].update_track("person", self.tracks[person_indice])
        #
        # for human_indice in unmatched_humans:
        #     self.human_tracks[human_indice].mark_missed()
        #
        # for person_indice in unmatched_persons:
        #     self.start_human_track(self.tracks[person_indice])
        #
        # matches, unmatched_faces, unmatched_humans = self.overlap_assignment.match(self.human_tracks, face_tracks)
        #
        # for face_indice, human_indice in matches:
        #     #print "update face"
        #     self.human_tracks[human_indice].update_track("face", self.tracks[face_indice])
        #     #print self.human_tracks[human_indice].rotation
        #
        # self.human_tracks = [t for t in self.human_tracks if not t.is_deleted()]

        return self.tracks
    #
    # def start_human_track(self, person_track):
    #     self.human_tracks.append(HumanTrack(person_track))
    #     return len(self.human_tracks)-1

    def start_track(self, rgb_image, detection):
        self.tracks.append(Track(detection, self.n_init, self.max_disappeared, self.max_age))
        if detection.class_label != "person":
            self.tracks[len(self.tracks)-1].start_tracker(detection.bbox, rgb_image)
        return len(self.tracks)-1
