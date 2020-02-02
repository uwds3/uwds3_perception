import cv2
import math
import numpy as np
from pyuwds3.types.vector.vector6d import Vector3D


class HeadPoseEstimator(object):
    def __init__(self):
        """HeadPoseEstimator constructor"""
        self.model_3d = np.float32([[0.0, 0.0, 0.0], # nose
                                    [0.0, -330.0, -65.0], # chin
                                    [-225.0, 170.0, -135.0], # left eye corner
                                    [225.0, 170.0, -135.0], # right eye corner
                                    [-150.0, -150.0, -125.0], # left mouth corner
                                    [150.0, -150.0, -125.0]])/1000/4.7 # right mouth corner

    def estimate(self, face_tracks, camera_matrix, dist_coeffs):
        """Estimate the head pose of the given face tracks"""
        for track in face_tracks:
            if track.is_confirmed():
                points_2d = track.features["facial_landmarks"].head_pose_points()

                if track.pose is None:
                    success, rot, trans = cv2.solvePnP(self.model_3d, points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                    rot[0] = .0
                    rot[2] = .0
                    rot[1] *= -1
                    rot[1] -= math.pi
                    success, rot, trans = cv2.solvePnP(self.model_3d, points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE, useExtrinsicGuess=True, rvec=rot, tvec=trans)
                else:
                    r = track.pose.rotation().to_array()
                    t = track.pose.position().to_array()
                    t *= -1
                    r[0] = .0
                    r[2] = .0
                    r[1] *= -1
                    r[1] -= math.pi
                    success, rot, trans = cv2.solvePnP(self.model_3d, points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE, useExtrinsicGuess=True, rvec=r, tvec=t)

                trans *= -1
                rot[1] += math.pi
                rot[1] *= -1
                rot[2] = .0
                track.update_pose(Vector3D(x=trans[0], y=trans[1], z=trans[2]),
                                  rotation=Vector3D(x=rot[0], y=rot[1], z=rot[2]))
