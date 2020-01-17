import cv2
import math
import numpy as np
from .facial_landmarks_estimator import RIGHT_EYE_CORNER, LEFT_EYE_CORNER, LEFT_MOUTH_CORNER, RIGHT_MOUTH_CORNER, CHIN, NOSE

class HeadPoseEstimator(object):
    def __init__(self):
        self.model_3d = np.float32([[0.0, 0.0, 0.0], # nose
                                    [0.0, -330.0, -65.0], # chin
                                    [-225.0, 170.0, -135.0], # left eye corner
                                    [225.0, 170.0, -135.0], # right eye corner
                                    [-150.0, -150.0, -125.0], # left mouth corner
                                    [150.0, -150.0, -125.0]]) /1000/4.5 # right mouth corner

    def estimate(self, landmarks, camera_matrix, dist_coeffs, previous_head_pose=None):

        points_2d = landmarks.head_pose_points()

        if previous_head_pose is None:
            _, rot, trans = cv2.solvePnP(self.model_3d, points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        else:
            r = previous_head_pose.rotation.to_array()
            t = previous_head_pose.translation.to_array()
            if r is not None and t is not None:
                r[1] = r[1] + math.pi
                t = (t*-1)
                _, rot, trans = cv2.solvePnP(self.model_3d, points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE, useExtrinsicGuess=True, rvec=r, tvec=t)
            else:
                _, rot, trans = cv2.solvePnP(self.model_3d, points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if trans[2] > 0:
            success = False
        else:
            success = True
            trans = trans * -1
            rot[1] = rot[1] - math.pi
        return success, trans, rot
