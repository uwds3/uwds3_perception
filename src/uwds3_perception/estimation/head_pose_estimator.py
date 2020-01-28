import cv2
import math
import numpy as np
import rospy
from pyuwds3.types.vector.vector6d import Vector6D


class HeadPoseEstimator(object):
    def __init__(self):
        self.model_3d = np.float32([[0.0, 0.0, 0.0], # nose
                                    [0.0, -330.0, -65.0], # chin
                                    [-225.0, 170.0, -135.0], # left eye corner
                                    [225.0, 170.0, -135.0], # right eye corner
                                    [-150.0, -150.0, -125.0], # left mouth corner
                                    [150.0, -150.0, -125.0]]) /1000/5.0 # right mouth corner

    def estimate(self, landmarks, camera_matrix, dist_coeffs, previous_head_pose=None):

        points_2d = landmarks.head_pose_points()

        if previous_head_pose is None:
            success, rot, trans = cv2.solvePnP(self.model_3d, points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            rot[0] = .0
            rot[2] = .0
            success, rot, trans = cv2.solvePnP(self.model_3d, points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE, useExtrinsicGuess=True, rvec=rot, tvec=trans)
        else:
            r = previous_head_pose.rotation().to_array()
            t = previous_head_pose.position().to_array()
            r[1] = r[1] + math.pi
            t = (t*-1)
            r[2] = .0
            r[0] = .0
            success, rot, trans = cv2.solvePnP(self.model_3d, points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE, useExtrinsicGuess=True, rvec=r, tvec=t)
        if trans[2] > 0:
            success = False
        else:
            success = True
            trans = trans * -1
            rot[1] = rot[1] - math.pi
        return success, Vector6D(x=trans[0], y=trans[1], z=trans[2],
                                 rx=rot[0], ry=rot[1], rz=rot[2])
