import cv2
import math
import numpy as np
from tf.transformations import euler_matrix
from pyuwds3.types.vector.vector6d import Vector3D

MAX_DIST = 2.0


class HeadPoseEstimator(object):
    def __init__(self, face_3d_model_filename):
        """HeadPoseEstimator constructor"""
        self.model_3d = np.load(face_3d_model_filename)/100.0/4.6889/2.0
        self.__rotate_model(-1.57, 3.14, 0.0)

    def __check_consistency(self, trans, rot):
        consistent = True
        if trans[2][0] > MAX_DIST or trans[2][0] < 0:
            consistent = False
        return consistent

    def __rotate_model(self, rx, ry, rz):
        r_mat = euler_matrix(rx, ry, rz, "rxyz")[:3, :3]
        self.model_3d = np.dot(self.model_3d, np.linalg.inv(r_mat))

    def estimate(self, faces, camera_matrix, dist_coeffs):
        """Estimate the head pose of the given face (z forward)"""
        for f in faces:
            if f.is_confirmed():
                points_2d = f.features["facial_landmarks"].data
                if f.pose is not None:
                    r = f.pose.rotation().to_array()
                    t = f.pose.position().to_array()
                    r *= -1
                    success, rot, trans = cv2.solvePnP(self.model_3d, points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE, useExtrinsicGuess=True, rvec=r, tvec=t)
                    success = self.__check_consistency(trans, rot)
                else:
                    success, rot, trans = cv2.solvePnP(self.model_3d, points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                    success = self.__check_consistency(trans, rot)
                if success:
                    rot *= -1
                    tvec = Vector3D(x=trans[0][0], y=trans[1][0], z=trans[2][0])
                    rvec = Vector3D(x=rot[0][0], y=rot[1][0], z=rot[2][0])
                    f.update_pose(tvec, rotation=rvec)
