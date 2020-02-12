import cv2
import math
import numpy as np
from tf.transformations import euler_matrix, euler_from_matrix, is_same_transform
from pyuwds3.types.vector.vector6d import Vector3D

MAX_DIST = 2.5

RX_OFFSET = - math.pi/2.0
RY_OFFSET = math.pi
RZ_OFFSET = 0.0


class HeadPoseEstimator(object):
    def __init__(self, face_3d_model_filename):
        """HeadPoseEstimator constructor"""
        self.model_3d = np.load(face_3d_model_filename)/100.0/4.6889/2.0

    def __check_consistency(self, tvec, rvec):
        consistent = True
        if tvec[2][0] > MAX_DIST or tvec[2][0] < 0:
            consistent = False
        return consistent

    def __add_offset(self, r, x, y, z):
        r[0][0] += x
        r[1][0] += y
        r[2][0] += z

    def __rodrigues2euler(self, rvec):
        R = cv2.Rodrigues(rvec)[0]
        T = np.zeros((4, 4))
        T[3, 3] = 1.0
        euler = np.array(euler_from_matrix(R, "sxyz"))
        euler[2] *= -1
        return euler.reshape((3, 1))

    def __euler2rodrigues(self, rot):
        R = euler_matrix(rot[0][0], rot[1][0], -rot[2][0], "sxyz")
        rvec = cv2.Rodrigues(R[:3, :3])[0]
        return rvec

    def estimate(self, faces, camera_matrix, dist_coeffs):
        """Estimate the head pose of the given face (z forward for rendering)"""
        for f in faces:
            if f.is_confirmed():
                if "facial_landmarks" in f.features:
                    points_2d = f.features["facial_landmarks"].data
                    if f.pose is not None:
                        r = f.pose.rotation().to_array()
                        t = f.pose.position().to_array()
                        self.__add_offset(r, -RX_OFFSET, -RY_OFFSET, -RZ_OFFSET)
                        rvec = self.__euler2rodrigues(r)
                        success, rvec, tvec, _ = cv2.solvePnPRansac(self.model_3d, points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE, useExtrinsicGuess=True, rvec=rvec, tvec=t)
                        success = self.__check_consistency(tvec, rvec)
                    else:
                        success, rvec, tvec, _ = cv2.solvePnPRansac(self.model_3d, points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                        success = self.__check_consistency(tvec, rvec)
                    if success:
                        r = self.__rodrigues2euler(rvec)
                        self.__add_offset(r, RX_OFFSET, RY_OFFSET, RZ_OFFSET)
                        trans = Vector3D(x=tvec[0][0], y=tvec[1][0], z=tvec[2][0])
                        rot = Vector3D(x=r[0][0], y=r[1][0], z=r[2][0])
                        f.update_pose(trans, rotation=rot)
