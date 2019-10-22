import cv2
import math
import numpy as np

class TranslationEstimator(object):
    def estimate(self, bbox, depth_image, camera_matrix, dist_coeffs):
        x = bbox.center().x
        y = bbox.center().y
        x = depth_image.shape[1]-1 if x > depth_image.shape[1] else x
        y = depth_image.shape[0]-1 if y > depth_image.shape[0] else y
        z = depth_image[y][x]
        if math.isnan(z):
            return False, None
        cx = camera_matrix[0][2]
        cy = camera_matrix[1][2]
        fx = camera_matrix[0][0]
        fy = camera_matrix[1][1]
        x_3d = (x - cx) * z / fx
        y_3d = (y - cy) * z / fy
        z_3d = z
        return True, np.array([x_3d, y_3d, z_3d])
