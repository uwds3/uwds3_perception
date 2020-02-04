import cv2
import numpy as np


class FaceFrontalizerEstimator(object):
    """Python implementation of :
    Tal Hassner, Shai Harel*, Eran Paz* and Roee Enbar,
    Effective Face Frontalization in Unconstrained Images,
    IEEE Conf. on Computer Vision and Pattern Recognition (CVPR 2015)

    modified from https://github.com/ChrisYang/facefrontalisation"""
    def __init__(self, face_3d_model_filename):
        raise NotImplementedError()

    def estimate(self, rgb_image, face_detection, camera_matrix, dist_coeffs):
        raise NotImplementedError()
