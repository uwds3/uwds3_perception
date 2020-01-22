import cv2
import numpy as np
import dlib
from pyuwds3.types.landmarks import FacialLandmarks


POINT_OF_SIGHT = 27
RIGHT_EYE_CORNER = 36
LEFT_EYE_CORNER = 45
NOSE = 30
MOUTH_UP = 51
MOUTH_DOWN = 57
MOUTH_UP = 51
RIGHT_MOUTH_CORNER = 48
LEFT_MOUTH_CORNER = 54
RIGHT_EAR = 0
LEFT_EAR = 16
CHIN = 8


class FacialLandmarksEstimator(object):
    def __init__(self, shape_predictor_config_file):
        """ """
        self.predictor = dlib.shape_predictor(shape_predictor_config_file)

    def estimate(self, rgb_image, face_track):
        """ """
        image_height, image_width, _ = rgb_image.shape
        bbox = face_track.bbox
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        offset = (bbox.ymax-bbox.ymin)*.05
        shape = self.predictor(gray, dlib.rectangle(int(bbox.xmin), int(bbox.ymin+offset), int(bbox.xmax), int(bbox.ymax-offset)))
        coords = np.zeros((68, 2), dtype=np.float32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x/float(image_width), shape.part(i).y/float(image_height))
        return FacialLandmarks(coords, image_width, image_height)
