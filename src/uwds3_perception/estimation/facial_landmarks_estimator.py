import cv2
import numpy as np
import dlib
from pyuwds3.types.landmarks import FacialLandmarks


class FacialLandmarksEstimator(object):
    def __init__(self, shape_predictor_config_file):
        """ """
        self.name = "facial_landmarks"
        self.predictor = dlib.shape_predictor(shape_predictor_config_file)

    def estimate(self, rgb_image, face_detections):
        """ """
        image_height, image_width, _ = rgb_image.shape
        for det in face_detections:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            shape = self.predictor(gray, dlib.rectangle(int(det.bbox.xmin), int(det.bbox.ymin), int(det.bbox.xmax), int(det.bbox.ymax)))
            coords = np.zeros((68, 2), dtype=np.float32)
            for i in range(0, 68):
                coords[i] = (shape.part(i).x/float(image_width), shape.part(i).y/float(image_height))
            det.features[self.name] = FacialLandmarks(coords, image_width, image_height)
