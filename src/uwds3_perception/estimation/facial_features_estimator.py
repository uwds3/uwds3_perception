import cv2
import numpy as np
from pyuwds3.types.features import Features
from .face_frontalizer_estimator import FaceFrontalizerEstimator


class FacialFeaturesEstimator(object):
    """Represents the facial description estimator"""
    def __init__(self, face_3d_model_filename, embedding_model_filename, frontalize=False):
        """FacialFeaturesEstimator constructor"""
        self.name = "facial_description"
        self.model = cv2.dnn.readNetFromTorch(embedding_model_filename)
        if frontalize is True:
            self.frontalizer = FaceFrontalizerEstimator(face_3d_model_filename)
        else:
            self.frontalizer = None

    def estimate(self, rgb_image, face_detections, camera_matrix=None, dist_coeffs=None):
        """Extracts the facial description features"""
        cropped_imgs = []
        for det in face_detections:
            x = int(det.bbox.center().x)
            y = int(det.bbox.center().y)
            w = int(det.bbox.width())
            h = int(det.bbox.height())
            cropped_imgs.append(rgb_image[y:y+h, x:x+w])
            if self.frontalizer is not None:
                frontalized_img = self.frontalizer.estimate(rgb_image, det, camera_matrix, dist_coeffs)
                cropped_imgs.append(np.round(frontalized_img).astype(np.uint8))
        if len(cropped_imgs) > 0:
            blob = cv2.dnn.blobFromImages(cropped_imgs,
                                          1.0 / 255,
                                          (96, 96),
                                          (0, 0, 0),
                                          swapRB=False,
                                          crop=False)
            self.model.setInput(blob)
            for det, features in zip(face_detections, self.model.forward()):
                det.features[self.name] = Features(self.name, np.array(features).flatten(), h/rgb_image.shape[0])
