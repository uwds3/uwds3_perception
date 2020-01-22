import cv2
import numpy as np
from pyuwds3.types.features import Features


class FacialFeaturesExtractor(object):
    """Represents the facial description extractor"""
    def __init__(self, embedding_model_filename):
        """FacialFeaturesExtractor constructor"""
        self.model = cv2.dnn.readNetFromTorch(embedding_model_filename)
        # self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        # self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

    def extract(self, rgb_image, face_detections, facial_landmarks=None):
        """Extracts the facial description features"""
        cropped_imgs = []
        if facial_landmarks is not None:
            assert len(facial_landmarks) == len(face_detections)
        for det in face_detections:
            x = int(det.bbox.center().x)
            y = int(det.bbox.center().y)
            w = int(det.bbox.width())
            h = int(det.bbox.height())
            cropped_imgs.append(rgb_image[y:y+h, x:x+w])
        if len(cropped_imgs) > 0:
            blob = cv2.dnn.blobFromImages(cropped_imgs,
                                          1.0 / 255,
                                          (96, 96),
                                          (0, 0, 0),
                                          swapRB=False,
                                          crop=False)
            self.model.setInput(blob)
            for det, features in zip(face_detections, self.model.forward()):
                det.features["facial_description"] = Features("facial_description",
                                                              np.array(features).flatten(),
                                                              h/rgb_image.shape[0])
