import cv2
import numpy as np
from pyuwds3.types.features import Features


class ColorFeaturesEstimator(object):
    """Represents a color features estimator"""
    def __init__(self):
        self.name = "color"

    def estimate(self, rgb_image, detections=None):
        """Extracts hue channel histogram as a features vector"""
        for det in detections:
            x = int(det.bbox.center().x)
            y = int(det.bbox.center().y)
            w = int(det.bbox.width())
            h = int(det.bbox.height())
            cropped_image = rgb_image[y:y+h, x:x+w]
            hsv = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2HSV)
            hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            hist = hist / np.max(hist)
            det.features[self.name] = Features(self.name, hist[:, 0], h/float(rgb_image.shape[0]))
