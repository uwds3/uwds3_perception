import cv2
import numpy as np

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=500,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)


def LucasKanadeTracker(self):
    def __init__():
        """ """
        self.corners = None

    def update(self, rgb_image, detection, depth_image=None):
        """ """
        mask = np.full((rgb_image.shape[0], rgb_image.shape[0], 1), 255)
        xmin = detection.bbox.xmin
        ymin = detection.bbox.ymin
        xmax = detection.bbox.xmax
        ymax = detection.bbox.ymax
        mask[ymin:ymax, xmin:xmax] = detection.mask
        # dilate the mask ?!
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        if self.corners is None:
            self.corners = cv2.goodFeaturesToTrack(gray_image, mask=mask, **feature_params)

    def predict(self, rgb_image, depth_image=None):
        """ """
