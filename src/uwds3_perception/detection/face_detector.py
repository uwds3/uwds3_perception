import cv2
import numpy as np
import rospy
import face_alignment
from pyuwds3.types.landmarks import FacialLandmarks
from pyuwds3.types.detection import Detection
from face_alignment.detection.sfd.sfd_detector import SFDDetector


class FaceDetector(object):
    def __init__(self):
        """ """
        self.name = "facial_landmarks"
        self.detector = SFDDetector("cuda")
        #self.face_alignment = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device="cuda", face_detector="sfd")

    def detect(self, rgb_image):
        """ """
        image_height, image_width, _ = rgb_image.shape
        detections = []
        preds = self.detector.detect_from_image(rgb_image)
        #preds = self.face_alignment.get_landmarks(rgb_image)
        if preds is not None:
            if len(preds) > 0:
                for bbox in preds:
                    xmin, ymin, xmax, ymax, confidence = bbox
                    det = Detection(xmin, ymin, xmax, ymax, "face", confidence)
                    detections.append(det)
        return detections
