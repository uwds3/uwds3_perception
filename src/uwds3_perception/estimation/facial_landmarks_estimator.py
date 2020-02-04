import cv2
import numpy as np
import rospy
import math
import face_alignment
import dlib
from pyuwds3.types.detection import Detection
from pyuwds3.types.landmarks import FacialLandmarks


class FacialLandmarksEstimator(object):
    def __init__(self, shape_predictor_config_file, facing_threshold=0.17):
        """ """
        self.name = "facial_landmarks"
        self.facing_threshold = facing_threshold
        self.fan_predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device="cuda", face_detector="folder")
        self.dlib_predictor = dlib.shape_predictor(shape_predictor_config_file)

    def __is_facing(self, rx, ry, rz):
        print("rx: {} ry: {} rz: {}".format(rx, ry, rz))
        if abs(ry - math.pi) < self.facing_threshold:
            return True
        return False

    def estimate(self, rgb_image, faces):
        """ """
        image_height, image_width, _ = rgb_image.shape
        detections = []
        for f in faces:
            if hasattr(f, "pose") is True:
                if f.pose is not None:
                    if self.__is_facing(f.pose.rot.x, f.pose.rot.y, f.pose.rot.z) is True:
                        #print("{} is facing".format(f.uuid[:6]))
                        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
                        shape = self.dlib_predictor(gray, dlib.rectangle(int(f.bbox.xmin), int(f.bbox.ymin), int(f.bbox.xmax), int(f.bbox.ymax)))
                        coords = np.zeros((68, 2), dtype=np.float32)
                        for i in range(0, 68):
                            coords[i] = (shape.part(i).x, shape.part(i).y)
                        f.features[self.name] = FacialLandmarks(coords, image_width, image_height)
                        continue
            det_with_conf = np.zeros((5,), dtype=np.float32)
            det_with_conf[:4] = f.bbox.to_xyxy().flatten()[:4]
            det_with_conf[4] = 1.0
            detections.append(det_with_conf)
        if len(detections) > 0:
            preds = self.fan_predictor.get_landmarks(rgb_image, detections)
            if preds is not None:
                if len(preds) > 0:
                    for f, landmarks in zip(faces, preds):
                        f.features[self.name] = FacialLandmarks(landmarks, image_width, image_height)
