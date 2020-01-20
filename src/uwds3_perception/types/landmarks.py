import numpy as np
import cv2
from .vector import Vector2D
from .features import Features


class FacialLandmarks68Index(object):
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

class FacialLandmarks5Index(object):
    LEFT_EYE = 0
    RIGHT_EYE = 1
    NOSE = 4
    RIGHT_MOUTH_CORNER = 2
    LEFT_MOUTH_CORNER = 3


class FacialLandmarks(Features):
    """Represents a 68 2D point facial landmarks"""
    def __init__(self, landmarks, image_width, image_height):
        """FacialLandmarks constructor"""
        self.data = landmarks
        self.image_width = image_width
        self.image_height = image_height

    def get_point(self, index):
        """Returns the 2D point specified by the given index"""
        return Vector2D(int(self.data[index][0]*self.image_width),
                        int(self.data[index][1]*self.image_height))

    def features(self):
        """Returns the facial landmarks features"""
        return Features("facial_landmarks", np.array(self.data, np.float64), 0.80)

    def draw(self, image, color, thickness):
        """Draws the facial landmarks"""
        for idx in range(0, 67):
            if idx == 16 or idx == 21 or idx == 26 or idx == 30 or idx == 35 \
               or idx == 41 or idx == 47 or idx == 66:
                pass
            else:
                point1 = self.get_point(idx)
                point2 = self.get_point(idx+1)
                cv2.line(image, (point1.x, point1.y),
                                (point2.x, point2.y), color, thickness)

    def head_pose_points(self):
        nose = self.get_point(FacialLandmarks68Index.NOSE).to_array()
        chin = self.get_point(FacialLandmarks68Index.CHIN).to_array()
        left_eye_corner = self.get_point(FacialLandmarks68Index.LEFT_EYE_CORNER).to_array()
        right_eye_corner = self.get_point(FacialLandmarks68Index.RIGHT_EYE_CORNER).to_array()
        left_mouth_corner = self.get_point(FacialLandmarks68Index.LEFT_MOUTH_CORNER).to_array()
        right_mouth_corner = self.get_point(FacialLandmarks68Index.RIGHT_MOUTH_CORNER).to_array()
        return np.concatenate([nose,
                              chin,
                              left_eye_corner,
                              right_eye_corner,
                              left_mouth_corner,
                              right_mouth_corner], axis=0)

    def to_msg(self):
        return Features("facial_landmarks", self.features, 1.0).to_msg()

#
# class BodyLandmarks(object):
#     def __init__(self, landmarks):
#         """BodyLandmarks constructor"""
#         self.landmarks = landmarks
#
#     def get_nose(self):
#         """Returns the nose 2D point"""
#         pass
