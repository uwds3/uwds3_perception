import uwds3_msgs
import numpy as np
from .vector import Vector2D


class Camera(object):
    """Represents a camera sensor (real or virtual)"""

    def __init__(self, fov, width, height, clipnear, clipfar):
        """Camera constructor"""
        self.fov = fov
        self.width = width
        self.height = height
        self.clipnear = clipnear
        self.clipfar = clipfar
        self.dist_coeffs = np.zeros((4, 1))

    def center(self):
        """Returns the camera's center"""
        return Vector2D(self.width/2, self.height/2)

    def focal_length(self):
        """Returns the camera's focal length"""
        return self.height

    def camera_matrix(self):
        """Returns the camera matrix"""
        center = self.center()
        return np.array([[self.focal_length(), 0, center.y],
                        [0, self.focal_length(), center.x],
                        [0, 0, 1]], dtype="double")

    def projection_matrix(self):
        """Returns the projection matrix"""
        center = self.center()
        return np.array([[self.focal_length(), 0, center.y, 0],
                        [0, self.focal_length(), center.x, 0],
                        [0, 0, 1, 0]], dtype="double")

    def from_msg(self, msg, fov=30, clipnear=0.3, clipfar=1000):
        pass

    def to_msg(self):
        """Converts into a ROS message"""
        camera = uwds3_msgs.msg.Camera()
        camera.fov = self.fov
        camera.clipnear = self.clipnear
        camera.clipfar = self.clipfar
        camera.info.width = self.width
        camera.info.height = self.height
        camera.info.distortion_model = "blob"
        camera.info.D = list(self.dist_coeffs.flatten())
        camera.info.K = list(self.camera_matrix().flatten())
        camera.info.P = list(self.projection_matrix().flatten())
        return camera


class HumanVisualModel(object):
    FOV = 60.0 # human field of view
    WIDTH = 70 # image width resolution for rendering
    HEIGHT = 68  # image height resolution for rendering
    CLIPNEAR = 0.3 # clipnear
    CLIPFAR = 1e+3 # clipfar
    ASPECT = 1.333 # aspect ratio for rendering
    SACCADE_THRESHOLD = 0.01 # angular variation in rad/s
    SACCADE_ESPILON = 0.005 # error in angular variation
    FOCUS_DISTANCE_FIXATION = 0.1 # focus distance when performing a fixation
    FOCUS_DISTANCE_SACCADE = 0.5 # focus distance when performing a saccade

class HumanCamera(Camera):
    def __init__(self):
        self.fov = HumanVisualModel.FOV
        self.width = HumanVisualModel.WIDTH
        self.height = HumanVisualModel.HEIGHT
        self.clipnear = HumanVisualModel.CLIPNEAR
        self.clipfar = HumanVisualModel.CLIPFAR
        self.dist_coeffs = np.zeros((4, 1))
