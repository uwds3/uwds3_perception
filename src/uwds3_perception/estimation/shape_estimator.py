import math
from uwds3_msgs.msg import PrimitiveShape


class ShapeEstimator(object):
    def __init__(self):
        self.shape_types = ["cylinder"]

    def estimate(self, track, camera_matrix, dist_coeffs, shape_type="cylinder"):
        shape = PrimitiveShape()
        if track.translation is None:
            return False, None
        if shape_type == "cylinder":
            shape.type = PrimitiveShape.CYLINDER
            z = track.translation[2]
            w = track.bbox.width()
            h = track.bbox.height()
            fx = camera_matrix[0][0]
            fy = camera_matrix[1][1]
            w_3d = w * z / fx
            h_3d = h * z / fy
            shape.dimensions.append(w_3d)
            shape.dimensions.append(w_3d)
            shape.dimensions.append(h_3d)
            return True, shape
        raise NotImplementedError("Only cylinder shape is currently available")
