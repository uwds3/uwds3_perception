import rospy
from pywuds3.types.vector.vector3d import Vector3D


class ObjectPoseEstimator(object):
    def estimate(self, objects, depth_image, camera_matrix, dist_coeffs):
        for o in objects:
            if o.bbox.depth is not None:
                if hasattr(o, "pose") is True:
                    fx = camera_matrix[0][0]
                    rotation = Vector3D()
                    position = Vector3D()
                    position.x =
                    o.update_pose(position, rotation)
        raise NotImplementedError()
