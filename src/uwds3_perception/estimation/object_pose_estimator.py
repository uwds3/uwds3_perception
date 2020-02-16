import numpy as np
from pyuwds3.types.vector.vector6d import Vector6D


class ObjectPoseEstimator(object):
    def estimate(self, objects, view_matrix, camera_matrix, dist_coeffs):
        """ """
        for o in objects:
            if o.bbox.depth is not None:
                fx = camera_matrix[0][0]
                fy = camera_matrix[1][1]
                cx = fx/2.0
                cy = fy/2.0
                c = o.bbox.center()
                z = o.bbox.depth
                x = (cx - c.x) * z / fx
                y = (cy - c.y) * z / fx
                sensor_transform = Vector6D(x=x, y=y, z=z).transform()
                world_pose = Vector6D().from_transform(np.dot(view_matrix, sensor_transform))
                position = world_pose.position()
                rotation = world_pose.rotation()
                o.update_pose(position, rotation)
