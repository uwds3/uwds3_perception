
class ObjectPoseEstimator(object):
    def __init__(self):
        raise NotImplementedError()

    def estimate(self, object_tracks, depth_image, camera_matrix, dist_coeffs):
        raise NotImplementedError()
