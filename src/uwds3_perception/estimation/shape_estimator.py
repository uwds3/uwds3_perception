
class ShapeEstimator(object):
    """ """
    def estimate(self, rgb_image, objects_tracks, camera_matrix, dist_coeffs):
        """ """
        for o in objects_tracks:
            if o.bbox.depth is not None:
                if o.label != "person":
                    if not o.has_shape():
                        o.shape = o.bbox.cylinder()
                        o.shape.pose.pos.x = .0
                        o.shape.pose.pos.y = .0
                        o.shape.pose.pos.z = .0
                else:
                    o.shape = o.bbox.cylinder()
                    o.shape.pose.pos.x = .0
                    o.shape.pose.pos.y = .0
                    o.shape.pose.pos.z = .0

    def __compute_dominant_color(self, rgb_image, bbox):
        raise NotImplementedError()
