import numpy as np
import cv2
from math import pow, sqrt
from .vector import Vector2D, Vector2DStabilized, ScalarStabilized
from .features import Features


class BoundingBox(object):
    """Represents a 2D aligned bounding box in the image space"""

    def __init__(self, xmin, ymin, xmax, ymax, depth=None):
        """BoundingBox constructor"""
        self.xmin = int(xmin)
        self.ymin = int(ymin)
        self.xmax = int(xmax)
        self.ymax = int(ymax)
        self.depth = depth

    def center(self):
        """Returns the bbox's center in pixels"""
        return Vector2D(self.xmin + self.width()/2, self.ymin + self.height()/2)

    def width(self):
        """Returns the bbox's width in pixels"""
        return self.xmax - self.xmin

    def height(self):
        """Returns the bbox's height in pixels"""
        return self.ymax - self.ymin

    def diagonal(self):
        """Returns the bbox's diagonal in pixels"""
        return sqrt(pow(self.width(), 2)+pow(self.height(), 2))

    def radius(self):
        """Returns the bbox's radius in pixels"""
        return self.diagonal()/2.0

    def area(self):
        """Returns the bbox's area in pixels"""
        return (self.width()+1)*(self.height()+1)

    def draw(self, frame, color, thickness):
        """Draws the bbox"""
        cv2.rectangle(frame, (self.xmin, self.ymin), (self.xmax, self.ymax), color, thickness)

    def to_array(self):
        return np.array([self.xmin, self.ymin, self.xmax, self.ymax], np.float32)

    def features(self, image_width, image_height, max_depth=25):
        """Returns the bbox geometric features"""
        features = [self.xmin/float(image_width),
                    self.ymin/float(image_height),
                    self.xmax/float(image_width),
                    self.ymax/float(image_height),
                    min(self.depth/float(max_depth), float(1.0))]
        return Features("geometric", np.array(features).flatten(), 0.89)


class BoundingBoxStabilized(BoundingBox):
    def __init__(self, xmin, ymin, xmax, ymax, depth=None, p_cov=10000, m_cov=1):
        self.xmin = int(xmin)
        self.ymin = int(ymin)
        self.xmax = int(xmax)
        self.ymax = int(ymax)
        w = xmax - xmin
        h = ymax - ymin
        center = self.center()
        x = center.x
        y = center.y
        a = w/float(h)
        self.center_filter = Vector2DStabilized(x=x, y=y, p_cov=p_cov, m_cov=m_cov)
        self.aspect_filter = ScalarStabilized(x=a, p_cov=p_cov, m_cov=m_cov)
        self.height_filter = ScalarStabilized(x=h, p_cov=p_cov, m_cov=m_cov)
        if depth is not None:
            self.depth_filter = ScalarStabilized(x=depth, p_cov=p_cov, m_cov=m_cov)
            self.depth = float(depth)
        else:
            self.depth_filter = None
            self.depth = None

    def update(self, xmin, ymin, xmax, ymax, depth=None):
        w = xmax - xmin
        h = ymax - ymin
        self.center_filter.update(xmin + w/2.0, ymin + h/2.0)
        self.aspect_filter.update(w/float(h))
        self.height_filter.update(h)
        if self.depth_filter is not None:
            if depth is not None:
                self.depth_filter.update(depth)
        h = self.height_filter.x
        w = self.height_filter.x * self.aspect_filter.x
        x = self.center_filter.x
        y = self.center_filter.y
        self.xmin = x - w/2.0
        self.ymin = y - h/2.0
        self.xmax = x + w/2.0
        self.ymax = y + h/2.0
        if self.depth_filter is not None:
            self.depth = self.depth_filter.x
        assert self.width() > 20 and self.height() > 20
        if self.depth is not None:
            assert self.depth < 100

    def predict(self):
        self.center_filter.predict()
        self.aspect_filter.predict()
        self.height_filter.predict()
        if self.depth_filter is not None:
            self.depth_filter.predict()
        h = self.height_filter.x
        w = self.height_filter.x * self.aspect_filter.x
        x = self.center_filter.x
        y = self.center_filter.y
        self.xmin = x - w/2.0
        self.ymin = y - h/2.0
        self.xmax = x + w/2.0
        self.ymax = y + h/2.0
        if self.depth_filter is not None:
            self.depth = self.depth_filter.x

    def update_cov(self, p_cov, m_cov):
        self.center_filter.update_cov(p_cov, m_cov)
        self.aspect_filter.update_cov(p_cov, m_cov)
        self.height_filter.update_cov(p_cov, m_cov)
        if self.depth_filter is not None:
            self.depth_filter.update_cov(p_cov, m_cov)


#####################################################################
# Metrics associated with the bbox
#####################################################################


def iou(bbox_a, bbox_b):
    """Returns the intersection over union metric"""
    xmin = int(min(bbox_a.xmin, bbox_b.xmin))
    ymin = int(min(bbox_a.ymin, bbox_b.ymin))
    xmax = int(max(bbox_a.xmax, bbox_b.xmax))
    ymax = int(max(bbox_a.ymax, bbox_b.ymax))
    intersection_area = BoundingBox(xmin, ymin, xmax, ymax).area()
    union_area = bbox_a.area() + bbox_b.area()
    if float(union_area - intersection_area) == 0.0:
        return 0.0
    else:
        return intersection_area / float(union_area - intersection_area)


def centroid(bbox_a, bbox_b):
    """Returns the euler distance between centroids"""
    xa = bbox_a.center().x
    xb = bbox_b.center().x
    ya = bbox_a.center().y
    yb = bbox_b.center().y
    return sqrt(pow(xa-xb, 2)+pow(ya-yb, 2))


def overlap(bbox_a, bbox_b):
    """Returns the overlap ratio"""
    xmin = int(min(bbox_a.xmin, bbox_b.xmin))
    ymin = int(min(bbox_a.ymin, bbox_b.ymin))
    xmax = int(max(bbox_a.xmax, bbox_b.xmax))
    ymax = int(max(bbox_a.ymax, bbox_b.ymax))
    intersection_area = BoundingBox(xmin, ymin, xmax, ymax).area()
    return intersection_area / bbox_a.area()
