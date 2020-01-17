import numpy as np
from math import pow, sqrt, pi


class ShapeType(object):
    """Represents the shape type"""
    UNKNOWN = 0
    BOX = 1
    CYLINDER = 2
    SPHERE = 3
    CAPSULE = 4
    MESH = 5


class Shape(object):
    """Represents an abstract 3D shape"""
    def __init__(self,
                 type=ShapeType.UNKNOWN,
                 dimensions=[],
                 position=[.0, .0, .0],
                 rotation=[.0, .0, .0]):
        """Shape constructor"""
        self.type = type
        self.dimensions = np.array(dimensions)
        self.position = np.array(position)
        self.rotation = np.array(rotation)

    def height(self):
        """Returns shape's height in meters"""
        raise NotImplementedError

    def width(self):
        """Returns shape's width in meters"""
        raise NotImplementedError

    def area(self):
        """Returns shape's area in cube meters"""
        raise NotImplementedError

    def radius(self):
        """Returns shape's radius in meters"""
        raise NotImplementedError

    def is_box(self):
        """Returns True if is a box"""
        return self.type == ShapeType.BOX

    def is_sphere(self):
        """Returns True if is a sphere"""
        return self.type == ShapeType.SPHERE

    def is_cylinder(self):
        """Returns True if is a cylinder"""
        return self.type == ShapeType.CYLINDER

    def is_mesh(self):
        """Returns True if is a mesh"""
        return self.type == ShapeType.MESH

    def to_msg(self):
        """Converts to ROS message"""
        raise NotImplementedError


class Box(Shape):
    """Represents a 3D box"""
    def __init__(self, x, y, z,
                 position=[.0, .0, .0],
                 rotation=[.0, .0, .0]):
        """Constructor"""
        self.type = ShapeType.BOX
        self.dimensions.append(x)
        self.dimensions.append(y)
        self.dimensions.append(z)

    def diagonal(self):
        """Returns the box's diagonal in meters"""
        return sqrt(pow(self.dimensions[0], 2)
                    + pow(self.dimensions[1], 2)
                    + pow(self.dimensions[2], 2))

    def radius(self):
        """Returns the box's radius in meters"""
        return self.diagonal()/2.0

    def width(self):
        """Returns the box's width in meters"""
        return self.dimensions[0]

    def height(self):
        """Returns the box's height in meters"""
        return self.dimensions[1]

    def area(self):
        """Returns the box's area in cube meters"""
        return self.dimensions[0] * self.dimensions[1] * self.dimensions[2]


class Sphere(Shape):
    """Represents a 3D sphere"""
    def __init__(self, d,
                 position=[.0, .0, .0],
                 rotation=[.0, .0, .0]):
        """Sphere constructor"""
        self.type = ShapeType.SPHERE
        self.dimensions.append(d)

    def radius(self):
        """Returns the sphere's radius in meters"""
        return self.width()/2.0

    def width(self):
        """Returns the sphere's width in meters"""
        return self.dimensions[0]

    def height(self):
        """Returns the sphere's height in meters"""
        return self.dimensions[0]

    def area(self):
        """Returns the sphere's area in cube meters"""
        return 4.0*pi*pow(self.radius(), 2)


class Cylinder(Shape):
    """Represents a 3D cylinder"""
    def __init__(self, d, h,
                 position=[.0, .0, .0],
                 rotation=[.0, .0, .0]):
        """Cylinder constructor"""
        self.type = ShapeType.CYLINDER
        self.dimensions.append(d)
        self.dimensions.append(h)

    def radius(self):
        """Returns the cylinder's radius in meters"""
        return self.width()/2.0

    def width(self):
        """Returns the cylinder's width in meters"""
        return self.dimensions[0]

    def height(self):
        """Returns the cylinder's height in meters"""
        return self.dimensions[1]

    def area(self):
        """Returns the cylinder's area in cube meters"""
        return 2.0*pi*self.radius()*self.height()
