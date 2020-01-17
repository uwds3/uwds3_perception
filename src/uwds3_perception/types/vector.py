################################################################################
## Underworlds spatial vector library
################################################################################

import cv2
import numpy as np
import geometry_msgs
from tf.transformations import translation_matrix, euler_matrix, euler_from_matrix
from tf.transformations import translation_from_matrix, quaternion_from_matrix
from tf.transformations import euler_from_quaternion

################################################################################
# Scalar related
################################################################################


class ScalarStabilized(object):
    """Represents a stabilized scalar"""
    def __init__(self, x=.0, vx=.0, p_cov=.01, m_cov=.1):
        """ScalarStabilized constructor"""
        self.x = x
        self.vx = vx
        self.filter = cv2.KalmanFilter(2, 1, 0)
        self.filter.statePost = self.to_array()
        self.filter.statePre = self.filter.statePost
        self.filter.transitionMatrix = np.array([[1, 1],
                                                 [0, 1]], np.float32)
        self.filter.measurementMatrix = np.array([[1, 1]], np.float32)
        self.update_cov(p_cov, m_cov)

    def from_array(self, array):
        """Updates the scalar stabilized state from array"""
        assert array.shape == (2, 1)
        self.x = array[0]
        self.vx = array[1]
        self.filter.statePre = self.filter.statePost

    def to_array(self):
        """Returns the scalar stabilizer state array representation"""
        return np.array([[self.x], [self.vx]], np.float32)

    def position(self):
        """Returns the scalar's position"""
        return self.x

    def velocity(self):
        """Returns the scalar's velocity"""
        return self.vx

    def update(self, x):
        """Updates/Filter the scalar"""
        self.filter.predict()
        self.filter.correct(np.array([[np.float32(x)]]))
        self.from_array(self.filter.statePost)

    def predict(self):
        """Predicts the scalar state"""
        self.filter.predict()
        self.from_array(self.filter.statePost)

    def update_cov(self, p_cov, m_cov):
        """Updates the process and measurement covariances"""
        self.filter.processNoiseCov = np.array([[1, 0],
                                                [0, 1]], np.float32) * p_cov

        self.filter.measurementNoiseCov = np.array([[1]], np.float32) * m_cov


################################################################################
# 2D Vector related
################################################################################


class Vector2D(object):
    """Represents a 2D vector"""
    def __init__(self, x=.0, y=.0):
        """Vector2D constructor"""
        self.x = x
        self.y = y

    def to_array(self):
        """Returns the 2D vector array representation"""
        return np.array([self.x, self.y], np.float32)

    def draw(self, frame, color, thickness):
        """Draws the 2D point"""
        cv2.circle(frame, (self.x, self.y), 2, color, thickness=1)

    def __len__(self):
        return 2

    def __add__(self, vector):
        assert len(vector) == 2
        pass

    def __sub__(self, vector):
        pass


class Vector2DStabilized(Vector2D):
    """"Represents a 2D vector stabilized"""
    def __init__(self, x=.0, y=.0, vx=.0, vy=.0, p_cov=0.01, m_cov=0.1):
        """Vector2DStablized constructor"""
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.filter = cv2.KalmanFilter(4, 2, 0)
        self.filter.statePost = self.to_array()
        self.filter.statePre = self.filter.statePost
        self.filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)

        self.filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.update_cov(p_cov, m_cov)

    def from_array(self, array):
        """Updates the 2D vector stabilized state from array"""
        assert array.shape == (4, 1)
        self.x = array[0]
        self.y = array[1]
        self.vx = array[2]
        self.vy = array[3]
        self.filter.statePost = array
        self.filter.statePre = self.filter.statePost

    def to_array(self):
        """Returns the 2D vector stabilizer state array representation"""
        return np.array([[self.x], [self.y], [self.vx], [self.vy]], np.float32)

    def position(self):
        """Returns the 2D vector stabilized position"""
        return Vector2D(x=self.x, y=self.y)

    def velocity(self):
        """"Returns the 2D vector stabilized velocity"""
        return Vector2D(x=self.vx, y=self.vy)

    def update(self, x, y):
        """Updates/Filter the 2D vector"""
        self.filter.predict()
        self.filter.correct(np.array([[x], [y]], np.float32))
        self.from_array(self.filter.statePost)

    def predict(self):
        """Predicts the 2D vector"""
        self.filter.predict()
        self.from_array(self.filter.statePost)

    def update_cov(self, p_cov, m_cov):
        """Updates the process and measurement covariances"""
        self.filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]], np.float32) * p_cov

        self.filter.measurementNoiseCov = np.array([[1, 0],
                                                    [0, 1]], np.float32) * m_cov


################################################################################
# 3D Vector related
################################################################################


class Vector3D(object):
    """Represents a 3D vector"""
    def __init__(self, x=.0, y=.0, z=.0):
        """Point 3D constructor"""
        self.x = x
        self.y = y
        self.z = z

    def from_array(self):
        """ """
        assert len(array) == 3
        self.x = array[0]
        self.y = array[1]
        self.z = array[2]

    def to_array(self):
        """Returns the 3D point's array representation"""
        return np.array([self.x, self.y, self.z])

    def to_msg(self):
        """Converts to ROS message"""
        return geometry_msgs.msg.Point(x=self.x, y=self.y, z=self.z)


class Vector3DStabilized(Vector3D):
    """Represents a 3D vector stabilized"""
    def __init__(self, x=.0, y=.0, z=.0,
                 vx=.0, vy=.0, vz=.0,
                 p_cov=0.01, m_cov=0.1):
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vz
        self.vz = vz
        self.filter = cv2.KalmanFilter(6, 3, 0)
        self.filter.statePost = self.to_array()
        self.filter.statePre = self.filter.statePost
        self.filter.transitionMatrix = np.array([[1, 0, 0, 1, 0, 0],
                                                 [0, 1, 0, 0, 1, 0],
                                                 [0, 0, 1, 0, 0, 1],
                                                 [0, 0, 0, 1, 0, 0],
                                                 [0, 0, 0, 0, 1, 0],
                                                 [0, 0, 0, 0, 0, 1]], np.float32)

        self.filter.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                                  [0, 1, 0, 0, 0, 0],
                                                  [0, 0, 1, 0, 0, 0]], np.float32)
        self.update_cov(p_cov, m_cov)

    def from_array(self, array):
        """ """
        assert array.shape == (6, 1)
        self.x = array[0]
        self.y = array[1]
        self.z = array[2]
        self.vx = array[3]
        self.vy = array[4]
        self.vz = array[5]
        self.filter.statePost = array
        self.filter.statePre = self.filter.statePost

    def to_array(self):
        """ """
        return np.array([[self.x],
                         [self.y],
                         [self.z],
                         [self.vx],
                         [self.vy],
                         [self.vz]], np.float32)

    def position(self):
        """ """
        return Vector3D(x=self.x, y=self.y, z=self.z)

    def velocity(self):
        """ """
        return Vector3D(x=self.vx, y=self.vy, z=self.vz)

    def update(self, x, y, z):
        """Updates/Filter the 3D vector"""
        self.filter.predict()
        self.filter.correct(np.array([[np.float32(x)],
                                      [np.float32(y)],
                                      [np.float32(z)]]))
        self.from_array(self.filter.statePost)

    def predict(self):
        """Predicts the 3D vector based on motion model"""
        self.filter.predict()
        self.from_array(self.filter.statePost)

    def update_cov(self, p_cov, m_cov):
        """Updates the process and measurement covariances"""
        self.filter.processNoiseCov = np.array([[1, 0, 0, 0, 0, 0],
                                                [0, 1, 0, 0, 0, 0],
                                                [0, 0, 1, 0, 0, 0],
                                                [0, 0, 0, 1, 0, 0],
                                                [0, 0, 0, 0, 1, 0],
                                                [0, 0, 0, 0, 0, 1]], np.float32) * p_cov

        self.filter.measurementNoiseCov = np.array([[1, 0, 0, 0, 0, 0],
                                                    [0, 1, 0, 0, 0, 0],
                                                    [0, 0, 1, 0, 0, 0]], np.float32) * m_cov


################################################################################
# 6D Vector related
################################################################################


class Vector6D(object):
    """Represents a 6D pose (position + orientation)"""
    def __init__(self, x=.0, y=.0, z=.0,
                 rx=.0, ry=.0, rz=.0):
        """Vector 6D constructor"""
        self.position = Vector3D(x=x, y=y, z=z)
        self.rotation = Vector3D(x=rx, y=ry, z=rz)

    def from_array(self, array):
        """"""
        assert len(array) == 6
        self.position.x = array[0]
        self.position.y = array[1]
        self.position.z = array[2]
        self.rotation.x = array[3]
        self.rotation.y = array[4]
        self.rotation.z = array[5]

    def to_array(self):
        """Returns the 6D vector's array representation"""
        return np.array([self.position.x, self.position.y, self.position.z,
                         self.rotation.x, self.rotation.y, self.rotation.z])

    def from_transform(self, transform):
        """Set the vector from an homogenous transform"""
        r = euler_from_matrix(transform, "rxyz")
        t = translation_from_matrix(transform)
        self.position.x = t[0]
        self.position.y = t[1]
        self.position.z = t[2]
        self.rotation.x = r[0]
        self.rotation.y = r[1]
        self.rotation.z = r[2]

    def transform(self):
        """Returns the homogenous transform"""
        mat_pos = translation_matrix(self.position.to_array())
        mat_rot = euler_matrix(self.rotation.x,
                               self.rotation.y,
                               self.rotation.z, "rxyz")
        return np.dot(mat_pos, mat_rot)

    def inv(self):
        """Inverse the vector"""
        return Vector6D().from_transform(np.linalg.inv(self.transform()))

    def from_quaternion(self, rx, ry, rz, rw):
        euler = euler_from_quaternion([rx, ry, rz, rw])
        self.rotation.x = euler[0]
        self.rotation.y = euler[1]
        self.rotation.z = euler[2]

    def quaternion(self):
        """Returns the rotation quaternion"""
        return quaternion_from_matrix(self.transform())

    def __len__(self):
        """Returns the vector's lenght"""
        return 6

    def __add__(self, vector):
        """ """
        return Vector6D().from_transform(np.dot(self.transform(), vector.transform()))

    def __sub__(self, vector):
        """ """
        return Vector6D().from_transform(np.dot(self.transform(), vector.inv().transform()))

    def to_msg(self):
        """Converts to ROS message"""
        msg = geometry_msgs.msg.Pose()
        msg.position.x = self.position.x
        msg.position.y = self.position.y
        msg.position.z = self.position.z
        q = self.quaternion()
        msg.orientation.x = q[0]
        msg.orientation.y = q[1]
        msg.orientation.z = q[2]
        msg.orientation.x = q[3]
        return msg


class Vector6DStabilized(Vector6D):
    """ """
    def __init__(self, x=.0, y=.0, z=.0,
                 vx=.0, vy=.0, vz=.0,
                 rx=.0, ry=.0, rz=.0,
                 vrx=.0, vry=.0, vrz=.0, p_cov=0.01, m_cov=0.1):
        """ """
        self.position = Vector3DStabilized(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, p_cov=p_cov, m_cov=m_cov)
        self.rotation = Vector3DStabilized(x=rx, y=ry, z=rz, vx=vrx, vy=vry, vz=vrz, p_cov=p_cov, m_cov=m_cov)

    def from_array(self, array):
        """ """
        assert array.shape == (12, 1)
        self.position.from_array(array[:6])
        self.rotation.from_array(array[6:])

    def to_array(self):
        """ """
        return np.concatenate(self.positon.to_array(),
                              self.rotation.to_array(),
                              axis=0)
