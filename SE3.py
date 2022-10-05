import numpy as np
import SO3

class SE3():
    """A class for representing the pose of a frame
    
    Members
    ---
    pose --- np.array()(float) --- 4 by 4 matrix of the transformation matrix
    """

    def __init__(self, transformation=np.eye(4), orientation=None, p=None) -> None:
        """Initialize the pose of the coordinate transformation
        
        Parameters
        ---
        transformation --- np.ndarray()(float) --- the 4 by 4 transforamtion --- default: np.eye(4)
        orientation --- np.ndarray()(float) --- the 3 by 3 orientation --- default: None
        p --- np.ndarray()(float) --- the position of the origin --- default: None
        """
        if transformation is None:
            self.pose = np.block([[orientation, p], [np.zeros((1, 3)), 1]])
        else:
            self.pose = transformation

    def __matmul__(self, other) -> "SE3, np.ndarray()(float)":
        """Overrides multiplication for SE3.
        
        Parameters
        ---
        other --- SE3 or np.ndarray()(float) --- the multiplicand
        """
        if type(other) == SE3:
            return SE3(self.pose @ other.pose)
        return self.pose @ other

    def get_rot(self) -> "np.array()(float)":
        """
        returns the 3 by 3 np.array() orientation of the transformation
        """
        return self.pose[:3, :3]

    def get_trans(self) -> "np.array()(float)":
        """
        returns the 3 by 1 np.array() position of the origin
        """
        return self.pose[:3, 3:]

    def rotate_elem(self, theta, axis) -> "SE3":
        """Rotates the transformation theta radians about axis
        
        Parameters
        ---
        theta --- float --- the angle to rotate about axis
        axis --- char --- the axis to rotate about
        """
        self.pose = (SE3(transformation=None, orientation=SO3.rot_elementary([(axis, theta)]), p=np.zeros((3, 1))) @ self).pose
        return self

    def translate(self, p) -> "SE3":
        """Translates self.pose
        
        Parameters
        ---
        p --- np.array()(float) --- the vector of the change of origin
        """
        self.pose = (SE3(transformation=None, orientation=np.eye(3), p=p) @ self).pose
        return self

    def inv(self):
        """Returns the inverse of the transformation"""
        inv_rot = self.get_rot().T
        return SE3(transformation=None, orientation=inv_rot, p=-inv_rot@self.get_trans())
        
