import numpy as np
import sympy as sp

##-------------- Constants ------------##
LEGAL_EULER_ANGLES = ['rpy', 'zxz', 'xyx', 'yzy', 'zyz', 'xzx', 'yxy', 'xyz', 'yzx', 'zxy', 'xzy', 'zyx', 'yxz']

##-------------- Classes ---------------##

class SO3():
    """A class for representing the orientation of a frame
    
    Members:
    ---
    orientation --- ndarray(3, 3)(float) --- the orientation of a frame
    __legal_euler_angles --- list(string) --- the possible euler angles allowed
    """

    def __init__(self, orientation=np.eye(3)) -> "SO3":
        """Initialize the orientation
        
        Parameters:
        ---
        orientation --- ndarray(3, 3)(float) --- the orientation of the frame
        """
        self.orientation = orientation
    
    def __matmul__(self, other) -> "SO3, np.ndarray()(float)":
        """Overrides multiplication for SO3 objects"""
        if type(other) == SO3:
            return SO3(self.orientation @ other.orientation)
        return self.orientation @ other
    
    def rot_elementary(self, rotations, current=True) -> "SO3":
        """Rotates the frame about the current axis if current=True else fixed given a list of rotations. Does not
            rotate the frame if update=False.
        
        Parameters:
        ---
        rotations --- list(tuples(char, float)) --- the list of rotations to peform where rotation[0]
            reads as the first rotation to perform if about the current frame. The first value in the
            tuple is the axis to rotate about and the second is the angle in radians to rotate over.
        current --- boolean --- whether to rotate about the current or fixed frame --- default: True

        Returns:
        ---
        the orientation of the frame after the rotations are applied
        """
        self.orientation = rot_elementary(rotations, current=current) @ self.orientation
        return self
    
    def euler(self, angles="zyx") -> "tuple()(float)":
        """calculates the euler angles of the current orientation
        
        Parameters
        ---
        angles --- string --- the euler angles type to return --- default: "zyx"

        Returns
        ---
        the euler angles of the current orientation in the order of given axis
        """
        return euler(self.orientation, angles)
    
    def rot_euler(self, rotations, angles='zyx') -> "SO3":
        """Aligns the frame to an euler rotation
        
        Parameters
        ---
        rotations --- tuple(float, float, float) --- the angles to rotate over according to angles
        angles --- string --- the angles to rotate about --- default: 'zyx'
        update --- boolean --- whether to update the frame or not --- default: True

        Returns
        ---
        The orientation of the given rotations
        """

        self.orientation = rot_euler(rotations, angles)
        return self
    
    def rot_angle_axis(self, r, theta) -> "SO3":
        """ Aligns the frame to an angle axis rotation
        
        Parameters
        ---
        r --- tuple(float, float, float) --- the components (rx, ry, rz) of the vector representing the axis to rotate about
        theta --- float --- the amount to rotate about the given axis
        
        Returns
        ---
        The orientation of the frame
        """
        self.orientation = rot_angle_axis(r, theta)
        return self

    def angle_axis(self) -> "tuple()(float)":
        """calculates the angle axis representation of the current orientation
        
        Returns
        ---
        the angle axis values in the form (rx, ry, rz, theta)
        """
        return angle_axis(self.orientation)
    
    def rot_quaternion(self, neta, epsilon) -> "SO3":
        """ Aligns the frame to the given quaternion representation
        
        Parameters
        ---
        neta --- float --- np.cos(theta/2) where theta is the angle to rotate
        epsilon --- (float, float, float) --- np.sin(theta/2)r where r is the vector to rotate about
        
        Returns
        ---
        The orientation of the frame
        """
        self.orientation = rot_quaternion(neta, epsilon)
        return self
    
    def quaternion(self) -> "tuple()(float)":
        """Calculates the quaternion representation of the current orientation

        Returns
        ---
        the quaternion values of the form (neta, epsilon)
        """
        return quaternion(self.orientation)
##--------------- Functions --------------##

def rot_elementary(rotations, current=True, symbolic=False) -> "np.ndarray(float)":
        """Gives the transformation matrix of a set of elementary rotations
        
        Parameters:
        ---
        rotations --- list(tuples(char, float)) --- the list of rotations to peform where rotation[0]
            reads as the first rotation to perform if about the current frame. The first value in the
            tuple is the axis to rotate about and the second is the angle in radians to rotate over.
        current --- boolean --- whether to rotate about the current or fixed frame --- default: True
        symbolic --- boolean --- whether to perform with sympy or numpy --- default: False
        
        Returns:
        ---
        The transformation of the given rotations

        Raises:
        ---
        ValueError --- raised when the given axis is not 'x', 'y', or 'z'
        """
        if not current:
            rotations = rotations[::-1]

        transformation = np.eye(3)
        if not symbolic:
            for rotation in rotations:
                axis = rotation[0]
                angle = rotation[1]

                if axis == 'x':
                    transformation = np.array([
                        [1, 0, 0],
                        [0, np.cos(angle), -np.sin(angle)],
                        [0, np.sin(angle), np.cos(angle)]
                    ]) @ transformation
                elif axis == 'y':
                    transformation = np.array([
                        [np.cos(angle), 0, np.sin(angle)],
                        [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)]
                    ]) @ transformation
                elif axis == 'z':
                    transformation = np.array([
                        [np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0],
                        [0, 0, 1]
                    ]) @ transformation
                else:
                    raise ValueError('axis is not "x", "y", or "z"')
        else:
            for rotation in rotations:
                axis = rotation[0]
                angle = rotation[1]
                if axis == 'x':
                    transformation = sp.Matrix([
                        [1, 0, 0],
                        [0, sp.cos(angle), -sp.sin(angle)],
                        [0, sp.sin(angle), sp.cos(angle)]
                    ]) @ transformation
                elif axis == 'y':
                    transformation = sp.Matrix([
                        [sp.cos(angle), 0, sp.sin(angle)],
                        [0, 1, 0],
                        [-sp.sin(angle), 0, sp.cos(angle)]
                    ]) @ transformation
                elif axis == 'z':
                    transformation = sp.Matrix([
                        [sp.cos(angle), -sp.sin(angle), 0],
                        [sp.sin(angle), sp.cos(angle), 0],
                        [0, 0, 1]
                    ]) @ transformation
                else:
                    raise ValueError('axis is not "x", "y", or "z"')

        return transformation

def rot_euler(rotations, angles='zyx', symbolic=False) -> "np.ndarray()(float)":
    """ Returns the transformation of the euler rotations
        
        Parameters
        ---
        rotations --- tuple(float, float, float) --- the angles to rotate over according to angles
        angles --- string --- the angles to rotate about --- default: 'zyx'
        symbolic --- boolean --- performs with sp if true --- default: False

        Returns
        ---
        The orientation of the given rotations
        """

    if angles not in LEGAL_EULER_ANGLES:
        raise ValueError('invalid euler angles')
    
    if angles == 'rpy':
        return rot_elementary([('z', rotations[0]), ('y', rotations[1]), ('x', rotations[2])], current=False, symbolic=symbolic)

    return rot_elementary([(angles[0], rotations[0]), (angles[1], rotations[1]), (angles[2], rotations[2])], symbolic=symbolic)

def euler(frame, angles="zyx") -> "tuple()(float)":
    """calculates the euler angles of the frame
        
        Parameters
        ---
        frame --- np.ndarray()(float) --- the frame
        angles --- string --- the euler angles type to return. 'rpy' is also an option and is the only fixed frame euler angles available --- default: "zyx"

        Returns
        ---
        the euler angles of the frame in the order of given axis

        Raises:
        ---
        ValueError --- raises if angles is not a correct euler angles
        ZeroDivisionError --- raises if transformation is singular
        """
    if angles not in LEGAL_EULER_ANGLES:
        raise ValueError('angles is not an euler angles')

    if angles == 'rpy':
        singularity_check = [frame[0][1]==0, frame[1][0]==0, frame[2][1]==0, frame[2][2]==0]
        if all(singularity_check):
            raise ZeroDivisionError('Transformation is singular')
        phi = np.arctan2(frame[1][0], frame[0][0])
        theta = np.arctan2(-frame[2][0], np.sqrt(frame[2][1]**2 + frame[2][2]**2))
        psi = np.arctan2(frame[2][1], frame[2][2])
    
    elif angles == 'zxz':
        singularity_check = [frame[0][2]==0, frame[1][2]==0, frame[2][0]==0, frame[2][1]==0]
        if all(singularity_check):
            raise ZeroDivisionError('Transformation is singular')
        phi = np.arctan2(frame[2][1], frame[2][0])
        theta = np.arctan2(frame[2][2], np.sqrt(frame[2][0]**2 + frame[2][1]**2))
        psi = np.arctan2(-frame[1][2], frame[0][2])

    elif angles == 'xyx':
        singularity_check = [frame[0][1]==0, frame[0][2]==0, frame[1][0]==0, frame[2][0]==0]
        if all(singularity_check):
            raise ZeroDivisionError('Transformation is singular')
        phi = np.arctan2(frame[0][2], frame[0][1])
        theta = np.arctan2(frame[0][0], np.sqrt(frame[0][1]**2 + frame[0][2]**2))
        psi = np.arctan2(-frame[2][0], frame[1][0])

    elif angles == 'yzy':
        singularity_check = [frame[0][1]==0, frame[1][0]==0, frame[1][2]==0, frame[2][1]==0]
        if all(singularity_check):
            raise ZeroDivisionError('Transformation is singular')
        phi = np.arctan2(frame[2][0], frame[2][2])
        theta = np.arctan2(frame[2][1], np.sqrt(frame[2][0]**2, frame[2][2]**2))
        psi = np.arctan2(frame[2][1], -frame[0][1])

    elif angles == 'zyz':
        singularity_check = [frame[0][2]==0, frame[1][2]==0, frame[2][0]==0, frame[2][1]==0]
        if all(singularity_check):
            raise ZeroDivisionError('Transformation is singular')
        phi = np.arctan2(frame[1][2], frame[0][2])
        theta = np.arctan2(np.sqrt(frame[0][2]**2 + frame[1][2]**2), frame[2][2])
        psi = np.arctan2(frame[2][1], -frame[2][0])

    elif angles == 'xzx':
        singularity_check = [frame[0][1]==0, frame[0][2]==0, frame[1][0]==0, frame[2][0]==0]
        if all(singularity_check):
            raise ZeroDivisionError('Transformation is singular')
        phi = np.arctan2(-frame[0][1], frame[0][2])
        theta = np.arctan2(frame[0][0], np.sqrt(frame[0][1]**2 + frame[0][2]**2))
        psi = np.arctan2(frame[1][0], frame[2][0])

    elif angles == 'yxy':
        singularity_check = [frame[0][1]==0, frame[1][0]==0, frame[1][2]==0, frame[2][1]==0]
        if all(singularity_check):
            raise ZeroDivisionError('Transformation is singular')
        phi = np.arctan2(-frame[1][2], frame[1][0])
        theta = np.arctan2(frame[1][1], np.sqrt(frame[1][0]**2 + frame[1][2]**2))
        psi = np.arctan2(frame[2][1], frame[0][1])

    elif angles == 'xyz':
        singularity_check = [frame[0][0]==0, frame[1][0]==0, frame[2][1]==0, frame[2][2]==0]
        if all(singularity_check):
            raise ZeroDivisionError('Transformation is singular')
        phi = np.arctan2(frame[2][2], frame[2][1])
        theta = np.arctan2(np.sqrt(frame[2][1]**2 + frame[2][2]**2), -frame[2][0])
        psi = np.arctan2(frame[0][0], frame[1][0])

    elif angles == 'yzx':
        singularity_check = [frame[0][0]==0, frame[0][2]==0, frame[1][1]==0, frame[2][1]==0]
        if all(singularity_check):
            raise ZeroDivisionError('Transformation is singular')
        phi = np.arctan2(frame[0][0], frame[0][2])
        theta = np.arctan2(np.sqrt(frame[0][0]**2 + frame[0][2]**2), -frame[0][1])
        psi = np.arctan2(frame[1][1], frame[2][1])

    elif angles == 'zxy':
        singularity_check = [frame[0][2]==0, frame[1][0]==0, frame[1][1]==0, frame[2][2]==0]
        if all(singularity_check):
            raise ZeroDivisionError('Transformation is singular')
        phi = np.arctan2(frame[1][1], frame[1][0])
        theta = np.arctan2(np.sqrt(frame[1][1]**2 + frame[1][0]**2), -frame[1][2])
        psi = np.arctan2(frame[2][2], frame[0][2])

    elif angles == 'xzy':
        singularity_check = [frame[0][0]==0, frame[1][1]==0, frame[1][2]==0, frame[2][0]==0]
        if all(singularity_check):
            raise ZeroDivisionError('Transformation is singular')
        phi = np.arctan2(frame[1][1], -frame[1][2])
        theta = np.arctan2(np.sqrt(frame[1][1]**2 + frame[1][2]**2), frame[1][0])
        psi = np.arctan2(frame[0][0], -frame[2][0])

    elif angles == 'zyx':
        singularity_check = [frame[0][0]==0, frame[0][1]==0, frame[1][2]==0, frame[2][2]==0]
        if all(singularity_check):
            raise ZeroDivisionError('Transformation is singular')
        phi = np.arctan2(frame[0][0], -frame[0][1])
        theta = np.arctan2(np.sqrt(frame[0][0]**2 + frame[0][1]**2), frame[0][2])
        psi = np.arctan2(frame[2][0], -frame[1][2])

    elif angles == 'yxz':
        singularity_check = [frame[0][2]==0, frame[1][2]==0, frame[2][0]==0, frame[2][1]==0]
        if all(singularity_check):
            raise ZeroDivisionError('Transformation is singular')
        phi = np.arctan2(frame[2][2], -frame[2][0])
        theta = np.arctan2(np.sqrt(frame[2][2]**2 + frame[2][0]**2), frame[2][1])
        psi = np.arctan2(frame[1][1], -frame[0][1])

    return (phi, theta, psi)

def rot_angle_axis(r, theta) -> "np.ndarray()(float)":
    """ Returns the transformation of the angle axis rotation

    Parameters
    ---
    r --- tuple(float, float, float) --- the components (rx, ry, rz) of the vector representing the axis to rotate about
    theta --- float --- the amount to rotate about the given axis
    
    Returns
    ---
    The transformation of the given rotation
    """
    # normalize the axis
    r = r/np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
    return np.array([
        [r[0]**2*(1 - np.cos(theta)) + np.cos(theta), r[0]*r[1]*(1 - np.cos(theta)) - r[2]*np.sin(theta), r[0]*r[2]*(1 - np.cos(theta)) + r[1]*np.sin(theta)],
        [r[0]*r[1]*(1 - np.cos(theta)) + r[2]*np.sin(theta), r[1]**2*(1 - np.cos(theta)) + np.cos(theta), r[1]*r[2]*(1 - np.cos(theta) - r[0]*np.sin(theta))],
        [r[0]*r[2]*(1 - np.cos(theta)) - r[1]*np.sin(theta), r[1]*r[2]*(1 - np.cos(theta)) + r[0]*np.sin(theta), r[2]**2*(1 - np.cos(theta)) + np.cos(theta)]
    ])
    
def angle_axis(frame) -> "tuple()(float)":
    """ Calculates the angle axis representation of frame
    
    Parameters
    ---
    frame --- np.ndarray()(float) --- the frame to get the representation from
    
    Returns
    ---
    a tuple of the form (float, float, float, float) where the first three terms are the axis componenets and the last is the angle.
    If the angle is 0, then (0, 0, 1, 0) is returned.
    """
    # normalize the axis
    theta = np.arccos((frame[0][0] + frame[1][1] + frame[2][2] - 1)/2)
    
    if theta == 0:
        return (0, 0, 1, theta)

    if theta == np.pi:
        return (
            np.sqrt((frame[0][0] + 1)/2),
            np.sqrt((frame[1][1] + 1)/2),
            np.sqrt((frame[2][2] + 1)/2),
            theta
        )
    
    return (
        (frame[2][1] - frame[1][2])/2/np.sin(theta),
        (frame[0][2] - frame[2][0])/2/np.sin(theta),
        (frame[1][0] - frame[0][1])/2/np.sin(theta),
        theta
    )

def rot_quaternion(neta, epsilon) -> "np.ndarray()(float)":
    """Returns the transformation of the quaternion rotation
    
    Paramters
    ---
    neta --- float --- np.cos(theta/2) where theta is the angle to rotate
    epsilon --- (float, float, float) --- np.sin(theta/2)r where r is the axis to rotate about
    
    Returns
    ---
    The transformation of the given rotation
    """
    return np.array([
        [2*(neta**2 + epsilon[0]**2) - 1, 2*(epsilon[0]*epsilon[1] - neta*epsilon[2]), 2*(epsilon[0]*epsilon[2] + neta*epsilon[1])],
        [2*(epsilon[0]*epsilon[1] + neta*epsilon[2]), 2*(neta**2 + epsilon[1]**2) - 1, 2*(epsilon[1]*epsilon[2] - neta*epsilon[0])],
        [2*(epsilon[0]*epsilon[2] - neta*epsilon[1]), 2*(epsilon[1]*epsilon[2] + neta*epsilon[0]), 2*(neta**2 + epsilon[2]**2) - 1]
    ])

def quaternion(frame) -> "tuple()(float)":
    """Calculates the quaternion representation of a frame
    
    Parameters
    ---
    frame --- np.ndarray()(float) --- the frame to find the quaternion representation from
    
    Returns
    ---
    a tuple of the form (float, tuple()(float)) where the first is neta nd the second is epsilon
    """
    neta = np.sqrt(frame[0][0] + frame[1][1] + frame[2][2] + 1)/2
    epsilon = (
        np.sign(frame[2][1] - frame[1][2])*np.sqrt(frame[0][0] - frame[1][1] - frame[2][2] + 1),
        np.sign(frame[0][2] - frame[2][0])*np.sqrt(frame[1][1] - frame[2][2] - frame[0][0] + 1),
        np.sign(frame[1][0] - frame[0][1])*np.sqrt(frame[2][2] - frame[0][0] - frame[1][1] + 1)
    )/2
    return (neta, epsilon)