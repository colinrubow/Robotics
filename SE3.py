import numpy as np
from Robotics.SO3 import SO3

class SE3():
    """A class for representing the pose of a frame
    
    Members:
    ---
    pose --- ndarray(4, 4)(float) --- the pose of the frame

    Methods:
    ---

    """

    def __init__(self, pose=np.ones((4, 4))) -> None:
        """Initialize the pose
        
        Parameters:
        ---
        pose --- ndarray(4, 4)(float) --- the pose of the frame
        """
        self.pose = pose
    
    