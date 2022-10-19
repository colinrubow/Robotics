import sympy as sp
from functools import reduce
from SO3 import SO3
from SE3 import SE3
import matplotlib.pyplot as plt
import numpy as np

class Kinematics():
    """A class for peforming kinematics on a robot
    
    Members
    ---
    dh_table --- DH_Table --- the dh-table of the robot
    T --- [sp.Matrix()] --- symbolic transformations of frame T[i] relative to T[i-1]
    Te --- sp.Matrix() --- symbolic transformation of frame e to world frame
    """

    def __init__(self, dh_table, joint_limits) -> None:
        """Takes a dh_table and gets the basic symbolic transformations of the robot
        
        Parameters
        ---
        dh_table --- DH_Table --- the table
        joint_limits --- [tuple()(float)] --- A list of the joint limits in the form [(lower_bound, upper_bound), ...].
        """

        self.dh_table = dh_table

        # get symbolic intermediate transformations
        self.T = [sp.lambdify((self.dh_table.symbols[i]), dh_table.get_trans_symbolic(start_link=i, end_link=i+1), modules='numpy') for i in range(dh_table.num_links)]
        # get transformation of the end effector to the world frame
        self.Te = sp.lambdify((self.dh_table.symbols), reduce((lambda x, y: x@y), [dh_table.get_trans_symbolic(start_link=i, end_link=i+1) for i in range(dh_table.num_links)]), modules='numpy')

        self.joint_limits = joint_limits

    def get_trans(self, link_num, q) -> SE3:
        """gets the SE3 transformation of link_num relative to link_num - 1
        
        Parameters
        ---
        link_num --- int --- the frame of link link_num relatvie to link_num - 1
        q --- float --- the joint angle variable
        """
        return SE3(self.T[link_num - 1](q))
    
    def forward(self, q) -> SE3:
        """gets the SE3 transformation of the end effector link to the world frame
        
        Parameters
        ---
        q --- (float) --- the joint angle variables from 1 to n
        """
        return SE3(self.Te(*q))

    def workspace(self, resolution=50, orientation=None, plot=False) -> None:
        """calculates and returns and posts a plot of the workspace of link_num with respect to the base frame with the given orientation
        
        Parameters
        ---
        resolution --- int --- the number of data points to check for each joint.
        orientation --- (float) --- the rpy euler angles to check for, if None only position is found --- defualt: None
        plot --- boolean --- plots a plot if desired --- default: False
        """
        joint_values = [np.linspace(self.joint_limits[i][0], self.joint_limits[i][1], resolution) for i in range(len(self.joint_limits))]
        grid = np.meshgrid(*joint_values)
        grid = [g.flatten() for g in grid]
        num_points = len(grid[0])
        Ts = self.Te(*grid)
        # expand the single values
        for i, row in enumerate(Ts):
            for j, column in enumerate(Ts):
                if type(Ts[i][j]) != np.ndarray:
                    Ts[i][j] = [Ts[i][j]]*num_points
        if orientation is None:
            # get all the points
            Ps = np.array([[Ts[0][3][i], Ts[1][3][i], Ts[2][3][i]] for i in range(num_points)])
        else:
            SO3s = [SO3(np.array([
                [Ts[0][0][i], Ts[0][1][i], Ts[0][2][i]],
                [Ts[1][0][i], Ts[1][1][i], Ts[1][2][i]],
                [Ts[2][0][i], Ts[2][1][i], Ts[2][2][i]]
            ])) for i in range(num_points)]
            Ps = np.array([[Ts[0][3][i], Ts[1][3][i], Ts[2][3][i]] for i in range(num_points) if False not in np.isclose(SO3s[i].euler('rpy'), orientation, atol=0.05)])

        # plot Ps
        if plot:
            ax = plt.axes(projection='3d')
            if len(Ps) == 0:
                ax.scatter3D([], [], [])
            else:
                Px = Ps[:, 0]
                Py = Ps[:, 1]
                Pz = Ps[:, 2]
                ax.scatter3D(Px, Py, Pz, s=1)
            plt.show()
        
        return Ps

        
        