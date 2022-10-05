import sympy as sp
from functools import reduce
from SE3 import SE3

class Kinematics():
    """A class for peforming kinematics on a robot
    
    Members
    ---
    dh_table --- DH_Table --- the dh-table of the robot
    T --- [sp.Matrix()] --- symbolic transformations of frame T[i] relative to T[i-1]
    Te --- sp.Matrix() --- symbolic transformation of frame e to world frame
    """

    def __init__(self, dh_table) -> None:
        """Takes a dh_table and gets the basic symbolic transformations of the robot
        
        Parameters
        ---
        dh_table --- DH_Table --- the table
        """

        self.dh_table = dh_table

        # get symbolic intermediate transformations
        self.T = [sp.lambdify((self.dh_table.symbols[i]), dh_table.get_trans_symbolic(start_link=i, end_link=i+1), modules='numpy') for i in range(dh_table.num_links)]
        # get transformation of the end effector to the world frame
        self.Te = sp.lambdify((self.dh_table.symbols), reduce((lambda x, y: x@y), [dh_table.get_trans_symbolic(start_link=i, end_link=i+1) for i in range(dh_table.num_links)]), modules='numpy')

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
