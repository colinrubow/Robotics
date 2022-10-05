import numpy as np
import sympy as sp
from SE3 import SE3
import SO3

class DH_Table():
    """A class for representing the dh parameters of a robot

    Members
    ---
    dh_table --- sympy.Matrix()(float, symbol) --- the table
    symbols --- [symbol] --- the symbols for each joint variable
    num_links --- int --- the number of links or rows in the dh-table
    """

    def __init__(self, dh_table, symbols) -> None:
        """Store the dh_table
        
        Parameters
        ---
        dh_table --- sp.Matrix()(float, symbol) --- the dh table
        symbols --- [symbol] --- the sp.symbols for the joints
        """
        self.dh_table = dh_table
        self.symbols = symbols
        self.num_links = dh_table.shape[0]
    
    def get_trans(self, q, start_link=0, end_link=-1) -> SE3:
        """Get the transformation from frame start_link to end_link
        
        Members
        ---
        q --- [float] --- set of length len(symbols) for joint angles or lengths
        start_link --- int --- the frame the chain is relative to --- default: 0
        end_link --- int --- the frame we start from --- default: -1 meaning T_e^0
        """
        end_link = end_link % (self.dh_table.shape[0] + 1)

        trans = SE3()

        for i in range(start_link, end_link):
            link = self.dh_table.row(i).subs(self.symbols[i], q[i])
            trans.rotate_elem(float(link[0]), 'z')
            trans.translate(np.array([[0], [0], [link[1]]]))
            trans.translate(np.array([[link[2]], [0], [0]]))
            trans.rotate_elem(float(link[3]), 'x')
        
        return trans
    
    def get_trans_symbolic(self, start_link=0, end_link=-1) -> sp.Matrix():
        """Get the transformation from frame start_link to end_link symbolically
        
        Members
        ---
        start_link --- int --- the frame the chain is relative to --- default: 0
        end_link --- int --- the frame we start from --- default: -1
        """
        end_link = end_link % (self.dh_table.shape[0] + 1)

        trans = sp.eye(4)

        for i in range(start_link, end_link):
            link = self.dh_table.row(i)
            trans = trans @ sp.Matrix([
                                [SO3.rot_elementary([('z', link[0])], symbolic=True), sp.zeros(3, 1)],
                                [sp.zeros(1, 3), sp.Matrix([1])]
                            ])
            trans = trans @ sp.Matrix([
                                [sp.eye(3), sp.Matrix([[0], [0], [link[1]]])],
                                [sp.zeros(1, 3), sp.Matrix([1])]
                            ])
            trans = trans @ sp.Matrix([
                                [sp.eye(3), sp.Matrix([[link[2]], [0], [0]])],
                                [sp.zeros(1, 3), sp.Matrix([1])]
                            ])
            trans = trans @ sp.Matrix([
                                [SO3.rot_elementary([('x', link[3])], symbolic=True), sp.zeros(3, 1)],
                                [sp.zeros(1, 3), sp.Matrix([1])]
                            ])
        
        return trans