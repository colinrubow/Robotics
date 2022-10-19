import sympy as sp
import numpy as np
from dh_table import DH_Table
from kinematics import Kinematics

q_1, q_2 = sp.symbols(['theta_1', 'theta_2'])

table = sp.Matrix([
    [q_1, 0, 1, 0],
    [q_2, 0, 1, 0]
])

dh_table = DH_Table(table, [q_1, q_2])
kinem = Kinematics(dh_table, [(0, 2*np.pi), (0, 2*np.pi)])
kinem.workspace(resolution=50, orientation=(np.pi/2, 0, 0), plot=True)
