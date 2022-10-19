import sympy as sp
import numpy as np
from dh_table import DH_Table
from kinematics import Kinematics

q_1, q_2, q_3, q_4, q_5 = sp.symbols(['theta_1', 'theta_2', 'theta_3', 'theta_4', 'theta_5'])

table = sp.Matrix([
    [q_1, 0, 0, -sp.pi/2],
    [-q_2, 0, 1, 0],
    [-q_3, 0, 1, 0],
    [-q_4, 0, 0, -sp.pi/2],
    [q_5, 0.1, 0, 0]
])

dh_table = DH_Table(table, [q_1, q_2, q_3, q_4, q_5])
kinem = Kinematics(dh_table, [(0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi)])
kinem.workspace(resolution=20, orientation=(0, np.pi/4, np.pi/4), plot=True)