import sympy as sp
import numpy as np
from dh_table import DH_Table
from kinematics import Kinematics

q_1, q_2, q_3, q_4 = sp.symbols(['theta_1', 'd_2', 'd_3', 'd_4'])

table = sp.Matrix([
    [q_1, 1, 0.1, 0],
    [sp.pi/2, q_2, 0, sp.pi/2],
    [0, q_3, 0, sp.pi/2],
    [0, q_4, 0, 0]
])

dh_table = DH_Table(table, [q_1, q_2, q_3, q_4])
kinem = Kinematics(dh_table, [(0, 2*np.pi), (-1, 1), (-1, 1), (-1, 1)])
kinem.workspace(resolution=10, orientation=(0, 0, 0), plot=True)