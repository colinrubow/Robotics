from dh_table import DH_Table
from kinematics import Kinematics
import sympy as sp
import numpy as np

q_1, q_2, q_3, q_4 = sp.symbols(['theta_1', 'd_2', 'd_3', 'd_4'])

table = sp.Matrix([
    [q_1, 1, 0.25, 0],
    [np.pi/2, q_2, 0, np.pi/2],
    [0, q_3, 0, np.pi/2],
    [0, q_4, 0, 0],
])

dh_table = DH_Table(table, [q_1, q_2, q_3, q_4])

kine = Kinematics(dh_table)

T_e_0 = kine.forward((0, 0, 0, 0))

print(T_e_0.pose)