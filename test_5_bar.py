import sympy as sp
import numpy as np
from dh_table import DH_Table

q_2, q_2p, theta_3, theta_3p, a = sp.symbols(['theta_2', 'theta_2p', 'theta_3', 'theta_3p', 'a'])

prime_table = sp.Matrix([
    [np.pi, 0, a, 0],
    [-np.pi/2 + q_2p, 0, a, 0],
    [-np.pi/4 + theta_3p, 0, np.sqrt(2)*a, 0]
])

table = sp.Matrix([
    [0, 0, a, 0],
    [np.pi/2 + q_2, 0, a, 0],
    [np.pi/4 + theta_3, 0, np.sqrt(2)*a, 0]
])

dh_table = DH_Table(table, [a, q_2, theta_3])
dh_table_prime = DH_Table(prime_table, [a, q_2p, theta_3p])

T_3_0 = sp.simplify(dh_table.get_trans_symbolic(0, 3))
sp.pprint(T_3_0, use_unicode=False)
print()
print()
print('prime')
T_3p_0 = sp.simplify(dh_table_prime.get_trans_symbolic(0, 3))
sp.pprint(T_3p_0, use_unicode=False)