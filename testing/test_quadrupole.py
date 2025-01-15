import sys
import os
import numpy as np
sys.path.append(os.path.abspath('./'))
from simulation import create_body
from quadrupole import get_Q_ddot, get_Q_3dot, get_Q_5dot

G = 6.6743e-11

def get_newtonian_force(m1, m2, r):
    """
    Get newtonian force between two objects
    """
    return (G * m1 * m2) / (r**2)

def generate_bodies():
    """
    Generate binary star system for testing
    """
    r = 1.0e9
    m = 2e30
    v_y = np.sqrt((2 * G * m) / r)
    a_x = get_newtonian_force(m, m, r) / m

    body_1 = create_body(m, r, 0, 0, 0, v_y, 0)
    body_1["a_x"] = -a_x
    body_1["a_y"] = 0
    body_1["a_z"] = 0

    body_2 = create_body(m, -r, 0, 0, 0, -v_y, 0)
    body_2["a_x"] = a_x
    body_2["a_y"] = 0
    body_2["a_z"] = 0

    return {0 : {
        "x_cm" : 0, "y_cm" : 0, "z_cm" : 0, 
        "v_xcm" : 0, "v_ycm" : 0, "v_zcm" : 0,
        "a_xcm" : 0, "a_ycm" : 0, "a_zcm" : 0,
        "body_list" : [body_1, body_2]
        }}

def test_Q_ddot():
    """
    Test symmetry of Q_ddot function
    """
    current_state = generate_bodies()[0]
    matrix = get_Q_ddot(current_state)
    assert matrix[0][1] == matrix[1][0]
    assert matrix[0][2] == matrix[2][0]
    assert matrix[1][2] == matrix[2][1]

def test_Q_3dot():
    """
    Test symmetry of Q_3dot function
    """
    bodies = generate_bodies()
    bodies[0]["j_xcm"] = 0
    bodies[0]["j_ycm"] = 0
    bodies[0]["j_zcm"] = 0
    # Get finite differences once Newtonian Force is implemented and do time steps with that
    #matrix = PhysicsEngine(bodies).get_Q_3dot()
    #assert matrix[0][1] == matrix[1][0]
    #assert matrix[0][2] == matrix[2][0]
    #assert matrix[1][2] == matrix[2][1]
