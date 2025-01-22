import sys
import os
import pytest
import numpy as np
sys.path.append(os.path.abspath('./'))
from simulation import create_body
from physics_engine import PhysicsEngine

def calculate_position(xi, v, a, t):
    """
    Calculate final position using kinematics
    """
    return xi + v * t + (1/2) * a * (t**2)

def calculate_velocity(vi, a, t):
    """
    Calculate final velocity using kinematics
    """
    return vi + a * t

@pytest.mark.parametrize("v", [0, 1, 10, 100])
def test_yoshida_single_body(v):
    """
    Test Yoshida integration on a single particle
    Only using X dimension
    """
    body = create_body(1, 0, 0, 0, v, 0, 0)
    bodies = {0 : {
        "x_cm" : 0, "y_cm" : 0, "z_cm" : 0, 
        "v_xcm" : v, "v_ycm" : 0, "v_zcm" : 0,
        "body_list" : [body]
        }}

    physics_engine = PhysicsEngine(bodies)

    time_step = 0.01
    new_body = body
    for _ in range(100):
        new_body = physics_engine.yoshida(body, time_step)
    assert new_body["x"] == pytest.approx(v, rel=1e-6, abs=1e-6)
