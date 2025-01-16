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

# Make Yoshida test cases
