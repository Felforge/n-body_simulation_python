import sys
import os
import pytest
import numpy as np
from simulation import create_body
from basic_physics import get_distance, get_cm, approximate_next, divide_matrix

sys.path.append(os.path.abspath('./'))

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
