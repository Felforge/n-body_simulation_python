import sys
import os
import pytest
import numpy as np
sys.path.append(os.path.abspath('./'))
from basic_physics import get_distance, get_cm, approximate_next, divide_matrix

@pytest.mark.parametrize("xi, yi, zi, xf, yf, zf, r",
                         [
                             (4, 5, 6, 1, 7, 10, np.sqrt(29)),
                             (1, 9, 1, 3, 4, 10, np.sqrt(110)),
                             (78, 74, 57, 23, 53, 9, np.sqrt(5770)),
                             (1, 1, 1, 5, 1, 1, 4.0),
                             (2, 2, 2, 2, 3, 2, 1.0)
                         ])
def test_distance(xi, yi, zi, xf, yf, zf, r):
    """
    Test 3D distance function
    """
    assert get_distance(xi, yi, zi, xf, yf, zf) == r

@pytest.mark.parametrize("bodies, mode, proven",
                         [
                             ([
                                 {"m" : 100, "x" : 0, "y" : 0, "z" : 0},
                                 {"m" : 100, "x" : 1, "y" : 0, "z" : 0},
                                 {"m" : 100, "x" : 0, "y" : 1, "z" : 0}],
                             "pos",
                             [1/3, 1/3, 0]),
                             ([
                                 {"m" : 100, "v_x" : 0, "v_y" : 0, "v_z" : 1},
                                 {"m" : 100, "v_x" : 0, "v_y" : 0, "v_z" : 0},
                                 {"m" : 100, "v_x" : 0, "v_y" : 0, "v_z" : -1}],
                             "v",
                             [0, 0, 0]),
                             ([
                                 {"m" : 100, "a_x" : 1, "a_y" : 1, "a_z" : 1},
                                 {"m" : 100, "a_x" : -1, "a_y" : 1, "a_z" : 1},
                                 {"m" : 100, "a_x" : 1, "a_y" : -1, "a_z" : 1},
                                 {"m" : 100, "a_x" : 1, "a_y" : 1, "a_z" : -1},
                                 {"m" : 100, "a_x" : -1, "a_y" : -1, "a_z" : 1},
                                 {"m" : 100, "a_x" : -1, "a_y" : 1, "a_z" : -1},
                                 {"m" : 100, "a_x" : 1, "a_y" : -1, "a_z" : -1},
                                 {"m" : 100, "a_x" : -1, "a_y" : -1, "a_z" : -1}],
                             "a",
                             [0, 0, 0])
                             # Add two more test cases
                         ])
def test_center_of_mass(bodies, mode, proven):
    """
    Test center of mass function
    """
    assert get_cm(bodies, mode=mode) == proven
