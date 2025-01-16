import sys
import os
import pytest
import numpy as np
from basic_physics import get_distance, get_cm, approximate_next, divide_matrix

sys.path.append(os.path.abspath('./'))

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
                                 {"m" : 100, "v_x" : 3, "v_y" : 4, "v_z" : 5},
                                 {"m" : 200, "v_x" : 5, "v_y" : 4, "v_z" : 3},
                                 {"m" : 300, "v_x" : -3, "v_y" : -4, "v_z" : -5}],
                             "v",
                             [2/3, 0, -2/3]),
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
                             [0, 0, 0]),
                             ([
                                 {"m" : 100, "x" : 0, "y" : 0, "z" : 0},
                                 {"m" : 300, "x" : 1e10, "y" : 0, "z" : 0},
                                 {"m" : 500, "x" : 1e10, "y" : 1e10, "z" : 0},
                                 {"m" : 700, "x" : 0, "y" : 1e10, "z" : 0},
                                 {"m" : 800, "x" : 0, "y" : 0, "z" : 1e10},
                                 {"m" : 600, "x" : 1e10, "y" : 0, "z" : 1e10},
                                 {"m" : 400, "x" : 1e10, "y" : 1e10, "z" : 1e10},
                                 {"m" : 200, "x" : 0, "y" : 1e10, "z" : 1e10}],
                             "pos",
                             [5e9, 5e9, 5e10/9])
                         ])
def test_center_of_mass(bodies, mode, proven):
    """
    Test center of mass function
    """
    assert get_cm(bodies, mode=mode) == proven

@pytest.mark.parametrize("ti, tf, xi, yi, zi, xf, yf, zf, proven",
                         [
                             [2, 4, 2, 5, 0, 2, 6, 2, [0, 1/2, 1]],
                             [0, 10, 0, 0, 0, -10, 0, 1, [-1, 0, 0.1]],
                             [0, 1e10, 0, 0, 0, 0, 0, 0, [0, 0, 0]],
                             [1, 1.001, 0, 0, 0, 1, 1, 1, [1000, 1000, 1000]],
                             [-5, -1, 10, 0, 0, 0, 0, 0, [-2.5, 0, 0]]
                         ])
def test_approximate_next(ti, tf, xi, yi, zi, xf, yf, zf, proven):
    """
    Test function to approximate next derivative
    """
    x, y, z = approximate_next(ti, tf, xi, yi, zi, xf, yf, zf)
    assert x == pytest.approx(proven[0], rel=1e-6)
    assert y == pytest.approx(proven[1], rel=1e-6)
    assert z == pytest.approx(proven[2], rel=1e-6)

@pytest.mark.parametrize("matrix_1, matrix_2, proven",
                         [
                             [
                                 [[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]],
                                 [[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]],
                                 0
                             ],
                             [
                                 [[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]],
                                 [[-9, -8, -7],
                                  [-6, -5, -4],
                                  [-3, -2, -1]],
                                 -1
                             ],
                             [
                                 [[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]],
                                 [[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]],
                                 0
                             ],
                             [
                                 [[1, 2, 3],
                                  [4, 5, 6],
                                  [7, 8, 9]],
                                 [[0.1, 0.1, 0.1],
                                  [0.1, 0.1, 0.1],
                                  [0.1, 0.1, 0.1]],
                                 90
                             ],
                             [
                                 [[0.1, 0.1, 0.1],
                                  [0.1, 0.1, 0.1],
                                  [0.1, 0.1, 0.1]],
                                 [[10, 10, 10],
                                  [10, 10, 10],
                                  [10, 10, 10]],
                                 0.01
                             ],
                             [
                                 [[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]],
                                 [[1, 0, 1],
                                  [0, 1, 0],
                                  [1, 0, 2]],
                                 1
                             ],
                             [
                                 [[100, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 0.01]],
                                 [[1, 1, 1],
                                  [1, 1, 1],
                                  [1, 1, 1]],
                             100]
                         ])
def test_matrix_division(matrix_1, matrix_2, proven):
    """
    Test division of matricies for scalar return value
    """
    assert divide_matrix(matrix_1, matrix_2) == proven
