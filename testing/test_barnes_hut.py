import sys
import os
import pytest
import numpy as np
sys.path.append(os.path.abspath('./'))
from load_config import load_config
from simulation import create_body
from barnes_hut import Octree

G = 6.6743e-11

@pytest.mark.parametrize("bodies, proven",
                         [
                             [ # Single Body
                                 [create_body(100, 0, 0, 0, 0, 0, 0)],
                                 [0, 0, 0]
                             ],
                             [ # Same Position
                                 [
                                     create_body(100, 0, 0, 0, 0, 0, 0),
                                     create_body(100, 0, 0, 0, 0, 0, 0)
                                 ],
                                 [0, 0, 0]
                             ],
                             [ # Infinitely Close
                                 [
                                     create_body(100, 0, 0, 0, 0, 0, 0),
                                     create_body(100, 1e-15, 1e-15, 1e-15, 0, 0, 0)
                                 ],
                                 [0, 0, 0]
                             ],
                             [ # Past simulation boundary (should still work for this)
                                 [
                                     create_body(100, 1.5e21, 1.5e21, 1.5e21, 0, 0, 0),
                                     create_body(100, -1.5e21, -1.5e21, -1.5e21, 0, 0, 0)
                                 ],
                                 [0, 0, 0]
                             ],
                             [ # One quad apart
                                 [
                                     create_body(100, 0, 0, 0, 0, 0, 0),
                                     create_body(100, 1.41911e21, 1.41911e21, 1.41911e21, 0, 0, 0)
                                 ],
                                 [0, 0, 0]
                             ],
                             [ # Zero mass
                                 [
                                     create_body(0, 0, 0, 0, 0, 0, 0),
                                     create_body(100, 1000, 1000, 1000, 0, 0, 0)
                                 ],
                                 [0, 0, 0]
                             ],
                             [ # Negative mass
                                 [
                                     create_body(-100, 0, 0, 0, 0, 0, 0),
                                     create_body(100, 1000, 1000, 1000, 0, 0, 0)
                                 ],
                                 [G * -0.01, G * -0.01, G * -0.01]
                             ],
                             [ # Extremely large mass
                                 [
                                     create_body(1e100, 0, 0, 0, 0, 0, 0),
                                     create_body(1e100, 1e10, 1e10, 1e10, 0, 0, 0)
                                 ],
                                 [G * 1e194, G * 1e194, G * 1e194]
                             ]
                         ])
def test_barnes_hut(bodies, proven):
    """
    Tets calculating force on object using Barnes-Hut algorithm
    First body in the list is the one being tested
    """
    bound = load_config()["simulation_distance"]
    octree = Octree(bodies, [-bound, bound], [-bound, bound], [-bound, bound])
    print(octree.mass)
    f_x, f_y, f_z = octree.get_force(bodies[0])
    assert f_x == pytest.approx(proven[0], rel=1e-6, abs=1e-6)
    assert f_y == pytest.approx(proven[1], rel=1e-6, abs=1e-6)
    assert f_z == pytest.approx(proven[2], rel=1e-6, abs=1e-6)
