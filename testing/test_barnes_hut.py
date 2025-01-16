import sys
import os
import pytest
import numpy as np
sys.path.append(os.path.abspath('./'))
from load_config import load_config
from simulation import create_body
from barnes_hut import Octree

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
                             ]
                         ])
def test_barnes_hut(bodies, proven):
    """
    Tets calculating force on object using Barnes-Hut algorithm
    First body in the list is the one being tested
    """
    bound = load_config()["simulation_distance"]
    f_x, f_y, f_z = Octree(bodies, [-bound, bound], [-bound, bound], [-bound, bound]).get_force(bodies[0])
    assert f_x == pytest.approx(proven[0], rel=1e-6)
    assert f_y == pytest.approx(proven[1], rel=1e-6)
    assert f_z == pytest.approx(proven[2], rel=1e-6)
