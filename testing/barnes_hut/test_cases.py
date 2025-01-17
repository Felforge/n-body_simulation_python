import sys
import os
import pytest
import numpy as np
sys.path.append(os.path.abspath('./'))
from load_config import load_config
from simulation import create_body
from barnes_hut import Octree
from basic_physics import get_distance

G = 6.6743e-11
SOLAR_MASS = 1.989e30 # Multiplier for solar mass to kilograms
PARSEC = 3.086e16 # Multiplier for parsecs to meters

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
                                     create_body(100, 1e-100, 1e-100, 1e-100, 0, 0, 0)
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
                                     create_body(1e100, 1000, 1000, 1000, 0, 0, 0)
                                 ],
                                 [G * 1e200 * 1000 / (get_distance(0, 0, 0, 1000, 1000, 1000)**3)] * 3
                             ],
                             [ # Extremely mass ratio
                                 [
                                     create_body(1e-100, 0, 0, 0, 0, 0, 0),
                                     create_body(1e100, 1000, 0, 0, 0, 0, 0)
                                 ],
                                 [G / (get_distance(0, 0, 0, 1000, 1000, 1000)**2), 0, 0]
                             ],
                             [ # Extreme velocities with 4 bodies and variable masses
                                 [
                                     create_body(6e24, 0, 0, 0, -2.9e8, -2.9e8, -2.9e8),
                                     create_body(5e24, 2e11, 0, 0, 2.9e8, 0, 0),
                                     create_body(6e23, 0, 2e11, 0, 0, 2.9e8, 0),
                                     create_body(2e27, 0, 0, 2e11, 0, 0, 2.9e8)
                                 ],
                                 [
                                     G * 6e24 * 5e24 / 4e22,
                                     G * 6e24 * 6e23 / 4e22,
                                     G * 6e24 * 2e27 / 4e22
                                 ]
                             ],
                             [ # Spiral Galaxy
                                [
                                    create_body(1e11 * SOLAR_MASS, 0, 0, 0, 0, 0, 0),
                                    create_body(1e8 * SOLAR_MASS, 1000 * PARSEC, 0, 0, 2e5, 0, 0),
                                    create_body(1e8 * SOLAR_MASS, 0, 1000 * PARSEC, 0, 0, 2e5, 0)
                                ],
                                [
                                    G * 1e19 * (SOLAR_MASS**2) / ((1000*PARSEC)**2),
                                    G * 1e19 * (SOLAR_MASS**2) / ((1000*PARSEC)**2),
                                    0
                                ]
                             ],
                             [ # Two galaxies approaching
                              [
                                  create_body(1e11 * SOLAR_MASS, -5e4 * PARSEC, 0, 0, 1e5, 0, 0),
                                  create_body(1e11 * SOLAR_MASS, 5e4 * PARSEC, 0, 0, -1e5, 0, 0)
                              ],
                              [
                                  G * 1e22 * (SOLAR_MASS**2) / ((1e5 * PARSEC)**2), 0, 0
                              ]
                             ],
                             [ # Multiple galaxies with dark matter
                                 [
                                     create_body(1e15 * SOLAR_MASS, 0, 0, 0, 0, 0, 0),
                                     create_body(1e12 * SOLAR_MASS, 1e6 * PARSEC, 0, 0, 5e5, 0, 0),
                                     create_body(1e12 * SOLAR_MASS, -1e6 * PARSEC, 0, 0, -5e5, 0, 0),
                                     create_body(1e12 * SOLAR_MASS, 0, 1e6 * PARSEC, 0, 0, 5e5, 0),
                                     create_body(1e12 * SOLAR_MASS, 0, -1e6 * PARSEC, 0, 0, -5e5, 0)
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
    octree = Octree(bodies, [-bound, bound], [-bound, bound], [-bound, bound])
    f_x, f_y, f_z = octree.get_force(bodies[0])
    assert f_x == pytest.approx(proven[0], rel=1e-6, abs=1e-6)
    assert f_y == pytest.approx(proven[1], rel=1e-6, abs=1e-6)
    assert f_z == pytest.approx(proven[2], rel=1e-6, abs=1e-6)
