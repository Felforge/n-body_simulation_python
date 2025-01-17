import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath('./'))
from barnes_hut import Octree
from basic_physics import get_distance
from simulation import create_body
from load_config import load_config

G = 6.6743e-11

def direct_force(body1, body2):
    """
    Calculate gravitational force between two bodies
    """
    dx = body2["x"] - body1["x"]
    dy = body2["y"] - body1["y"]
    dz = body2["z"] - body1["z"]
    d = get_distance(body1["x"], body1["y"], body1["z"], body2["x"], body2["y"], body2["z"])
    if d == 0:
        return 0, 0, 0
    f = (G * body1["m"] * body2["m"]) / (d**2)
    return f * dx/d, f * dy/d, f * dz/d

def calculate_direct_forces(bodies):
    """Calculate forces using direct N-body method"""
    total_forces = np.zeros((len(bodies), 3))
    for i, body1 in enumerate(bodies):
        for body2 in bodies:
            if body1 != body2:
                fx, fy, fz = direct_force(body1, body2)
                total_forces[i] += [fx, fy, fz]
    return total_forces

def create_random_bodies(n, bound):
    """
    Create n random bodies within simulation bounds
    """
    bodies = []
    for _ in range(n):
        mass = np.random.uniform(1e22, 1e32)
        x = np.random.uniform(-bound, bound)
        y = np.random.uniform(-bound, bound)
        z = np.random.uniform(-bound, bound)
        bodies.append(create_body(mass, x, y, z, 0, 0, 0))
    return bodies

def run_performance_test(n_values):
    """Run performance tests for different numbers of bodies"""
    direct_times = []
    barnes_hut_times = []
    
    config = load_config()
    bound = config["simulation_distance"]
    for n in n_values:
        print(f"Testing with {n} bodies...")
        bodies = create_random_bodies(n, bound)
        start_time = time.time()
        calculate_direct_forces(bodies) # No need to save data just need execution time
        direct_time = time.time() - start_time
        direct_times.append(direct_time)
        start_time = time.time()
        octree = Octree(bodies, [-bound, bound], [-bound, bound], [-bound, bound])
        np.array([octree.get_force(body) for body in bodies]) # No need to save data just need execution time
        barnes_hut_time = time.time() - start_time
        barnes_hut_times.append(barnes_hut_time)
        print(f"Direct method: {direct_time:.2f}s")
        print(f"Barnes-Hut: {barnes_hut_time:.2f}s")
        print(f"Speedup: {direct_time/barnes_hut_time:.2f}x\n")
    return direct_times, barnes_hut_times

def plot_results(n_values, direct_times, barnes_hut_times):
    """Plot performance comparison results"""
    _, ax = plt.subplots(figsize=(10, 5))
    ax.plot(n_values, direct_times, 'bo-', label='Direct N-body')
    ax.plot(n_values, barnes_hut_times, 'ro-', label='Barnes-Hut')
    ax.set_xlabel('Number of Bodies')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Execution Time Comparison')
    ax.legend()
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test with all these different numbers of bodies
    n_list = [100, 200, 500, 1000, 2000, 5000]
    direct_times, barnes_hut_times = run_performance_test(n_list)
    plot_results(n_list, direct_times, barnes_hut_times)
