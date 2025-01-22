import numpy as np
from barnes_hut import Octree
from load_config import load_config
from basic_physics import get_cm, approximate_next, divide_matrix
from quadrupole import get_Q_ddot, get_Q_3dot, get_Q_5dot
from yoshida import *

class PhysicsEngine:
    """
    Physice engine for N-body simulation
    Modes are "galactic" and "cosmic"
    """
    def __init__(self, bodies, mode="galactic"):
        self.G = 6.6743e-11
        self.c = 3e8
        self.bodies = bodies
        self.mode = mode
        self.config = load_config()
        self.prev_time = min(list(self.bodies.keys()))
        self.start_time = max(list(self.bodies.keys()))
        self.current_state = bodies[self.start_time]
        self.current_bodies = bodies[self.start_time]["body_list"]
        self.loaded = "s_xcm" in list(self.current_bodies[0].keys())

    def get_total_force(self, target_body) -> float | float | float | float:
        """
        Add up Newtonian, Radiation, Post-Newtonian, Tidal, and Spin forces for galactic
        Add up Newtonian, Dark Matter, and Dark Enegery forces for cosmic
        Return, X, Y, Z and total
        """

    def get_newtonian_force(self, target_body) -> float | float | float:
        """
        Use Barnes-Hut Traversal to calculate Newtonian force
        Return, X, Y, Z
        """
        bound = self.config["simulation_distance"]
        octree = Octree(self.current_bodies, [-bound, bound], [-bound, bound], [-bound, bound])
        return octree.get_force(target_body)

    def get_time_step(self, accuracy, accuracy_rad, softening, s, s_dot) -> float:
        """
        Get valid time step in seconds
        """
        t_1 = accuracy * np.sqrt(softening / self.get_max_acceleration_magnitude())
        if self.mode == "galactic":
            if not self.loaded:
                return t_1
            C = abs(divide_matrix(get_Q_ddot(self.current_state), get_Q_3dot(self.current_state)))
            t_2 = accuracy_rad * C
            t_3 = abs(s) / abs(s_dot)
            return min(t_1, t_2, t_3)
        elif self.mode == "cosmic":
            return t_1

    def get_acceleration(self, target_body) -> float | float | float:
        """
        On galactic mode before all derivatives are calculated
        Just newtonian force is used as their is a circular dependency
        """
        if not self.loaded and self.mode == "galactic":
            accels = [num / target_body["m"] for num in self.get_newtonian_force(target_body)]
        else:
            accels = [num / target_body["m"] for num in self.get_total_force(target_body)]
        accels_dict = {}
        for i, dim in enumerate(["x", "y", "z"]):
            accels_dict[dim] = accels[i]
        return accels_dict

    def get_max_acceleration_magnitude(self) -> int:
        """
        Return maximum acceleration magnitude across all bodies
        """
        max_a = 0
        for body in self.current_bodies:
            accels_dict = abs(self.get_acceleration(body))
            a = np.sqrt(sum(value ** 2 for value in accels_dict.values()))
            max_a = max(max_a, a)
        return max_a

    def get_radiation_reaction_force(self, target_body) -> float | float | float:
        """
        Get radiation reaction force components on target_body
        """
        multiple = -1 * 2 * self.G * target_body["m"] / (5 * (self.c**5))
        Q_5dot = get_Q_5dot(self.current_state)
        forces = {}
        dimensions = ["x", "y", "z"]
        for i, row in enumerate(Q_5dot):
            component = 0
            for num in row:
                component += num * (target_body[dimensions[i]] - target_body[f"{dimensions[i]}_cm"])
            forces[dimensions[i]] = multiple * component
        return forces["x"], forces["y"], forces["z"]

    def update_bodies(self, time_step):
        """
        Update bodies dict
        """
        for i, body in enumerate(self.current_bodies):

            new_body = self.yoshida(body, time_step)
            self.current_bodies[i] = new_body

            accels_dict = self.get_acceleration(body)
            for dim in ["x", "y", "z"]:
                self.current_bodies[i][f"a_{dim}"] = accels_dict[dim]

            if self.mode == "galactic":
                prev_state = self.bodies[self.prev_time]["body_list"][i]
                derivatives = ["a", "j", "s", "c"]
                for j, der in enumerate(derivatives[1:]):
                    prev_der = derivatives[j]
                    if prev_state[f"{prev_der}_x"] is not None:
                        xi = prev_state[f"{prev_der}_x"]
                        yi = prev_state[f"{prev_der}_y"]
                        zi = prev_state[f"{prev_der}_z"]
                        x, y, z = approximate_next(self.prev_time, self.start_time, xi, yi, zi, x, y, z)
                        self.current_bodies[i][f"{der}_x"] = x
                        self.current_bodies[i][f"{der}_y"] = y
                        self.current_bodies[i][f"{der}_z"] = z
                    else:
                        break

        self.current_state["body_list"] = self.current_bodies

        x_cm, y_cm, z_cm = get_cm(self.current_bodies, mode="a")
        v_xcm, v_ycm, v_zcm = get_cm(self.current_bodies, mode="v")
        self.current_state["x_cm"] = x_cm
        self.current_state["y_cm"] = y_cm
        self.current_state["z_cm"] = z_cm
        self.current_state["v_xcm"] = v_xcm
        self.current_state["v_ycm"] = v_ycm
        self.current_state["v_zcm"] = v_zcm

        x_cm, y_cm, z_cm = get_cm(self.current_bodies, mode="a")
        self.current_state["a_xcm"] = x_cm
        self.current_state["a_ycm"] = y_cm
        self.current_state["a_zcm"] = z_cm

        if self.mode == "galactic":
            prev_state = self.bodies[self.prev_time]
            derivatives = ["a", "j", "s", "c"]
            for i, der in enumerate(derivatives[1:]):
                prev_der = derivatives[i]
                if prev_state[f"{prev_der}_xcm"] is not None:
                    x_icm = prev_state[f"{prev_der}_xcm"]
                    y_icm = prev_state[f"{prev_der}_ycm"]
                    z_icm = prev_state[f"{prev_der}_zcm"]
                    x_cm, y_cm, z_cm = approximate_next(self.prev_time, self.start_time, x_icm, y_icm, z_icm, x_cm, y_cm, z_cm)
                    self.current_state[f"{der}_xcm"] = x_cm
                    self.current_state[f"{der}_ycm"] = y_cm
                    self.current_state[f"{der}_zcm"] = z_cm
                else:
                    break

        if self.prev_time != self.start_time:
            del self.bodies[self.prev_time]
        final_time = self.start_time + time_step
        self.bodies[final_time] = self.current_state

    def yoshida(self, body, time_step) -> float | float:
        """
        Use Yoshida integrator to solve for position and velocity
        """
        updated_body = body
        dims = ["x", "y", "z"]

        for dim in dims:
            updated_body[dim] = yoshida_pos_step_1_and_4(body[dim], body[f"v_{dim}"], time_step)

        for dim in dims:
            updated_body[f"v_{dim}"] = yoshida_v_step_1_and_3(body[f"v_{dim}"],
                                            self.get_acceleration(updated_body)[dim], time_step)

        for dim in dims:
            updated_body[dim] = yoshida_pos_step_2_and_3(updated_body[dim],
                                        updated_body[f"v_{dim}"], time_step)

        for dim in dims:
            updated_body[f"v_{dim}"] = yoshida_v_step_2(updated_body[f"v_{dim}"],
                                            self.get_acceleration(updated_body)[dim], time_step)

        for dim in dims:
            updated_body[dim] = yoshida_pos_step_2_and_3(updated_body[dim],
                                        updated_body[f"v_{dim}"], time_step)

        for dim in dims:
            updated_body[f"v_{dim}"] = yoshida_v_step_1_and_3(updated_body[f"v_{dim}"],
                                            self.get_acceleration(updated_body)[dim], time_step)

        for dim in dims:
            updated_body[dim] = yoshida_pos_step_1_and_4(updated_body[dim],
                                        updated_body[f"v_{dim}"], time_step)

        return updated_body
