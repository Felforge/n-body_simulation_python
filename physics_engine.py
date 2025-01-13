import json
import numpy as np

"""
Bodies Contains:
Body List in dict with keys being time ellapsed
x_cm, y_cm, z_cm
v_xcm, v_ycm, v_zcm
a_xcm, a_ycm, a_zcm
j_xcm, j_ycm, j_zcm
s_xvm, s_yvm, s_zvm
Body List:
mass (m)
x, y, z
v_x, v_y, v_z
a_x, a_y, a_z
j_x, j_y, j_z
s_x, s_y, s_z
"""

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
        self.config = self.load_config()
        self.prev_time = min(list(self.bodies.keys()))
        self.start_time = max(list(self.bodies.keys()))
        self.current_state = bodies[self.start_time]
        self.current_bodies = bodies[self.start_time]["body_list"]
        self.loaded = "s_xcm" in list(self.current_bodies[0].keys())

    def load_config(self):
        """
        Load config file
        """
        f = open("config.json", encoding="utf-8")
        return json.load(f)

    def get_bodies(self):
        """
        Return bodies
        """
        return self.bodies

    def get_total_force(self, target_body) -> float | float | float | float:
        """
        Add up Newtonian, Radiation, Post-Newtonian, Tidal, and Spin forces
        Return, X, Y, Z and total
        """

    def get_newtonian_force(self, target_body) -> float | float | float | float:
        """
        Use Barnes-Hut Traversal to calculate Newtonian force
        Return, X, Y, Z and total
        """

    def get_time_step(self, accuracy, accuracy_rad, softening, s, s_dot) -> float:
        """
        Get valid time step in seconds
        """
        t_1 = accuracy * np.sqrt(softening / self.get_max_acceleration_magnitude())
        if self.mode == "galactic":
            if not self.loaded:
                return t_1
            C = abs(self.divide_matrix(self.get_Q_ddot, self.get_Q_3dot))
            t_2 = accuracy_rad * C
            t_3 = abs(s) / abs(s_dot)
            return min(t_1, t_2, t_3)
        elif self.mode == "cosmic":
            return t_1

    def get_acceleration(self, target_body) -> float | float | float | float:
        """
        On galactic mode before all derivatives are calculated
        Just newtonian force is used as their is a circular dependency
        Asumed to be isolated X, Y or Z, not vector
        """
        if not self.loaded and self.mode == "galactic":
            return [num / target_body[0] for num in self.get_newtonian_force(target_body)]
        return [num / target_body[0] for num in self.get_total_force(target_body)]

    def get_max_acceleration_magnitude(self) -> int:
        """
        Return maximum acceleration magnitude across all bodies
        """
        max_a = 0
        for body in self.current_bodies:
            _, _, _, approx_a = abs(self.get_acceleration(body))
            max_a = max(max_a, approx_a)
        return max_a

    def divide_matrix(self, matrix_1, matrix_2) -> int:
        """
        Divide matrix and return max ratio of corresponding terms
        Both matricies must be same dimensions
        Needed for Q_ddot / Q_3dot
        """
        max_ratio = None
        for i, row in enumerate(matrix_1):
            for j, num in enumerate(row):
                if max_ratio is None:
                    max_ratio = num / matrix_2[i][j]
                else:
                    max_ratio = max(max_ratio, num / matrix_2[i][j])
        return max_ratio

    def get_radiation_reaction_force(self, target_body) -> float | float | float:
        """
        Get radiation reaction force components on target_body
        """
        multiple = -1 * 2 * self.G * target_body["m"] / (5 * (self.c**5))
        Q_5dot = self.get_Q_5dot()
        forces = {}
        dimensions = ["x", "y", "z"]
        for i, row in enumerate(Q_5dot):
            component = 0
            for num in row:
                component += num * (target_body[dimensions[i]] - target_body[f"{dimensions[i]}_cm"])
            forces[dimensions[i]] = multiple * component
        return forces["x"], forces["y"], forces["z"]

    def get_Q_ddot(self):
        """
        Get second derivative of Quadrupole moment with respect to time
        Returns 3x3 matrix
        """
        matrix = []
        for dim1 in ["x", "y", "z"]:
            row = []
            for dim2 in ["x", "y", "z"]:
                sum_total = 0
                for body in self.current_bodies:
                    current_sum = 0
                    if dim1 == dim2:
                        current_sum += 3 * body[dim1] * body[f"a_{dim1}"]
                        current_sum += 3 * (body[f"v_{dim1}"]**2)
                        for dim in ["x", "y", "z"]:
                            current_sum -= (body[f"v_{dim}"] - self.current_state[f"v_{dim}cm"])**2
                            current_sum -= (body[dim] - self.current_state[f"{dim}_cm"]) * (body[f"a_{dim}"] - self.current_state[f"a_{dim}cm"])
                        current_sum *= 2
                    else:
                        current_sum += 3 * body[f"a_{dim1}"] * body[dim2]
                        current_sum += 3 * body[f"a_{dim2}"] * body[dim1]
                        current_sum += 6 * body[f"v_{dim1}"] * body[f"v_{dim2}"]
                    sum_total += current_sum * body["m"]
                row.append(sum_total)
            matrix.append(row)
        return matrix


    def get_Q_3dot(self):
        """
        Get third derivative of Quadrupole moment with respect to time
        Returns 3x3 matrix
        """
        matrix = []
        for dim1 in ["x", "y", "z"]:
            row = []
            for dim2 in ["x", "y", "z"]:
                sum_total = 0
                for body in self.current_bodies:
                    current_sum = 0
                    if dim1 == dim2:
                        current_sum += 9 * body[f"v_{dim1}"] * body[f"a_{dim1}"]
                        current_sum += 3 * body[dim1] * body[f"j_{dim1}"]
                        for dim in ["x", "y", "z"]:
                            current_sum -= 3 * (body[f"v_{dim}"] - self.current_state[f"v_{dim}cm"]) * (body[f"a_{dim}"] - self.current_state[f"a_{dim}cm"])
                            current_sum -= (body[dim] - self.current_state[f"{dim}_cm"]) * (body[f"j_{dim}"] - self.current_state[f"j_{dim}cm"])
                        current_sum *= 2
                    else:
                        current_sum += 3 * body[f"j_{dim1}"] * body[dim2]
                        current_sum += 3 * body[f"j_{dim2}"] * body[dim1]
                        current_sum += 9 * body[f"a_{dim1}"] * body[f"v_{dim2}"]
                        current_sum += 9 * body[f"v_{dim1}"] * body[f"a_{dim2}"]
                    sum_total += current_sum * body["m"]
                row.append(sum_total)
            matrix.append(row)
        return matrix

    def get_Q_5dot(self):
        """
        Get fifth derivative of Quadrupole moment with respect to time
        Returns 3x3 matrix
        """
        matrix = []
        for dim1 in ["x", "y", "z"]:
            row = []
            for dim2 in ["x", "y", "z"]:
                sum_total = 0
                for body in self.current_bodies:
                    current_sum = 0
                    if dim1 == dim2:
                        current_sum += 30 * body[f"a_{dim1}"] * body[f"j_{dim1}"]
                        current_sum += 15 * body[f"v_{dim1}"] * body[f"s_{dim1}"]
                        current_sum += 3 * body[dim1] * body[f"c_{dim1}"]
                        for dim in ["x", "y", "z"]:
                            current_sum -= 10 * (body[f"a_{dim}"] - self.current_state[f"a_{dim}cm"]) * (body[f"j_{dim}"] - self.current_state[f"j_{dim}cm"])
                            current_sum -= 5 * (body[f"v_{dim}"] - self.current_state[f"v_{dim}cm"]) * (body[f"s_{dim}"] - self.current_state[f"s_{dim}cm"])
                            current_sum -= (body[dim] - self.current_state[f"{dim}_cm"]) * (body[f"c_{dim}"] - self.current_state[f"c_{dim}cm"])
                        current_sum *= 2
                    else:
                        current_sum += 30 * body[f"a_{dim1}"] * body[f"j_{dim2}"]
                        current_sum += 30 * body[f"a_{dim2}"] * body[f"j_{dim1}"]
                        current_sum += 15 * body[f"v_{dim1}"] * body[f"s_{dim2}"]
                        current_sum += 15 * body[f"v_{dim2}"] * body[f"s_{dim1}"]
                        current_sum += 3 * body[dim1] * body[f"c_{dim2}"]
                        current_sum += 3 * body[dim2] * body[f"c_{dim1}"]
                    sum_total += current_sum * body["m"]
                row.append(sum_total)
            matrix.append(row)
        return matrix

    def get_cm(self, mode="pos") -> int | int | int:
        """
        Get center of mass components
        """
        total_mass = 0
        result = [0, 0, 0]
        for body in self.current_bodies:
            for i, dim in enumerate(["x", "y", "z"]):
                total_mass += body["m"]
                if mode == "pos":
                    result[i] += body["m"] * body[dim]
                elif mode == "v":
                    result[i] += body["m"] * body[f"v_{dim}"]
                elif mode == "a":
                    result[i] += body["m"] * body[f"a_{dim}"]
        return [num / total_mass for num in result]

    def approximate_next(self, xi, yi, zi, xf, yf, zf) -> float | float | float:
        """
        Approximate next derivative using slope formula
        """
        delta_t = self.start_time - self.prev_time
        return (xf - xi) / delta_t, (yf - yi) / delta_t, (zf - zi) / delta_t

    def update_bodies(self, time_step):
        """
        Update bodies dict
        """
        for i, body in enumerate(self.current_bodies):

            for dim in ["x", "y", "z"]:
                pos, v = self.yoshida(dim, body, time_step)
                self.current_bodies[i][dim] = pos
                self.current_bodies[i][f"v_{dim}"] = v

            x, y, z, _ = _ = self.get_acceleration(body)
            self.current_bodies[i]["a_x"] = x
            self.current_bodies[i]["a_y"] = y
            self.current_bodies[i]["a_z"] = z

            if self.mode == "galactic":
                prev_state = self.bodies[self.prev_time]["body_list"][i]
                derivatives = ["a", "j", "s", "c"]
                for j, der in enumerate(derivatives[1:]):
                    prev_der = derivatives[j]
                    if prev_state[f"{prev_der}_x"] is not None:
                        xi = prev_state[f"{prev_der}_x"]
                        yi = prev_state[f"{prev_der}_y"]
                        zi = prev_state[f"{prev_der}_z"]
                        x, y, z = self.approximate_next(xi, yi, zi, x, y, z)
                        self.current_bodies[i][f"{der}_x"] = x
                        self.current_bodies[i][f"{der}_y"] = y
                        self.current_bodies[i][f"{der}_z"] = z
                    else:
                        break

        self.current_state["body_list"] = self.current_bodies

        x_cm, y_cm, z_cm = self.get_cm(mode="a")
        v_xcm, v_ycm, v_zcm = self.get_cm(mode="v")
        self.current_state["x_cm"] = x_cm
        self.current_state["y_cm"] = y_cm
        self.current_state["z_cm"] = z_cm
        self.current_state["v_xcm"] = v_xcm
        self.current_state["v_ycm"] = v_ycm
        self.current_state["v_zcm"] = v_zcm

        x_cm, y_cm, z_cm = self.get_cm(mode="a")
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
                    x_cm, y_cm, z_cm = self.approximate_next(x_icm, y_icm, z_icm, x_cm, y_cm, z_cm)
                    self.current_state[f"{der}_xcm"] = x_cm
                    self.current_state[f"{der}_ycm"] = y_cm
                    self.current_state[f"{der}_zcm"] = z_cm
                else:
                    break

        if self.prev_time != self.start_time:
            del self.bodies[self.prev_time]
        final_time = self.start_time + time_step
        self.bodies[final_time] = self.current_state

    def yoshida(self, dim, body, time_step) -> int | int:
        """
        Use Yoshida integrator to solve for position
        """
        pos_1 = self.yoshida_pos_step_1_and_4(body[dim], body[f"v_{dim}"], time_step)
        updated_body = body
        updated_body[dim] = pos_1
        v_1 = self.yoshida_v_step_1_and_3(body[f"v_{dim}"], self.get_acceleration(updated_body), time_step)
        pos_2 = self.yoshida_pos_step_2_and_3(pos_1, v_1, time_step)
        updated_body[dim] = pos_2
        v_2 = self.yoshida_v_step_2(v_1, self.get_acceleration(updated_body), time_step)
        pos_3 = self.yoshida_pos_step_2_and_3(pos_2, v_2, time_step)
        updated_body[dim] = pos_3
        v_3 = self.yoshida_v_step_1_and_3(v_2, self.get_acceleration(updated_body), time_step)
        pos_4 = self.yoshida_pos_step_1_and_4(pos_3, v_3, time_step)
        return pos_4, v_3

    def yoshida_pos_step_1_and_4(self, pos, v, time_step):
        """
        First step of Yoshida integration for position
        Velocity is assumed to be just X, Y, or Z
        """
        c_1 = (-1 / 2) * np.cbrt(2) / (2 - np.cbrt(2))
        return pos + c_1 * v * time_step

    def yoshida_v_step_1_and_3(self, v, a, time_step):
        """
        First step of Yoshida integration for velocity
        Velocity and Acceleration are assumed to be just X, Y, or Z
        """
        d_1 = -1 * np.cbrt(2) / (2 - np.cbrt(2))
        return v + d_1 * a * time_step

    def yoshida_pos_step_2_and_3(self, pos, v, time_step):
        """
        Second step of Yoshida integration for position
        Velocity is assumed to be just X, Y, or Z
        """
        w_0 = -1 * np.cbrt(2) / (2 - np.cbrt(2))
        w_1 = 1 / (2 - np.cbrt(2))
        c_2 = (w_0 + w_1) / 2
        return pos + c_2 * v * time_step

    def yoshida_v_step_2(self, v, a, time_step):
        """
        Second step of Yoshida integration for velocity
        Velocity and Acceleration are assumed to be just X, Y, or Z
        """
        d_2 = 1 / (2 - np.cbrt(2))
        return v + d_2 * a * time_step
