import json
import numpy as np

"""
Bodies Contains:
Body List in dict with keys being time ellapsed
Each body list contains:
mass (m)
x, y, z
v_x, v_y, v_z
a_x, a_y, a_z
x_cm, y_cm, z_cm
v_xcm, v_ycm, v_zcm
a_xcm, a_ycm, a_zcm
"""

class PhysicsEngine:
    """
    Physice engine for N-body simulation
    """
    def __init__(self, bodies):
        self.G = 6.6743e-11
        self.c = 3e8
        self.bodies = bodies
        self.config = self.load_config()
        self.loaded = bodies[0] is not None
        self.prev_time = min(self.bodies.keys)
        self.start_time = max(self.bodies.keys)

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

    def get_total_force(self, target_body) -> int | int | int | int:
        """
        Add up Newtonian, Radiation, Post-Newtonian, Tidal, and Spin forces
        Return, X, Y, Z and total
        """

    def get_newtonian_force(self, target_body) -> int | int | int | int:
        """
        Use Barnes-Hut Traversal to calculate Newtonian force
        Return, X, Y, Z and total
        """

    def get_time_step(self, accuracy, accuracy_rad, softening, s, s_dot) -> int:
        """
        Get valid time step in seconds
        """
        t_1 = accuracy * np.sqrt(softening / self.get_max_acceleration_magnitude())
        if not self.loaded:
            return t_1
        C = abs(self.divide_matrix(self.get_Q_ddot, self.get_Q_3dot))
        t_2 = accuracy_rad * C
        t_3 = abs(s) / abs(s_dot)
        return min(t_1, t_2, t_3)

    def get_acceleration(self, target_body) -> int | int | int | int:
        """
        Approximate acceleration using just Newtonian Force
        This is needed to start the simulation up
        Asumed to be isolated X, Y or Z, not vector
        """
        if not self.loaded:
            return [num / target_body[0] for num in self.get_newtonian_force(target_body)]
        return [num / target_body[0] for num in self.get_total_force(target_body)]

    def get_max_acceleration_magnitude(self) -> int:
        """
        Return maximum acceleration magnitude across all bodies
        """
        max_a = 0
        for body in self.bodies[1].values:
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

    def get_radiation_reaction_force(self, target_body) -> int | int | int:
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

    def get_Q_3dot(self):
        """
        Get third derivative of Quadrupole moment with respect to time
        Returns 3x3 matrix
        """

    def get_Q_5dot(self):
        """
        Get fifth derivative of Quadrupole moment with respect to time
        Returns 3x3 matrix
        """

    def get_cm(self, mode="pos") -> int | int | int:
        """
        Get center of mass components
        """
        total_mass = 0
        result = [0, 0, 0]
        for body in self.bodies[self.start_time]:
            for i, dim in enumerate(["x", "y", "z"]):
                total_mass += body["m"]
                if mode == "pos":
                    result[i] += body["m"] * body[dim]
                elif mode == "v":
                    result[i] += body["m"] * body[f"v_{dim}"]
                elif mode == "a":
                    result[i] += body["m"] * body[f"a_{dim}"]
        return [num / total_mass for num in result]

    def get_a_cm_dot(self) -> int | int | int:
        """
        Get components of derivative of acceleration of center of mass of system
        """

    def update_bodies(self, time_step):
        """
        Update bodies dict
        """
        final_time = self.start_time + time_step
        self.bodies[final_time] = self.bodies[self.start_time]
        for i, body in enumerate(self.bodies[self.start_time].values):
            a_x, a_y, a_z, _ = _ = self.get_acceleration(body)
            self.bodies[final_time][i]["a_x"] = a_x
            self.bodies[final_time][i]["a_y"] = a_y
            self.bodies[final_time][i]["a_z"] = a_z
            for dim in ["x", "y", "z"]:
                pos, v = self.yoshida(dim, body, time_step)
                self.bodies[final_time][i][dim] = pos
                self.bodies[final_time][i][f"v_{dim}"] = v
        x_cm, y_cm, z_cm = self.get_cm(mode="a")
        v_xcm, v_ycm, v_zcm = self.get_cm(mode="v")
        a_xcm, a_ycm, a_zcm = self.get_cm(mode="a")
        self.bodies[final_time]["x_cm"] = x_cm
        self.bodies[final_time]["y_cm"] = y_cm
        self.bodies[final_time]["z_cm"] = z_cm
        self.bodies[final_time]["v_xcm"] = v_xcm
        self.bodies[final_time]["v_ycm"] = v_ycm
        self.bodies[final_time]["v_zcm"] = v_zcm
        self.bodies[final_time]["a_xcm"] = a_xcm
        self.bodies[final_time]["a_ycm"] = a_ycm
        self.bodies[final_time]["a_zcm"] = a_zcm
        del self.bodies[self.prev_time]

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
