from basic_physics import get_distance

G = 6.6743e-11

class Octree:
    """
    Octree for Barnes Hut algorithm
    Bounds are in a [min, max] format
    """
    def __init__(self, bodies, x_bound, y_bound, z_bound):
        self.bodies = bodies
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.z_bound = z_bound
        self.mass = 0
        self.cm = [0, 0, 0]
        if len(bodies) > 1:
            self.tree = self.create_children()
        elif len(bodies) == 1:
            self.mass = bodies[0]["m"]
            self.cm = [bodies[0]["x"], bodies[0]["y"], bodies[0]["z"]]

    def get_subidivision_bounds(self, mode="ppp"):
        """
        Get bounds for subidivision
        Mode Ex: ppp = +X +Y +Z
        pmp = +X -Y +Z
        """
        bounds = [self.x_bound, self.y_bound, self.z_bound]
        for i, sign in enumerate(mode):
            if sign == "p":
                bounds[i][0] = sum(bounds[i]) / 2
            elif sign == "m":
                bounds[i][1] = sum(bounds[i]) / 2
            else:
                raise BarnesHutError("Invalid Mode Format")
        return bounds[0], bounds[1], bounds[2]

    def subdivide_bodies(self, bodies, *bounds):
        """
        Return bodies within given quadrant
        """
        return_bodies = []
        for body in bodies:
            for i, dim in enumerate(["x", "y", "z"]):
                if body[dim] < bounds[i][0] or body[dim] > bounds[i][1]:
                    break
            else:
                return_bodies.append(body)
        return return_bodies

    def create_children(self):
        """
        Divide area into 8 sub-areas
        """
        tree = []
        for mode in ["ppp", "ppm", "pmp", "mpp", "mmp", "mpm", "pmm", "mmm"]:
            x_bound, y_bound, z_bound = self.get_subidivision_bounds(mode)
            tree.append(Octree(self.subdivide_bodies(self.bodies, x_bound, y_bound, z_bound),
                                x_bound, y_bound, z_bound))
        return tree

    def get_bodies(self):
        """
        Return bodies list
        """
        return self.bodies
    
    def get_force_components(self):
        """
        Return components needed for force
        """
        return self.mass, self.cm[0], self.cm[1], self.cm[2]

    def get_force(self, body):
        """
        Get force on given body
        """
        if body not in self.bodies:
            x, y, z = self.cm
            r = get_distance(body["x"], body["y"], body["z"], x, y, z)
            f = (G * body["m"] * self.mass) / (r**2)
            f_x = f * (x - body["x"]) / r
            f_y = f * (y - body["y"]) / r
            f_z = f * (z - body["z"]) / r
            return f_x, f_y, f_z
        else:
            if len(self.bodies) <= 1:
                return 0, 0, 0
            for child in self.tree:
                f_x = 0
                f_y = 0
                f_z = 0
                # Add forces from recursion and return

    # def add_body(self, body):
    #     """
    #     Add body to node
    #     """
    #     self.mass = body["m"]
    #     self.x_cm = body["x"]
    #     self.y_cm = body["y"]
    #     self.z_cm = body["z"]
    # def add_bodies(self, bodies):
    #     """
    #     Handle multiple bodies
    #     """
        
    # def get_bound(self):
    #     """
    #     Get bounds for subdivision
    #     """
    #     coord_bounds = []
    #     mid_x = (self.x_bound[0] + self.x_bound[1]) // 2
    #     mid_y = (self.y_bound[0] + self.y_bound[1]) // 2
    #     mid_z = (self.z_bound[0] + self.z_bound[1]) // 2
    #     x_1 = [self.z_bound[0], mid_x]
    #     x_2 = [mid_x, self.z_bound[1]]
    #     y_1 = [self.z_bound[0], mid_x]
    #     y_2 = [mid_x, self.z_bound[1]]
    #     y_1 = [self.z_bound[0], mid_x]
    #     y_2 = [mid_x, self.z_bound[1]]
    #     for x in self.x_bound:
    #         for y in self.y_bound:
    #             for z in self.z_bound:
    #                 if x < mid_x:

class BarnesHutError(Exception):
    """
    Error flag for Barnes-Hut algorithm
    """
    def __init__(self, message):
        super().__init__(message)
