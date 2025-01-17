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

        if len(bodies) == 1:
            self.mass = bodies[0]["m"]
            self.cm = [bodies[0]["x"], bodies[0]["y"], bodies[0]["z"]]
            self.tree = []
        elif len(bodies) > 1:
            self.tree = self.create_children()

    # def get_subidivision_bounds(self, mode="ppp"):
    #     """
    #     Get bounds for subidivision
    #     Mode Ex: ppp = +X +Y +Z
    #     pmp = +X -Y +Z
    #     """
    #     bounds = [self.x_bound, self.y_bound, self.z_bound]
    #     for i, sign in enumerate(mode):
    #         if sign == "p":
    #             bounds[i][0] = sum(bounds[i]) / 2
    #         elif sign == "m":
    #             bounds[i][1] = sum(bounds[i]) / 2
    #         else:
    #             raise BarnesHutError("Invalid Mode Format")
    #     return bounds[0], bounds[1], bounds[2]
    
    def get_subidivision_bounds(self, mode="ppp"):
        """
        Get bounds for subidivision
        Mode Ex: ppp = +X +Y +Z
        pmp = +X -Y +Z
        """
        new_bounds = [
            self.x_bound.copy(),
            self.y_bound.copy(),
            self.z_bound.copy()
        ]
        for i, sign in enumerate(mode):
            mid = (new_bounds[i][0] + new_bounds[i][1]) / 2  # Calculate true midpoint
            if sign == "p":
                new_bounds[i][0] = mid  # Take upper half
            elif sign == "m":
                new_bounds[i][1] = mid  # Take lower half
        return new_bounds[0], new_bounds[1], new_bounds[2]

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

    # def create_children(self):
    #     """
    #     Divide area into 8 sub-areas
    #     """
    #     total_mass = 0
    #     mass_pos = [0, 0, 0]

    #     children = []
    #     for mode in ["ppp", "ppm", "pmp", "mpp", "mmp", "mpm", "pmm", "mmm"]:
    #         x_bound, y_bound, z_bound = self.get_subidivision_bounds(mode)
    #         bodies_in_octant = self.subdivide_bodies(self.bodies, x_bound, y_bound, z_bound)

    #         for child in bodies_in_octant:
    #             total_mass += child["m"]
    #             for i, pos in enumerate(["x", "y", "z"]):
    #                 mass_pos[i] += child["m"] * child[pos]

    #         children.append([bodies_in_octant, x_bound, y_bound, z_bound])

    #     self.mass = total_mass
    #     if total_mass > 0:
    #         self.cm = [pos / total_mass for pos in mass_pos]

    #     return [Octree(child[0], child[1], child[2], child[3]) for child in children]
    
    def create_children(self):
        """
        Divide area into 8 sub-areas
        """
        total_mass = 0
        mass_pos = [0, 0, 0]
        children = []
        
        print(f"Creating children for {len(self.bodies)} bodies")
        for mode in ["ppp", "ppm", "pmp", "mpp", "mmp", "mpm", "pmm", "mmm"]:
            x_bound, y_bound, z_bound = self.get_subidivision_bounds(mode)
            bodies_in_octant = self.subdivide_bodies(self.bodies, x_bound, y_bound, z_bound)
            print(f"Mode {mode}: Found {len(bodies_in_octant)} bodies")
            
            for child in bodies_in_octant:
                total_mass += child["m"]
                for i, pos in enumerate(["x", "y", "z"]):
                    mass_pos[i] += child["m"] * child[pos]
                    
            print(f"Current total_mass: {total_mass}")
            children.append([bodies_in_octant, x_bound, y_bound, z_bound])
        
        self.mass = total_mass
        if total_mass > 0:
            self.cm = [pos / total_mass for pos in mass_pos]
        print(f"Final mass for this node: {self.mass}")
        
        return [Octree(child[0], child[1], child[2], child[3]) for child in children]

    def get_force_components(self):
        """
        Return components needed for force
        """
        return self.mass, self.cm[0], self.cm[1], self.cm[2]

    def get_force(self, body, theta=0.5):
        """
        Get force on given body
        """
        if self.mass == 0 or len(self.bodies) == 1 and self.bodies[0] == body:
            return 0, 0, 0

        if body not in self.bodies:
            x, y, z = self.cm
            d = get_distance(body["x"], body["y"], body["z"], x, y, z)
            if d == 0:
                return 0, 0, 0

            s = self.x_bound[1] - self.x_bound[0]
            if (s/d < theta) or len(self.bodies) == 1:
                f = (G * body["m"] * self.mass) / (d**2)
                f_x = f * (x - body["x"]) / d
                f_y = f * (y - body["y"]) / d
                f_z = f * (z - body["z"]) / d
                return f_x, f_y, f_z

        f_x = f_y = f_z = 0
        for child in self.tree:
            if child.mass > 0:
                cf_x, cf_y, cf_z = child.get_force(body, theta)
                f_x += cf_x
                f_y += cf_y
                f_z += cf_z
        return f_x, f_y, f_z

class BarnesHutError(Exception):
    """
    Error flag for Barnes-Hut algorithm
    """
    def __init__(self, message):
        super().__init__(message)
