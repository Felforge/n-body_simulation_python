import numpy as np

def get_distance(x1, y1, z1, x2, y2, z2):
    """
    Get distance between points
    """
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

def get_cm(bodies, mode="pos"):
    """
    Get center of mass components
    mode = "pos" for position
    mode = "v" for velocity
    mdde = "a" for acceleration
    """
    total_mass = 0
    result = [0, 0, 0]
    for body in bodies:
        total_mass += body["m"]
        for i, dim in enumerate(["x", "y", "z"]):
            if mode == "pos":
                result[i] += body["m"] * body[dim]
            elif mode == "v":
                result[i] += body["m"] * body[f"v_{dim}"]
            elif mode == "a":
                result[i] += body["m"] * body[f"a_{dim}"]
    return [num / total_mass for num in result]

def approximate_next(ti, tf, xi, yi, zi, xf, yf, zf):
    """
    Approximate next derivative using slope formula
    """
    delta_t = tf - ti
    return (xf - xi) / delta_t, (yf - yi) / delta_t, (zf - zi) / delta_t

def divide_matrix(matrix_1, matrix_2):
    """
    Divide matrix and return max ratio of corresponding terms
    Both matricies must be same dimensions
    Needed for Q_ddot / Q_3dot
    """
    max_ratio = 0
    for i, row in enumerate(matrix_1):
        for j, num in enumerate(row):
            if matrix_2[i][j] != 0:
                if max_ratio == 0:
                    max_ratio = num / matrix_2[i][j]
                else:
                    ratio = num / matrix_2[i][j]
                    if abs(ratio) > max_ratio:
                        max_ratio = ratio
    return max_ratio
