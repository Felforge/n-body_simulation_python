import numpy as np

def get_distance(x1, y1, z1, x2, y2, z2):
    """
    Get distance between points
    """
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
