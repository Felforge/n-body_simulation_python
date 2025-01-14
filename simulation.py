# Create Bodies List Function
# Bootstrap Simulation Function

def create_body(mass, radius, x, y, z, v_x, v_y, v_z):
    """
    mass is in kilograms
    x, y, z is relative to center of simulation in meters
    v_x, v_y, v_z is in meters/second
    """
    body = {}
    body["m"] = mass
    body["r"] = radius
    body["x"] = x
    body["y"] = y
    body["z"] = z
    body["v_x"] = v_x
    body["v_y"] = v_y
    body["v_z"] = v_z
    return body
