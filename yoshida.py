import numpy as np

def yoshida_pos_step_1_and_4(pos, v, time_step):
    """
    First step of Yoshida integration for position
    Velocity is assumed to be just X, Y, or Z
    """
    c_1 = (-1 / 2) * np.cbrt(2) / (2 - np.cbrt(2))
    return pos + c_1 * v * time_step

def yoshida_v_step_1_and_3(v, a, time_step):
    """
    First step of Yoshida integration for velocity
    Velocity and Acceleration are assumed to be just X, Y, or Z
    """
    d_1 = -1 * np.cbrt(2) / (2 - np.cbrt(2))
    return v + d_1 * a * time_step

def yoshida_pos_step_2_and_3(pos, v, time_step):
    """
    Second step of Yoshida integration for position
    Velocity is assumed to be just X, Y, or Z
    """
    w_0 = -1 * np.cbrt(2) / (2 - np.cbrt(2))
    w_1 = 1 / (2 - np.cbrt(2))
    c_2 = (w_0 + w_1) / 2
    return pos + c_2 * v * time_step

def yoshida_v_step_2(v, a, time_step):
    """
    Second step of Yoshida integration for velocity
    Velocity and Acceleration are assumed to be just X, Y, or Z
    """
    d_2 = 1 / (2 - np.cbrt(2))
    return v + d_2 * a * time_step
