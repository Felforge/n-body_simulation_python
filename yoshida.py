def yoshida_pos_step_1_and_4(pos, v, time_step):
    """
    First step of Yoshida integration for position
    Velocity is assumed to be just X, Y, or Z
    """
    c_1 = 0.5 / (2.0 - 2.0**(1.0/3.0))
    return pos + c_1 * v * time_step

def yoshida_v_step_1_and_3(v, a, time_step):
    """
    First step of Yoshida integration for velocity
    Velocity and Acceleration are assumed to be just X, Y, or Z
    """
    w_1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    return v + w_1 * a * time_step

def yoshida_pos_step_2_and_3(pos, v, time_step):
    """
    Second step of Yoshida integration for position
    Velocity is assumed to be just X, Y, or Z
    """
    w_1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    w_2 = 1.0 - 2.0 * w_1
    c_2 = (w_1 + w_2) / 2.0
    return pos + c_2 * v * time_step

def yoshida_v_step_2(v, a, time_step):
    """
    Second step of Yoshida integration for velocity
    Velocity and Acceleration are assumed to be just X, Y, or Z
    """
    w_2 = 1.0 - (2.0 / (2.0 - 2.0**(1.0/3.0)))
    return v + w_2 * a * time_step
