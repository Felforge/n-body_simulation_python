def get_Q_ddot(current_state):
    """
    Get second derivative of Quadrupole moment with respect to time
    Returns 3x3 matrix
    """
    bodies = current_state["body_list"]
    matrix = []
    for dim1 in ["x", "y", "z"]:
        row = []
        for dim2 in ["x", "y", "z"]:
            sum_total = 0
            for body in bodies:
                current_sum = 0
                if dim1 == dim2:
                    current_sum += 3 * body[dim1] * body[f"a_{dim1}"]
                    current_sum += 3 * (body[f"v_{dim1}"]**2)
                    for dim in ["x", "y", "z"]:
                        current_sum -= (body[f"v_{dim}"] - current_state[f"v_{dim}cm"])**2
                        current_sum -= (body[dim] - current_state[f"{dim}_cm"]) * (body[f"a_{dim}"] - current_state[f"a_{dim}cm"])
                    current_sum *= 2
                else:
                    current_sum += 3 * body[f"a_{dim1}"] * body[dim2]
                    current_sum += 3 * body[f"a_{dim2}"] * body[dim1]
                    current_sum += 6 * body[f"v_{dim1}"] * body[f"v_{dim2}"]
                sum_total += current_sum * body["m"]
            row.append(sum_total)
        matrix.append(row)
    return matrix

def get_Q_3dot(current_state):
    """
    Get third derivative of Quadrupole moment with respect to time
    Returns 3x3 matrix
    """
    bodies = current_state["body_list"]
    matrix = []
    for dim1 in ["x", "y", "z"]:
        row = []
        for dim2 in ["x", "y", "z"]:
            sum_total = 0
            for body in bodies:
                current_sum = 0
                if dim1 == dim2:
                    current_sum += 9 * body[f"v_{dim1}"] * body[f"a_{dim1}"]
                    current_sum += 3 * body[dim1] * body[f"j_{dim1}"]
                    for dim in ["x", "y", "z"]:
                        current_sum -= 3 * (body[f"v_{dim}"] - current_state[f"v_{dim}cm"]) * (body[f"a_{dim}"] - current_state[f"a_{dim}cm"])
                        current_sum -= (body[dim] - current_state[f"{dim}_cm"]) * (body[f"j_{dim}"] - current_state[f"j_{dim}cm"])
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

def get_Q_5dot(current_state):
    """
    Get fifth derivative of Quadrupole moment with respect to time
    Returns 3x3 matrix
    """
    bodies = current_state["body_list"]
    matrix = []
    for dim1 in ["x", "y", "z"]:
        row = []
        for dim2 in ["x", "y", "z"]:
            sum_total = 0
            for body in bodies:
                current_sum = 0
                if dim1 == dim2:
                    current_sum += 30 * body[f"a_{dim1}"] * body[f"j_{dim1}"]
                    current_sum += 15 * body[f"v_{dim1}"] * body[f"s_{dim1}"]
                    current_sum += 3 * body[dim1] * body[f"c_{dim1}"]
                    for dim in ["x", "y", "z"]:
                        current_sum -= 10 * (body[f"a_{dim}"] - current_state[f"a_{dim}cm"]) * (body[f"j_{dim}"] - current_state[f"j_{dim}cm"])
                        current_sum -= 5 * (body[f"v_{dim}"] - current_state[f"v_{dim}cm"]) * (body[f"s_{dim}"] - current_state[f"s_{dim}cm"])
                        current_sum -= (body[dim] - current_state[f"{dim}_cm"]) * (body[f"c_{dim}"] - current_state[f"c_{dim}cm"])
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
