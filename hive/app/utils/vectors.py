from typing import List, Dict


def new_steady_state_vector(labels_list: List[str], uptime_dict: Dict[str, float]) -> List[float]:
    """
    Generates a row vector whose entries summation is one.
    :param List[str] labels_list: contains the name of all peers, this param is required to ensure order is kept
    :param Dict[str, float] uptime_dict: names of all peers in the system and their uptimes
    :returns List[float]: desired distribution vector
    """
    stochastic_vector: List[float] = list()
    for i in range(len(labels_list)):
        stochastic_vector.append(uptime_dict[labels_list[i]])

    uptimes_sum = sum(stochastic_vector)
    return [uptime/uptimes_sum for uptime in stochastic_vector]
