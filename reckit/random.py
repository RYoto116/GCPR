from reckit.cython import pyx_randint_choice
from collections import Iterable

def randint_choice(high, size=1, replace=True, p=None, exclusion=None):
    """Sample random integers from [0, high).

    :param high:
    :param size:
    :param replace:
    :param p:
    :param exclusion:
    :return:
    """
    index = pyx_randint_choice(high, size, replace, p, exclusion)
    if isinstance(index, Iterable):
        return list(index)
    return [index]
