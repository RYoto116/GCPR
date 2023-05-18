from reckit.cython.random_choice import pyx_randint_choice
from collections import Iterable

def randint_choice(high, size=1, replace=True, p=None, exclusion=None):
    """Sample random integers from [0, high).
    """
    index = pyx_randint_choice(high, size, replace, p, exclusion)
    if isinstance(index, Iterable):
        return list(index)
    return [index]
