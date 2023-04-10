from reckit.cython import pyx_randint_choice

def randint_choice(high, size=1, replace=True, p=None, exclusion=None):
    """Sample random integers from [0, high).

    :param high:
    :param size:
    :param replace:
    :param p:
    :param exclusion:
    :return:
    """
    return pyx_randint_choice(high, size, replace, p, exclusion)