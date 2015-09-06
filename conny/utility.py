import itertools
import inspect


def pairwise(iterable):
    a = iter(iterable)
    return itertools.izip(a, a)


def implements(argument, interface):
    """
    Whether the argument is a subclass of the second type.
    """
    if not inspect.isclass(argument):
        return False
    return issubclass(argument, interface)
