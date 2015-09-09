import inspect
import itertools


def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def implements(argument, interface):
    """
    Whether the argument is a subclass of the second type.
    """
    if not inspect.isclass(argument):
        return False
    return issubclass(argument, interface)

def flatten(iterable, keep=lambda x: False):
    if keep(iterable):
        yield iterable
        return
    for item in iterable:
        try:
            yield from flatten(item, keep)
        except ValueError:
            yield item


def connect_layers(node):
    """
    Connect the children of the passed node pairwise and set the first one to
    input and the last one to output.
    """
    for last, current in pairwise(node):
        last.connect(current)
    node[0].input = True
    node[-1].output = True
