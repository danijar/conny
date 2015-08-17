from itertools import izip


def pairwise(iterable):
    a = iter(iterable)
    return izip(a, a)
