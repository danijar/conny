import pytest

from conny.core import Node, Network
from conny.function import Constant, Sum, Product, Sigmoid


def test_network_count_single():
    amount = 10

    node = Node(Constant, inout=True)
    layer = Node(node * amount)
    leaves = layer.get_leaves()
    assert all(hasattr(node, 'outgoing') for node in leaves)

    network = Network(layer)
    assert len(network.current) == amount

