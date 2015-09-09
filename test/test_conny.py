import pytest

from conny.core import Node, Network
from conny.function import Constant, Sum, Product, Sigmoid
from conny.utility import pairwise, connect_layers


@pytest.fixture(params=[1, 2, 7, 13])
def layers(request):
    node = Node(Constant, inout=True)
    layer = Node(node * request.param)
    layers = Node(layer * request.param)
    layers.amount = request.param * request.param
    return layers


class TestNode:

    def test_from_empty_child_list(self):
        with pytest.raises(ValueError):
            Node([])

    def test_leaves_all_outgoing(self, layers):
        leaves = layers.get_leaves()
        assert all(hasattr(node, 'outgoing') for node in leaves)


class TestNetwork:

    def test_count(self, layers):
        network = Network(layers)
        assert len(network.current) == layers.amount
        assert len(network.previous) == layers.amount
        assert len(network.types) == layers.amount
        assert network.weights.shape == (layers.amount, layers.amount)
        assert network.gradient.shape == (layers.amount, layers.amount)

    def test_connections_1(self):
        # ->(0)->(2)->(4)->
        #      \/^  \/^
        #      /\v  /\v
        # ->(1)->(3)->(5)->
        node = Node(Constant, inout=True)
        layer = Node(node * 2)
        layers = Node(layer * 3, inout=True)
        connect_layers(layers)
        network = Network(layers)

        assert len(network.current) == 6
        assert network.weights.getnnz() == 8
        assert network.edges == [
            (0, 2), (0, 3), (1, 2), (1, 3),
            (2, 4), (2, 5), (3, 4), (3, 5)]

    def test_connections_2(self):
        # -> ((0)->(1)->(2)) -> ((3)->(4)->(5)) ->
        input_ = Node(Constant, input=True)
        hidden = Node(Constant)
        output = Node(Constant, output=True)
        input_.connect(hidden)
        hidden.connect(output)

        cell = Node(input_, hidden, output, inout=True)
        cells = Node(cell * 2)
        cells[0].input = True
        cells[1].output = True
        cells[0].connect(cells[1])
        network = Network(cells)

        assert len(network.current) == 6
        assert network.weights.getnnz() == 5
        assert network.edges == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
