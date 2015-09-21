import pytest

from conny.core import Node
from conny.function import Constant


@pytest.fixture(params=[1, 2, 7, 13])
def layers(request):
    node = Node(Constant, inout=True)
    layer = Node(node * request.param)
    layers = Node(layer * request.param)
    layers.amount = request.param * request.param
    return layers
