import pytest

from conny.core import Node
from test.fixtures import layers


class TestNode:

    def test_from_empty_child_list(self):
        with pytest.raises(ValueError):
            Node([])

    def test_leaves_all_outgoing(self, layers):
        leaves = layers.get_leaves()
        assert all(hasattr(node, 'outgoing') for node in leaves)

