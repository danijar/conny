import copy
import itertools
import inspect

import numpy as np
from scipy.sparse import lil_matrix, csc_matrix

from conny import utility


class Function:

    def compute(self, inputs):
        raise NotImplemented

    def derive(self, activation, input_):
        raise NotImplemented

    def gradient(self, num_inputs, activation):
        return list(self.derive(i) for i in range(num_inputs))


class Node(list):

    def __init__(self, *args, **kwargs):
        self._parse_args(args)
        self._parse_kwargs(kwargs)

    def __repr__(self):
        attrs = []
        if len(self):
            attrs.append('children=' + str(len(self)))
        else:
            attrs.append('outgoing=' + str(len(self.outgoing)))
        if self.input:
            attrs.append('input')
        if self.output:
            attrs.append('output')
        return '<Node {}>'.format(' '.join(attrs))

    def __eq__(self, other):
        return id(self) == id(other)

    def __mul__(self, repeat):
        return list(copy.deepcopy(self) for _ in range(repeat))

    def connect(self, target, strategy='full'):
        """
        Connections exist between leaf nodes only. Calling connect() on a
        parent node creates connections between all leaf nodes to which a path
        of output or input nodes exist respectively.
        """
        outputs = self._filter_leaves_or_self('output', True)
        inputs = target._filter_leaves_or_self('input', True)
        function = '_connect_' + strategy
        if not hasattr(self, function):
            raise NotImplemented
        pairs = getattr(self, function)(outputs, inputs)
        for left, right in pairs:
            left.outgoing.append(right)
        return pairs

    def get_leaves(self):
        if len(self):
            for child in self:
                yield from child.get_leaves()
        else:
            yield self

    def _parse_args(self, args):
        # Construct from activation function
        if len(args) == 1 and utility.implements(args[0], Function):
            self.function = args[0]
            self.outgoing = []
            return
        # Construct from one or more lists of children
        args = list(utility.flatten(args, lambda x: isinstance(x, Node)))
        if args and all(isinstance(arg, Node) for arg in args):
            self += copy.deepcopy(args)
            return
        raise ValueError('No matching constructor')

    def _parse_kwargs(self, kwargs):
        inout = kwargs.get('inout', False)
        # Whether to consider this node if someone connects to the parent.
        self.input = kwargs.get('input', False) or inout
        # Whether to consider this node if the parent connects to someone.
        self.output = kwargs.get('output', False) or inout

    def _filter_leaves(self, attibute, value):
        if len(self):
            for child in self:
                if getattr(child, attibute) == value:
                    yield from child._filter_leaves(attibute, value)
        elif getattr(self, attibute) == value:
            yield self

    def _filter_leaves_or_self(self, attribute, value):
        if len(self):
            return list(self._filter_leaves(attribute, value))
        else:
            return [self]

    def _connect_full(self, outputs, inputs):
        yield from itertools.product(outputs, inputs)


class Network:

    def __init__(self, root, **kwargs):
        self.root = root
        self.nodes = list(self.root.get_leaves())
        self._init_nodes()
        self._init_edges()
        self._init_functions()

    def _init_nodes(self):
        # TODO: Take per neuron bias into account
        shape = len(self.nodes)
        # Index of activation function inside self.functions
        self.types = np.zeros(shape, dtype=np.int8)
        # Current activation vector of the neurons
        self.current = np.zeros(shape, dtype=np.float32)
        # Previous activation vector of the neurons
        self.previous = np.zeros(shape, dtype=np.float32)

    def _init_edges(self, scale=0.1):
        shape = (len(self.nodes), len(self.nodes))
        # Sparse matrix of weights between the neurons
        self.weights = lil_matrix(shape, dtype=np.float32)
        # Sparse matrix of derivatives with respect to the weights
        self.gradient = lil_matrix(shape, dtype=np.float32)
        # Initialize used weights. All other weights are zero in sparse matrix
        # representation and thus don't affect products of the activation
        # vector and the weight matrix.
        self.edges = []
        for i, source in enumerate(self.nodes):
            for target in source.outgoing:
                j = self.nodes.index(target)
                self.edges.append((i, j))
                self.weights[i, j] = scale * np.random.normal()
                self.gradient[i, j] = 0
        # Compress matrices into efficient formats
        self.weights = csc_matrix(self.weights)
        self.gradient = csc_matrix(self.gradient)

    def _init_functions(self):
        # Ordered list of activation functions used in this network
        self.functions = list(set(node.function for node in self.nodes))
        assert len(self.functions) < 256, 'Too many activation functions'
        for index, node in enumerate(self.nodes):
            self.types[index] = self.functions.index(node.function)

        self.gradient = csc_matrix(self.gradient)
