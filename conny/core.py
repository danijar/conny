import copy
import itertools
import inspect

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix

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
        length = len(self)
        return '<Node children={length}>'.format(**locals())

    def __mul__(self, repeat):
        return list(copy.deepcopy(self) for _ in range(repeat))

    def first(self):
        return self[0]

    def last(self):
        return self[-1]

    def connect(self, target, strategy='full'):
        """
        Connections exist between leaf nodes only. Calling connect() on a
        parent node creates connections between all leaf nodes for which all
        parents and themselves are marked as output and input respectively.
        """
        outputs = self._filter_leaves('output', True)
        inputs = target._filter_leaves('input', True)
        function = '_connect_' + strategy
        if hasattr(self, function):
            getattr(self, function)(outputs, inputs)
        else:
            raise NotImplemented

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
        if len(args) and isinstance(args[0], list):
            args = sum(args, [])
        if all(isinstance(arg, Node) for arg in args):
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

    def _connect_full(self, outputs, inputs):
        for output in outputs:
            output.outgoing += inputs


class Network:

    def __init__(self, *args, **kwargs):
        self.nodes = list(Node(*args).get_leaves())
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
        self.weights = coo_matrix(shape, dtype=np.float32)
        # Sparse matrix of derivatives with respect to the weights
        self.gradient = coo_matrix(shape, dtype=np.float32)
        # Initialize used weights. All other weights are zero in sparse matrix
        # representation and thus don't affect products of the activation
        # vector and the weight matrix.
        for i, source in enumerate(self.nodes):
            for target in source.outgoing:
                j = self.nodes.index(target)
                self.weights[i, j] = scale * np.random.normal()
                # TODO: Zero is correct but doesn't work with sparse matrix
                self.gradient[i, j] = 0.0000001
        # Compress matrices into efficient formats
        self.weights = csc_matrix(self.weights)

    def _init_functions(self):
        # Ordered list of activation functions used in this network
        self.functions = list(set(node.function for node in self.nodes))
        assert len(self.functions) < 256, 'Too many activation functions'
        for index, node in enumerate(self.nodes):
            self.types[index] = self.functions.index(node.function)

        self.gradient = csc_matrix(self.gradient)
