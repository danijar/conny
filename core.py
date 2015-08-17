import copy
import itertools


class Function:

    def compute(self, inputs):
        raise NotImplemented

    def derive(self, activation, input_):
        raise NotImplemented

    def gradient(self, num_inputs, activation):
        return list(self.derive(i) for i in range(num_inputs))


class Node(list):

    def __init__(self, *args, **kwargs):
        flat_args = list(itertools.chain(*args))
        # Construct from activation function
        if len(args) == 1 and isinstance(args[0], Function):
            self.function = args[0]
            self.outgoing = []
        # Construct from one or more lists of children
        elif all(isinstance(flat_args, Node) for arg in args):
            self += copy.deepcopy(args)
        else:
            raise ValueError('No matching constructor')
        # Whether to consider this node if someone connects to the parent.
        self.input = kwargs.get('input', False) or kwargs.get('inout', False)
        # Whether to consider this node if the parent connects to someone.
        self.output = kwargs.get('output', False) or kwargs.get('inout', False)

    def first(self):
        return self[0]

    def last(self):
        return self[-1]

    def __mul__(self, repeat):
        return list(copy.deepcopy(self) for _ in range(repeat))

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

    def get_leafs(self):
        if len(self):
            for child in self:
                yield from child.get_leafs()
        else:
            yield self

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
