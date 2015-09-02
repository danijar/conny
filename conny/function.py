import numpy as np

from conny.core import Function


class Constant(Function):

    def compute(self, inputs):
        assert len(inputs) == 1
        return inputs[0]

    def derive(self, activation, input_):
        return 0


class Sum(Function):

    def compute(self, inputs):
        return np.sum(inputs)

    def derive(self, activation, input_):
        return 1


class Product(Function):

    def compute(self, inputs):
        return np.product(inputs)

    def derive(self, activation, input_):
        return activation / input_


class Sigmoid(Function):

    def compute(self, inputs):
        return 1 / (1 + np.exp(-np.sum(inputs)))

    def derive(self, activation, input_):
        value = self.compute(input_)
        return value * (1 - value)
