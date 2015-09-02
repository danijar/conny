from conny.core import Node
from conny.function import Constant, Sum, Product, Sigmoid
from conny.utility import pairwise


def fully_connected_network(input_size, hidden_size, output_size,
        input_func=Sigmoid, hidden_func=Sigmoid, output_func=Sigmoid):
    # Define layers
    input_ = Node(Node(input_func) * input_size, input=True)
    hidden = Node(Node(hidden_func, inout=True) * hidden_size)
    output = Node(Node(output_func) * output_size, output=True)
    # Combine and connect them
    network = Node(input_, hidden, output)
    network.children.connect(network.children)
    return network


def feed_forward_network(input_size, hidden_size, output_size, hidden_count,
        input_func=Constant, hidden_func=Sigmoid, output_func=Sigmoid):
    # Define layers
    input_ = Node(Node(input_func) * input_size, input=True)
    hidden = Node(Node(hidden_func, inout=True) * hidden_size)
    output = Node(Node(output_func) * output_size, output=True)
    # Combine and connect them
    network = Node(input_, hidden * hidden_count, output)
    for last, current in pairwise(network):
        last.connect(current)
    return network


def lstm_network(input_size, hidden_size, output_size, hidden_count,
        input_func=Constant, output_func=Sigmoid):
    # Define layers
    input_ = Node(Node(input_func) * input_size, input=True)
    hidden = Node(Node(lstm_unit(), inout=True) * hidden_size)
    output = Node(Node(output_func) * output_size, output=True)
    # Combine and connect them
    network = Node(input_, hidden * hidden_count, output)
    for last, current in pairwise(network):
        last.connect(current)
    return network


def lstm_unit():
    # Define neurons
    read = Node(Product, input=True)
    remember = Node(Product, input=True)
    internal = Node(Sum)
    output = Node(Product, inout=True)
    # Combine and connect them
    read.connect(internal)
    remember.connect(internal)
    internal.connect(output, remember)
    return Node(read, remember, internal, output)
from function import Constant, Sum, Product, Sigmoid
from utility import pairwise


def fully_connected_network(input_size, hidden_size, output_size,
        input_func=Sigmoid, hidden_func=Sigmoid, output_func=Sigmoid):
    # Define layers
    input_ = Node(Node(input_func) * input_size, input=True)
    hidden = Node(Node(hidden_func, inout=True) * hidden_size)
    output = Node(Node(output_func) * output_size, output=True)
    # Combine and connect them
    network = Node(input_, hidden, output)
    network.children.connect(network.children)
    return network


def feed_forward_network(input_size, hidden_size, output_size, hidden_count,
        input_func=Constant, hidden_func=Sigmoid, output_func=Sigmoid):
    # Define layers
    input_ = Node(Node(input_func) * input_size, input=True)
    hidden = Node(Node(hidden_func, inout=True) * hidden_size)
    output = Node(Node(output_func) * output_size, output=True)
    # Combine and connect them
    network = Node(input_, hidden * hidden_count, output)
    for last, current in pairwise(network):
        last.connect(current)
    return network


def lstm_network(input_size, hidden_size, output_size, hidden_count,
        input_func=Constant, output_func=Sigmoid):
    # Define layers
    input_ = Node(Node(input_func) * input_size, input=True)
    hidden = Node(Node(lstm_unit(), inout=True) * hidden_size)
    output = Node(Node(output_func) * output_size, output=True)
    # Combine and connect them
    network = Node(input_, hidden * hidden_count, output)
    for last, current in pairwise(network):
        last.connect(current)
    return network


def lstm_unit():
    # Define neurons
    read = Node(Product, input=True)
    remember = Node(Product, input=True)
    internal = Node(Sum)
    output = Node(Product, inout=True)
    # Combine and connect them
    read.connect(internal)
    remember.connect(internal)
    internal.connect(output, remember)
    return Node(read, remember, internal, output)
