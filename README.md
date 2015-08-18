[![Code Climate][1]][2]
[![Test Coverage][3]][4]

[1]: https://codeclimate.com/github/danijar/conny/badges/gpa.svg
[2]: https://codeclimate.com/github/danijar/conny
[3]: https://codeclimate.com/github/danijar/conny/badges/coverage.svg
[4]: https://codeclimate.com/github/danijar/conny/coverage

Conny
=====

Conny is a neural network library. Instead of organizing networks into layers,
it allows for arbitrary connections including recurrent ones. This makes it a
good tool to experiment with new topologies.

Data Layout
-----------

All neurons are stores as a one-dimensional vector. This works since in
backpropagation through time, gradients are only based on the previous
activations. This allows for arbitrary connections between neurons. The network
and its state are still stored in a compact way allowing for efficient algebra
routines on the CPU and GPU.

|       Variable      |   Type  | Dimensions | Storage |
| ------------------- | :-----: | :--------: | :-----: |
| Activation Function |   int8  |     N      |  Dense  |
| Current activation  | float32 |     N      |  Dense  |
| Previous activation | float32 |     N      |  Dense  |
| Weights             | float32 |   N x N    |  Sparse |
| Gradient            | float32 |   N x N    |  Sparse |

N refers to the total number of neurons.

The activation function is stored as an enumeration value.

Instructions
------------

```
virtualenv .
source bin/activate
pip install -U pip
pip install -r requirements.txt
```
