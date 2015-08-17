Data Layout
-----------

Neural networks are not separated into layers internally. Instead, all neurons
are stores as a one-dimensional vector. This works since in backpropagation
through time, gradients are only based on the previous activations. This allows
for arbitrary topologies while promising high-end performance.

|       Variable      |   Type  | Dimensions | Storage |
| ------------------- | :-----: | :--------: | :-----: |
| Activation Function |   int8  |     N      |  Dense  |
| Current activation  | float32 |     N      |  Dense  |
| Previous activation | float32 |     N      |  Dense  |
| Weights             | float32 |   N x N    |  Sparse |
| Gradient            | float32 |   N x N    |  Sparse |

`N` refers to the total number of neurons.

The activation function is stored as an enumeration value.

Instructions
------------

```
virtualenv .
source bin/activate
pip install -U pip
pip install -r requirements.txt
```
