Data Layout
-----------

Neural networks are represented in the following way. This decision is
motivated by allowing arbitrary topologies and providing high performance.

|       Variable      |   Type  | Dimensions | Storage |
| ------------------- | :-----: | :--------: | :-----: |
| Activation Function |   int8  |     N      |  Dense  |
| Current activation  | float32 |     N      |  Dense  |
| Previous activation | float32 |     N      |  Dense  |
| Weights             | float32 |   N x N    |  Sparse |
| Gradient            | float32 |   N x N    |  Sparse |

`N` refers to the number of neurons. The activation function is stored as an
enumeration value. Sparse matrices are constructed in coordinate format and
then converted to compressed sparse column format for higher performance.

Instructions
------------

```
virtualenv .
source bin/activate
pip install -U pip
pip install -r requirements.txt
```

