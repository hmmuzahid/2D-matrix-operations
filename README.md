# 2D Matrix Operations with NumPy Behavior

A lightweight 2D matrix class written in Python.

## Features
- NumPy-like broadcasting
- Matrix multiplication (`@`)
- Reshape and transpose
- Mean and standard deviation
- Boolean masking
- `where`
- Copy, ravel, flatten
- In-place operations
- Negative indexing and slicing

## Example

```python
from mat_opr_python import array

a = array([[1, 2, 3],
           [4, 5, 6]])

b = array([[10],
           [20]])

print(a + b)