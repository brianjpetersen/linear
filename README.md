# linear: LINear Expressions for Algebraic Relief

The package linear makes it easy and quick to develop sparse matrices from linear expressions
without resorting to laborious indexing and index mapping.  Just express your equations in
natural syntax and generate sparse matrices as output.

# Examples

```python
>>> a = Variable('a')
>>> b = Variable('b')
>>> c = Variable('c')
>>> expressions = [2*a + 1, a + b + 2, (3*c - 3)/2, 1, a]
>>> lhs, rhs = to_arrays(expressions, [a, b, c])
>>> lhs.nnz
5
>>> lhs.todense()
array([[2. , 0. , 0. ],
        [1. , 1. , 0. ],
        [0. , 0. , 1.5],
        [0. , 0. , 0. ],
        [1. , 0. , 0. ]])
>>> rhs
[-1, -2, 1.5, -1, 0]
>>> lhs, rhs = to_arrays(expressions, [a, b, ])
>>> lhs.todense()
array([[2, 0],
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 0]])
>>> rhs
[-1, -2, -1.5*c + 1.5, -1, 0]

>>> A = numpy.asarray([[1, 2], [3, 4], [5, 6]])
>>> x = [Variable(f'c_{i}') for i in range(2)]
>>> expressions = A @ x
>>> expressions
array([c_0 + 2*c_1, 3*c_0 + 4*c_1, 5*c_0 + 6*c_1], dtype=object)
>>> lhs, rhs = to_arrays(expressions, x)
>>> numpy.all(lhs == A)
True
```