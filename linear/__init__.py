"""linear: LINear Expressions for Algebraic Relief
The package linear makes it easy and quick to develop sparse matrices from linear expressions
without resorting to laborious indexing and index mapping.  Just express your equations in
natural syntax and generate sparse matrices as output.
"""
import scipy
import numpy


NumberLikeTypes = (int, float, complex, )


class Expression:
    """A linear expression that can be added, subtracted, multiplied, and divided
    so long as it remains linear.  Currently, it is a simple accumulator and does
    not, for example, support expressions such as:

    Variable('a')/(2*Variable('a') = 0.5

    Arguments:
        variables (dict):
            A dictionary representing the variables and their coefficients in the expression.
            The keys are instances of the Variable class, and the values are the corresponding
            coefficients.
        constant (float, int, complex, optional):
            A constant term in the expression. Default is 0.

    Examples:
        >>> 2 * Variable('a') + 1
        2*a + 1
        >>> (Variable('a') - 1) / 2
        0.5*a - 0.5

    Attributes:
        _coefficients (dict):
            A dict containing the variables and their coefficients in the expression.
            The keys are instances of the Variable class, and the values are the corresponding
            coefficients.
        _constant (float, int, or complex):
            The constant term in the expression.

    Methods:
        __hash__(): Returns the hash value of the Expression object.
        __eq__(other): Checks if the current Expression object is equal to the 'other' object.
        __ne__(other): Checks if the current Expression object is not equal to the 'other' object.
        __repr__(): Returns a string representation of the Expression object.
        __add__(other): Adds the current Expression object to the 'other' object.
        __mul__(other): Multiplies the current Expression object by the 'other' object.
        __truediv__(other): Divides the current Expression object by the 'other' object.
        __sub__(other): Subtracts the 'other' object from the current Expression object.
        __rsub__(other): Subtracts the current Expression object from the 'other' object.
        __radd__(other): Adds the 'other' object to the current Expression object.
        __rmul__(other): Multiplies the 'other' object by the current Expression object.

    Additional Examples:
        >>> 2*Variable('a') + 1
        2*a + 1
        >>> 1*Variable('a') + 0
        a
        >>> 3 + 4*Variable('a')
        4*a + 3
        >>> Variable('a')/2 + 2
        0.5*a + 2
        >>> Variable('a') + 2*Variable('b')
        a + 2*b
        >>> (Variable('a') + 2*Variable('b')) + (Variable('a') + 1)
        2*a + 2*b + 1
        >>> (Variable('a') + 1) + (Variable('a') + 2*Variable('b'))
        2*a + 2*b + 1
        >>> (Variable('a') + 1) + (Variable('a') + 2*Variable('b')) + 2
        2*a + 2*b + 3
        >>> (Variable('a') + 1) + (Variable('a') + 2*Variable('b')) + Variable('c') + 1
        2*a + 2*b + c + 2
        >>> (Variable('a') - 1) - (Variable('a') - 2*Variable('b')) - Variable('c') + 1
        2*b - c
        >>> hash(Variable('a')) == hash(Variable('a'))
        True
        >>> Variable('a') == Variable('a')
        True
        >>> Variable('a') == Variable('b')
        False
        >>> Variable('a') != Variable('b')
        True
        >>> (Variable('a') - 1)/2
        0.5*a - 0.5
        >>> 2*(Variable('a') - 2) + 2
        2*a - 2
        >>> 1/(Variable('a') - 2)*2 + 2
        Traceback (most recent call last):
        ...
        TypeError: unsupported operand type(s) for /: 'int' and 'Expression'
        >>> Variable('b')/(Variable('a') - 2)*2 + 2
        Traceback (most recent call last):
        ...
        TypeError: unsupported operand type(s) for /: 'Variable' and 'Expression'
        >>> expression = Variable('a') + 2*Variable('b') + 2
        >>> 2 in expression
        True
        >>> 3 in expression
        False
        >>> Variable('a') in expression
        True
        >>> Variable('e') in expression
        False
        >>> -(Variable('a') - 1)
        -a + 1
        >>> +(Variable('a') - 1)
        a - 1
        >>> -(1.2*Variable('a') + Variable('b')) + 2
        -1.2*a - b + 2
        >>> -1.5*Variable('a') - 1.3*Variable('b')
        -1.5*a - 1.3*b
        >>> expression = -1.5*Variable('a') - 1.3*Variable('b') + 0.1
        >>> expression
        -1.5*a - 1.3*b + 0.1
        >>> expression.constant
        0.1
        >>> expression.coefficients()[Variable('a')] = -1000
        >>> expression + Variable('a')
        -0.5*a - 1.3*b + 0.1
        >>> expression.variables() == set((Variable('a'), Variable('b')))
        True
        >>> expression
        -1.5*a - 1.3*b + 0.1
        >>> expression.subs({Variable('a'): 1})
        -1.3*b - 1.4
        >>> -(3*Variable('c') - 3)/2
        -1.5*c + 1.5
        >>> expression = -1.5
        >>> expression = expression - 1.5*Variable('c')
        >>> expression
        -1.5*c - 1.5

    """

    def __init__(self, coefficients, constant=0):
        self._coefficients = {
            var: coef
            for var, coef in coefficients.items() if coef != 0
        }
        self._constant = constant

    def subs(self, coefficients):
        common_variables = set(coefficients).intersection(self._coefficients)
        constant = self.constant
        for variable in common_variables:
            constant += self.coefficient(variable) * coefficients[variable]
        expression = Expression(
            {
                var: coef
                for var, coef in self._coefficients.items()
                if var not in common_variables
            }, constant)
        if len(expression._coefficients) == 0:
            return expression.constant
        else:
            return expression

    def coefficient(self, variable):
        if isinstance(variable, Variable):
            return self._coefficients[variable]
        else:
            raise TypeError()

    def variables(self):
        return set(self._coefficients.keys())

    @property
    def constant(self):
        return self._constant

    def coefficients(self):
        return self._coefficients.copy()

    def __hash__(self):
        return hash(
            tuple(items for items in self._coefficients.items()) +
            (self._constant, ))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __eq__(self, other):
        return hash(self) != hash(other)

    def __repr__(self):
        if len(self._coefficients) == 0:
            return str(self._constant)
        else:
            coefficients = sorted(self._coefficients.items(),
                                  key=lambda item: item[0]._name)
            variable, coefficient = coefficients[0]
            if coefficient == -1:
                repr = f'-{variable._name}'
            elif coefficient == 1:
                repr = f'{variable._name}'
            elif isinstance(coefficient, complex):
                repr = f'{coefficient}*{variable._name}'
            else:
                repr = f'{coefficient}*{variable._name}'
            for variable, coefficient in coefficients[1:]:
                if coefficient == -1:
                    repr += f' - {variable._name}'
                elif coefficient == 1:
                    repr += f' + {variable._name}'
                elif isinstance(coefficient, complex):
                    repr += f' + {coefficient}*{variable._name}'
                elif coefficient < 0:
                    repr += f' - {abs(coefficient)}*{variable._name}'
                elif coefficient > 0:
                    repr += f' + {coefficient}*{variable._name}'
                else:
                    continue
            if isinstance(self._constant, complex):
                repr += f' + {self._constant}'
            elif self._constant < 0:
                repr += f' - {abs(self._constant)}'
            elif self._constant > 0:
                repr += f' + {self._constant}'
            return repr

    def __add__(self, other):
        if isinstance(other, NumberLikeTypes):
            return Expression(self._coefficients, self._constant + other)
        elif isinstance(other, Variable):
            coefficients = self.coefficients()
            if other in coefficients:
                coefficients[other] += 1
            else:
                coefficients[other] = 1
            return Expression(coefficients, self._constant)
        elif isinstance(other, Expression):
            coefficients = self.coefficients()
            for variable, coefficient in other._coefficients.items():
                if variable in coefficients:
                    coefficients[variable] += coefficient
                else:
                    coefficients[variable] = coefficient
            return Expression(coefficients, self._constant + other._constant)
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, NumberLikeTypes):
            coefficients = self.coefficients()
            for variable, coefficient in self._coefficients.items():
                coefficients[variable] = coefficient * other
            return Expression(coefficients, self._constant * other)
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (float, int)):
            coefficients = {
                variable: coeff / other
                for variable, coeff in self._coefficients.items()
            }
            return Expression(coefficients, self._constant / other)
        else:
            return NotImplemented

    def __pos__(self):
        return 1 * self

    def __neg__(self):
        return -1 * self

    def __sub__(self, other):
        if isinstance(other, NumberLikeTypes):
            return Expression(self._coefficients, self._constant - other)
        elif isinstance(other, Variable):
            coefficients = self.coefficients()
            if other in coefficients:
                coefficients[other] -= 1
            else:
                coefficients[other] = -1
            return Expression(coefficients, self._constant)
        elif isinstance(other, Expression):
            coefficients = self.coefficients()
            for variable, coefficient in other._coefficients.items():
                if variable in coefficients:
                    coefficients[variable] -= coefficient
                else:
                    coefficients[variable] = -coefficient
            return Expression(coefficients, self._constant - other._constant)
        else:
            return NotImplemented

    def __contains__(self, other):
        if isinstance(other, NumberLikeTypes):
            return other == self._constant
        if isinstance(other, Variable):
            return other in self._coefficients

    def __rsub__(self, other):
        if isinstance(other, NumberLikeTypes):
            coefficients = {
                var: -coef
                for var, coef in self._coefficients.items()
            }
            return Expression(coefficients, other - self._constant)
        elif isinstance(other, Variable):
            coefficients = {
                var: -coef
                for var, coef in self._coefficients.items()
            }
            if other in coefficients:
                coefficients[other] += 1
            else:
                coefficients[other] = 1
            return Expression(coefficients, self._constant)
        elif isinstance(other, Expression):
            coefficients = {
                var: -coef
                for var, coef in self._coefficients.items()
            }
            for variable, coefficient in other._coefficients.items():
                if variable in coefficients:
                    coefficients[variable] += coefficient
                else:
                    coefficients[variable] = coefficient
            return Expression(coefficients, other._constant - self._constant)
        else:
            return NotImplemented

    __radd__ = __add__
    __rmul__ = __mul__


def is_iterable(obj):
    try:
        iterator = iter(obj)
    except TypeError:
        return False
    else:
        return True


def variables_in_expressions(expressions):
    variables = set()
    for expression in expressions:
        variables.update(expression.variables())
    return sorted(variables)


def solve(expressions, variables=None, solver='dense'):
    if variables is None:
        variables = variables_in_expressions(expressions)
    variables = list(variables)
    lhs, rhs = to_arrays(expressions, variables)
    if len(variables) < 10 or solver == 'dense':
        s = numpy.linalg.solve(lhs.todense(), rhs)
    elif solver == 'banded':
        s = numpy.linalg.solve_banded(lhs, rhs)
    elif solver == 'sparse':
        s = scipy.sparse.linalg.spsolve(lhs, rhs)
    else:
        raise ValueError
    return {vi: si for vi, si in zip(variables, s)}


def subs(expressions, variables):
    substituted_expressions = []
    for expression in expressions:
        if isinstance(expression, Expression):
            substituted_expressions.append(expression.subs(variables))
        elif isinstance(expression, Variable):
            try:
                substituted_expressions.append(variables[expression])
            except KeyError:
                substituted_expressions.append(expression)
        elif isinstance(expression, (int, float, complex)):
            substituted_expressions.append(expression)
        else:
            raise TypeError
    return substituted_expressions


def to_arrays(expressions, variables):
    """Convert a list of linear expressions to a sparse LHS array and a dense RHS vector.

    This function takes a list of linear expressions and an optional list of variables
    as input. It constructs a sparse matrix representing the left-hand side (LHS)
    of the equations and a vector representing the right-hand side (RHS) values.

    Arguments:
        expressions (list):
            A list of linear expressions.
        variables (list, optional):
            A list of variables used in the expressions. If not provided,
            the variables will be determined automatically from the expressions.
            Defaults to None.

    Returns:
        A tuple containing the LHS sparse matrix and the RHS array.


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

    """
    variables = list(variables)
    # create mapping from variable to column j
    variable_to_j = {variable: j for j, variable in enumerate(variables)}
    # cleanup expressions, handling cases where they are composed exclusively
    # of numbers NumberLikeTypes or Variables
    for i in range(len(expressions)):
        expression = expressions[i]
        if isinstance(expression, NumberLikeTypes):
            expressions[i] = Expression({}, expression)
        if isinstance(expression, Variable):
            expressions[i] = Expression({expression: 1})
    M, N = len(expressions), len(variables)
    rows = []
    cols = []
    data = []
    rhs = []
    for i, expression in enumerate(expressions):
        # avoid -0
        rhsi = -expression.constant
        """
        if expression.constant == 0:
            rhs.append(0)
        else:
            rhs.append(-expression._constant)
        """
        # for variable in expression._coefficients:
        for variable, coefficient in expression._coefficients.items():
            j = variable_to_j.get(variable)
            if j is None:
                rhsi -= coefficient * variable
                continue
            rows.append(i)
            cols.append(j)
            data.append(coefficient)
        rhs.append(rhsi)
    lhs = scipy.sparse.coo_array((data, (rows, cols)), shape=(M, N))
    rhs = rhs
    return lhs, rhs


class Variable:
    """Represents a variable in a linear expression.

    Arguments:
        name (str):
            The name of the variable.

    Examples:
        >>> a = Variable('a')
        >>> a
        a
        >>> 2*a
        2*a

    Attributes:
        name (str): The name of the variable.

    Methods:
        __repr__(): Returns a string representation of the Variable object.
        __hash__(): Returns the hash value of the Variable object.
        __eq__(other): Checks if the current Variable object is equal to the 'other' object.
        __ne__(other): Checks if the current Variable object is not equal to the 'other' object.
        __add__(other): Adds the current Variable object to the 'other' object.
        __sub__(other): Subtracts the 'other' object from the current Variable object.
        __mul__(other): Multiplies the current Variable object by the 'other' object.
        __truediv__(other): Divides the current Variable object by the 'other' object.
        __radd__(other): Adds the 'other' object to the current Variable object.
        __rmul__(other): Multiplies the 'other' object by the current Variable object.

    """

    def __init__(self, name):
        assert isinstance(name, (str, bytes))
        self._name = name

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return self._name

    def __hash__(self):
        return hash(self._name)

    def __eq__(self, other):
        return self._name == other._name

    def __ne__(self, other):
        return self._name != other._name

    def __gt__(self, other):
        return self._name > other._name

    def __lt__(self, other):
        return self._name < other._name

    def __ge__(self, other):
        return self._name >= other._name

    def __le__(self, other):
        return self._name <= other._name

    def __add__(self, other):
        if isinstance(other, (int, float, complex)):
            return Expression({self: 1}, other)
        elif isinstance(other, Variable):
            return Expression({self: 1, other: 1})
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, (int, float, complex)):
            return Expression({self: 1}, -other)
        elif isinstance(other, Variable):
            return Expression({self: 1, other: -1})
        else:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float, complex)):
            return Expression({self: -1}, other)
        elif isinstance(other, Variable):
            return Expression({self: -1, other: 1})
        else:
            return NotImplemented

    def __neg__(self):
        return Expression({self: -1})

    def __pos__(self):
        return Expression({self: 1})

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            return Expression({self: other})
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float, complex)):
            return Expression({self: 1 / other})
        else:
            return NotImplemented

    __radd__ = __add__
    __rmul__ = __mul__


if __name__ == '__main__':
    import doctest
    doctest.testmod()
