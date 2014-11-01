# symbolic


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 15.1. Diving into symbolic computing with SymPy

SymPy is a pure Python package for symbolic mathematics.

First, we import SymPy, and enable rich display LaTeX-based printing in the IPython notebook (using the MathJax Javascript library).


``` python
from sympy import *
init_printing()
```


With NumPy and the other packages we have been using so far, we were dealing with numbers and numerical arrays. With SymPy, we deal with symbolic variables. It's a radically different shift of paradigm, which mathematicians may be more familiar with.

To deal with symbolic variables, we need to declare them.


``` python
var('x y')
```


The var function creates symbols and injects them into the namespace. This function should only be used in interactive mode. In a Python module, it is better to use the symbol function which returns the symbols.


``` python
x, y = symbols('x y')
```


We can create mathematical expressions with these symbols.


``` python
expr1 = (x + 1)**2
expr2 = x**2 + 2*x + 1
```


Are these expressions equal?


``` python
expr1 == expr2
```


These expressions are mathematically equal, but not syntactically identical. To test whether they are equal, we can ask SymPy to simplify the difference algebraically.


``` python
simplify(expr1-expr2)
```


A very common operation with symbolic expressions is substitution of a symbol by another symbol, expression, or a number.


``` python
expr1.subs(x, expr1)
```



``` python
expr1.subs(x, pi)
```


A rational number cannot be written simply as "1/2" as this Python expression evaluates to 0. A possibility is to use a SymPy object for 1, for example using the function S.


``` python
expr1.subs(x, S(1)/2)
```


Exactly-represented numbers can be evaluated numerically with evalf:


``` python
_.evalf()
```


We can transform this *symbolic* function into an actual Python function that can be evaluated on NumPy arrays, using the `lambdify` function.


``` python
f = lambdify(x, expr1)
```



``` python
import numpy as np
f(np.linspace(-2., 2., 5))
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 15.2. Solving equations and inequalities


``` python
from sympy import *
init_printing()
```



``` python
var('x y z a')
```


Use the function solve to resolve equations (the right hand side is always 0).


``` python
solve(x**2 - a, x)
```


You can also solve inequations. You may need to specify the domain of your variables. Here, we tell SymPy that x is a real variable.


``` python
x = Symbol('x')
solve_univariate_inequality(x**2 > 4, x)
```


## Systems of equations

This function also accepts systems of equations (here a linear system).


``` python
solve([x + 2*y + 1, x - 3*y - 2], x, y)
```


Non-linear systems are also supported.


``` python
solve([x**2 + y**2 - 1, x**2 - y**2 - S(1)/2], x, y)
```


Singular linear systems can also be solved (here, there are infinitely many equations because the two equations are colinear).


``` python
solve([x + 2*y + 1, -x - 2*y - 1], x, y)
```


Now, let's solve a linear system using matrices with symbolic variables.


``` python
var('a b c d u v')
```


We create the augmented matrix, which is the horizontal concatenation of the system's matrix with the linear coefficients, and the right-hand side vector.


``` python
M = Matrix([[a, b, u], [c, d, v]]); M
```



``` python
solve_linear_system(M, x, y)
```


This system needs to be non-singular to have a unique solution, which is equivalent to say that the determinant of the system's matrix needs to be non-zero (otherwise the denominators in the fractions above are equal to zero).


``` python
det(M[:2,:2])
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 15.3. Analyzing real-valued functions


``` python
from sympy import *
init_printing()
```



``` python
var('x z')
```


We define a new function depending on x.


``` python
f = 1/(1+x**2)
```


Let's evaluate this function in 1.


``` python
f.subs(x, 1)
```


We can compute the derivative of this function...


``` python
diff(f, x)
```


limits...


``` python
limit(f, x, oo)
```


Taylor series...


``` python
series(f, x0=0, n=9)
```


Definite integrals...


``` python
integrate(f, (x, -oo, oo))
```


indefinite integrals...


``` python
integrate(f, x)
```


and even Fourier transforms!


``` python
fourier_transform(f, x, z)
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 15.4. Computing exact probabilities and manipulating random variables


``` python
from sympy import *
from sympy.stats import *
init_printing()
```


## Rolling dice

Let's roll two dices X and Y.


``` python
X, Y = Die('X', 6), Die('Y', 6)
```


We can compute probabilities defined by equalities (with the Eq operator) or inequalities...


``` python
P(Eq(X, 3))
```



``` python
P(X>3)
```


Conditions can also involve multiple random variables...


``` python
P(X>Y)
```


Conditional probabilities...


``` python
P(X+Y>6, X<5)
```


## Continuous random variables

We can also work with arbitrary discrete or continuous random variables.


``` python
Z = Normal('Z', 0, 1)  # Gaussian variable
```



``` python
P(Z>pi)
```


We can compute expectancies and variances...


``` python
E(Z**2), variance(Z**2)
```


as well as densities.


``` python
f = density(Z)
```


This is a lambda function, it can be evaluated on a SymPy symbol:


``` python
var('x')
f(x)
```


We can plot this density.


``` python
%matplotlib inline
plot(f(x), (x, -6, 6));
```


SymPy.stats works by using integrals and summations for computing probabilistic quantities. For example, P(Z>pi) is:


``` python
Eq(Integral(f(x), (x, pi, oo)), 
   simplify(integrate(f(x), (x, pi, oo))))
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 15.5. A bit of number theory with SymPy


``` python
from sympy import *
init_printing()
```



``` python
import sympy.ntheory as nt
```


## Prime numbers

Test whether a number is prime.


``` python
nt.isprime(2011)
```


Find the next prime after a given number.


``` python
nt.nextprime(2011)
```


What is the 1000th prime number?


``` python
nt.prime(1000)
```


How many primes less than 2011 are there?


``` python
nt.primepi(2011)
```


We can plot $\pi(x)$, the prime-counting function (the number of prime numbers less than or equal to some number x). The famous *prime number theorem* states that this function is asymptotically equivalent to $x/\log(x)$. This expression approximately quantifies the distribution of the prime numbers among all integers.


``` python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```



``` python
x = np.arange(2, 10000)
plt.plot(x, list(map(nt.primepi, x)), '-k', label='$\pi(x)$');
plt.plot(x, x / np.log(x), '--k', label='$x/\log(x)$');
plt.legend(loc=2);
```


Let's compute the integer factorization of some number.


``` python
nt.factorint(1998)
```



``` python
2 * 3**3 * 37
```


## Chinese Remainder Theorem

A lazy mathematician is counting his marbles. When they are arranged in three rows, the last column contains one marble. When they form four rows, there are two marbles in the last column, and there are three with five rows. The Chinese Remainer Theorem can give him the answer directly.


``` python
from sympy.ntheory.modular import solve_congruence
```



``` python
solve_congruence((1, 3), (2, 4), (3, 5))
```


There are infinitely many solutions: 58, and 58 plus any multiple of 60. Since 118 seems visually too high, 58 is the right answer.

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 15.6. Finding a Boolean propositional formula from a truth table


``` python
from sympy import *
init_printing()
```


Let's define a few variables.


``` python
var('x y z')
```


We can define propositional formulas with symbols and a few operators.


``` python
P = x & (y | ~z); P
```



``` python
P.subs({x: True, y: False, z: True})
```


Now, we want to find a propositional formula depending on x, y, z, with the following truth table:


``` python
%%HTML
<style>
table.truth_table tr {
    margin: 0;
    padding: 0;
}
table.truth_table td, table.truth_table th {
    width: 30px;
    text-align: center;
    margin: 0;
    padding: 0;
}
</style>
<table class="truth_table">
<tr>
<th>x</th><th>y</th><th>z</th><th>??</th>
</tr>
<tr>
<td>T</td><td>T</td><td>T</td><th>*</th>
</tr>
<tr>
<td>T</td><td>T</td><td>F</td><th>*</th>
</tr>
<tr>
<td>T</td><td>F</td><td>T</td><th>T</th>
</tr>
<tr>
<td>T</td><td>F</td><td>F</td><th>T</th>
</tr>
<tr>
<td>F</td><td>T</td><td>T</td><th>F</th>
</tr>
<tr>
<td>F</td><td>T</td><td>F</td><th>F</th>
</tr>
<tr>
<td>F</td><td>F</td><td>T</td><th>F</th>
</tr>
<tr>
<td>F</td><td>F</td><td>F</td><th>T</th>
</tr>
</table>
```


Let's write down all combinations that we want to evaluate to True, and those for which the outcome does not matter.


``` python
minterms = [[1,0,1], [1,0,0], [0,0,0]]
dontcare = [[1,1,1], [1,1,0]]
```


Now, we use the SOPform function to derive an adequate proposition.


``` python
Q = SOPform(['x', 'y', 'z'], minterms, dontcare); Q
```


Let's test that this proposition works.


``` python
Q.subs({x: True, y: False, z: False}), Q.subs({x: False, y: True, z: True})
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 15.7. Analyzing a nonlinear differential system: Lotka-Volterra (predator-prey) equations

Here, we conduct a brief analytical study of a famous nonlinear differential system: the Lotka-Volterra equations, also known as predator-prey equations. This simple model describes the evolution of two interacting populations (e.g. sharks and sardines), where the predators eat the preys. This example illustrates how we can use SymPy to obtain exact expressions and results for fixed points and their stability.


``` python
from sympy import *
init_printing()
```



``` python
var('x y')
var('a b c d', positive=True)
```


The variables x and y represent the populations of the preys and predators, respectively. The parameters a, b, c and d are positive parameters (described more precisely in "How it works..."). The equations are:

$$\begin{align}
\frac{dx}{dt} &= f(x) = x(a-by)\\
\frac{dy}{dt} &= g(x) = -y(c-dx)
\end{align}$$


``` python
f = x * (a - b*y)
g = -y * (c - d*x)
```


Let's find the fixed points of the system (solving f(x,y) = g(x,y) = 0).


``` python
solve([f, g], (x, y))
```



``` python
(x0, y0), (x1, y1) = _
```


Let's write the 2D vector with the two equations.


``` python
M = Matrix((f, g)); M
```


Now we can compute the Jacobian of the system, as a function of (x, y).


``` python
J = M.jacobian((x, y)); J
```


Let's study the stability of the two fixed points by looking at the eigenvalues of the Jacobian at these points.


``` python
M0 = J.subs(x, x0).subs(y, y0); M0
```



``` python
M0.eigenvals()
```


The parameters a and c are strictly positive, so the eigenvalues are real and of opposite signs, and this fixed point is a saddle point. Since this point is unstable, the extinction of both populations is unlikely in this model.


``` python
M1 = J.subs(x, x1).subs(y, y1); M1
```



``` python
M1.eigenvals()
```


The eigenvalues are purely imaginary so this fixed point is not hyperbolic, and we cannot draw conclusions about the qualitative behavior of the system around this fixed point from this linear analysis. However, one can show with other methods that oscillations occur around this point.

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).

