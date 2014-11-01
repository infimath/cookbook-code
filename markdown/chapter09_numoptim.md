# numoptim


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 9.1. Finding the root of a mathematical function

1. Let's import NumPy, SciPy, scipy.optimize, and matplotlib.


``` python
import numpy as np
import scipy as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt
%matplotlib inline
```


2. We define the mathematical function $f(x)=\cos(x)-x$ in Python. We will try to find a root of this function numerically, which corresponds to a fixed point of the cosine function.


``` python
f = lambda x: np.cos(x) - x
```


3. Let's plot this function on the interval $[-5, 5]$.


``` python
x = np.linspace(-5, 5, 1000)
y = f(x)
plt.figure(figsize=(5,3));
plt.plot(x, y);
plt.axhline(0, color='k');
plt.xlim(-5,5);
```


4. We see that this function has a unique root on this interval (this is because the function's sign changes on this interval). The scipy.optimize module contains a few root-finding functions that are adapted here. For example, the `bisect` function implements the **bisection method** (also called the **dichotomy method**). It takes as input the function and the interval to find the root in.


``` python
opt.bisect(f, -5, 5)
```


Let's visualize the root on the plot.


``` python
plt.figure(figsize=(5,3));
plt.plot(x, y);
plt.axhline(0, color='k');
plt.scatter([_], [0], c='r', s=100);
plt.xlim(-5,5);
```


5. A faster and more powerful method is `brentq` (Brent's method). This algorithm also requires that $f$ is continuous and that $f(a)$ and $f(b)$ have different signs.


``` python
opt.brentq(f, -5, 5)
```


The `brentq` method is faster than `bisect`. If the conditions are satisfied, it is a good idea to try Brent's method first.


``` python
%timeit opt.bisect(f, -5, 5)
%timeit opt.brentq(f, -5, 5)
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 9.2. Minimizing a mathematical function

1. We import the libraries.


``` python
import numpy as np
import scipy as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt
%matplotlib inline
```


2. First, let's define a simple mathematical function (the opposite of the **cardinal sine**). This function has many local minima but a single global minimum. (http://en.wikipedia.org/wiki/Sinc_function)


``` python
f = lambda _: 1-np.sin(_)/_
```


3. Let's plot this function on the interval $[-20, 20]$.


``` python
x = np.linspace(-20., 20., 1000)
y = f(x)
```



``` python
plt.figure(figsize=(5,5));
plt.plot(x, y);
```


4. The `scipy.optimize` module comes with many function minimization routines. The `minimize` function offers a unified interface to many algorithms. The **Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm** (default in `minimize`) gives good results in general. The `minimize` function requires an initial point as argument. For scalar univariate functions, we can also use `minimize_scalar`.


``` python
x0 = 3
xmin = opt.minimize(f, x0).x
```


Starting from $x_0=3$, the algorithm was able to find the actual global minimum, as shown on the following figure.


``` python
plt.figure(figsize=(5,5));
plt.plot(x, y);
plt.scatter(x0, f(x0), marker='o', s=300);
plt.scatter(xmin, f(xmin), marker='v', s=300);
plt.xlim(-20, 20);
```


5. Now, if we start from an initial point that is further away from the actual global minimum, the algorithm converges towards a *local* minimum only.


``` python
x0 = 10
xmin = opt.minimize(f, x0).x
```



``` python
plt.figure(figsize=(5,5));
plt.plot(x, y);
plt.scatter(x0, f(x0), marker='o', s=300);
plt.scatter(xmin, f(xmin), marker='v', s=300);
plt.xlim(-20, 20);
```


6. Like most function minimization algorithms, the BFGS algorithm is efficient at finding *local* minima, but not necessarily *global* minima, especially on complicated or noisy objective functions. A general strategy to overcome this problem consists in combining such algorithms with an exploratory grid search on the initial points. Another option is to use a different class of algorithms based on heuristics and stochastic methods. A popular example is the **simulated annealing method**.


``` python
xmin = opt.minimize(f, x0, method='Anneal').x
```



``` python
plt.figure(figsize=(5,5));
plt.plot(x, y);
plt.scatter(x0, f(x0), marker='o', s=300);
plt.scatter(xmin, f(xmin), marker='v', s=300);
plt.xlim(-20, 20);
```


This time, the algorithm was able to find the global minimum.

7. Now, let's define a new function, in two dimensions this time. This function is called the **Lévi function**. It is defined by

$$f(x,y) = \sin^{2}\left(3\pi x\right)+\left(x-1\right)^{2}\left(1+\sin^{2}\left(3\pi y\right)\right)+\left(y-1\right)^{2}\left(1+\sin^{2}\left(2\pi y\right)\right)$$

This function is very irregular and may be difficult to minimize in general. It is one of the many **test functions for optimization** that researchers have developed to study and benchmark optimization algorithms. (http://en.wikipedia.org/wiki/Test_functions_for_optimization)


``` python
def g(X):
    # X is a 2*N matrix, each column contains
    # x and y coordinates.
    x, y = X
    return np.sin(3*np.pi*x)**2+(x-1)**2*(1+np.sin(3*np.pi*y)**2)+(y-1)**2*(1+np.sin(2*np.pi*y)**2)
```


8. Let's display this function with `imshow`, on the square $[-10,10]^2$.


``` python
n = 200
k = 10
X, Y = np.mgrid[-k:k:n*1j,-k:k:n*1j]
```



``` python
Z = g(np.vstack((X.ravel(), Y.ravel()))).reshape(n,n)
```



``` python
plt.figure(figsize=(5, 5));
# We use a logarithmic scale for the color here.
plt.imshow(np.log(Z), cmap=plt.cm.hot_r);
plt.xticks([]); plt.yticks([]);
```


9. The BFGS algorithm also works in multiple dimensions.


``` python
x0, y0 = opt.minimize(g, (8, 3)).x
```



``` python
plt.figure(figsize=(5, 5));
plt.imshow(np.log(Z), cmap=plt.cm.hot_r,
           extent=(-k, k, -k, k), origin=0);
plt.scatter(x0, y0, s=100);
plt.xticks([]); plt.yticks([]);
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 9.3. Fitting a function to data with nonlinear least squares

1. Let's import the usual libraries.


``` python
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
%matplotlib inline
np.random.seed(3)
```


2. We define a logistic function with four parameters.

$$f_{a,b,c,d}(x) = \frac{a}{1 + \exp\left(-c (x-d)\right)} + b$$


``` python
def f(x, a, b, c, d):
    return a/(1. + np.exp(-c * (x-d))) + b
```


3. Let's define four random parameters.


``` python
a, c = np.random.exponential(size=2)
b, d = np.random.randn(2)
```


4. Now, we generate random data points, by using the sigmoid function and adding a bit of noise.


``` python
n = 100
x = np.linspace(-10., 10., n)
y_model = f(x, a, b, c, d)
y = y_model + a * .2 * np.random.randn(n)
```


5. Here is a plot of the data points, with the particular sigmoid used for their generation.


``` python
plt.figure(figsize=(6,4));
plt.plot(x, y_model, '--k');
plt.plot(x, y, 'o');
```


6. We now assume that we only have access to the data points. These points could have been obtained during an experiment. By looking at the data, the points appear to approximately follow a sigmoid, so we may want to try to fit such a curve to the points. That's what **curve fitting** is about. SciPy's function `curve_fit` allows us to fit a curve defined by an arbitrary Python function to the data.


``` python
(a_, b_, c_, d_), _ = opt.curve_fit(f, x, y, (a, b, c, d))
```


7. Now, let's take a look at the fitted simoid curve.


``` python
y_fit = f(x, a_, b_, c_, d_)
```



``` python
plt.figure(figsize=(6,4));
plt.plot(x, y_model, '--k');
plt.plot(x, y, 'o');
plt.plot(x, y_fit, '-');
```


The fitted sigmoid appears to be quite close from the original sigmoid used for data generation.

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 9.4. Finding the equilibrium state of a physical system by minimizing its potential energy

1. Let's import NumPy, SciPy and matplotlib.


``` python
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
%matplotlib inline
```


2. We define a few constants in the International System of Units.


``` python
g = 9.81  # gravity of Earth
m = .1  # mass, in kg
n = 20  # number of masses
e = .1  # initial distance between the masses
l = e  # relaxed length of the springs
k = 10000  # spring stiffness
```


3. We define the initial positions of the masses. They are arranged on a two-dimensional grid with two lines and $n/2$ columns.


``` python
P0 = np.zeros((n, 2))
P0[:,0] = np.repeat(e*np.arange(n//2), 2)
P0[:,1] = np.tile((0,-e), n//2)
```


4. Now, let's define the connectivity matrix between the masses. Coefficient $i,j$ is $1$ if masses $i$ and $j$ are connected by a spring.


``` python
A = np.eye(n, n, 1) + np.eye(n, n, 2)
```


5. We also specify the spring stiffness of each spring. It is $l$, except for *diagonal* springs where it is $l \times \sqrt{2}$.


``` python
L = l * (np.eye(n, n, 1) + np.eye(n, n, 2))
for i in range(n//2-1):
    L[2*i+1,2*i+2] *= np.sqrt(2)
```


6. We also need the indices of the spring connections.


``` python
I, J = np.nonzero(A)
```


7. The `dist` function computes the distance matrix (distance between any pair of masses).


``` python
dist = lambda P: np.sqrt((P[:,0]-P[:,0][:, np.newaxis])**2 + 
                         (P[:,1]-P[:,1][:, np.newaxis])**2)
```


7. We define a function that displays the system. The springs are colored according to their tension.


``` python
def show_bar(P):
    plt.figure(figsize=(5,4));
    # Wall.
    plt.axvline(0, color='k', lw=3);
    # Distance matrix.
    D = dist(P)
    # We plot the springs.
    for i, j in zip(I, J):
        # The color depends on the spring tension, which
        # is proportional to the spring elongation.
        c = D[i,j] - L[i,j]
        plt.plot(P[[i,j],0], P[[i,j],1], 
                 lw=2, color=plt.cm.copper(c*150));
    # We plot the masses.
    plt.plot(P[[I,J],0], P[[I,J],1], 'ok',);
    # We configure the axes.
    plt.axis('equal');
    plt.xlim(P[:,0].min()-e/2, P[:,0].max()+e/2);
    plt.ylim(P[:,1].min()-e/2, P[:,1].max()+e/2);
    plt.xticks([]); plt.yticks([]);
```


8. Here is the system in its initial configuration.


``` python
show_bar(P0);
plt.title("Initial configuration");
```


9. To find the equilibrium state, we need to minimize the total potential energy of the system. The following function computes the energy of the system, given the positions of the masses. This function is explained in *How it works...*.


``` python
def energy(P):
    # The argument P is a vector (flattened matrix).
    # We convert it to a matrix here.
    P = P.reshape((-1, 2))
    # We compute the distance matrix.
    D = dist(P)
    # The potential energy is the sum of the
    # gravitational and elastic potential energies.
    return (g * m * P[:,1].sum() + 
            .5 * (k * A * (D - L)**2).sum())
```


10. Let's compute the potential energy of the initial configuration.


``` python
energy(P0.ravel())
```


11. Now, let's minimize the potential energy with a function minimization method. We need a **constrained optimization algorithm**, because we make the assumption that the two first masses are fixed to the wall. Therefore, their positions cannot change. The **L-BFGS-B** algorithm, a variant of the BFGS algorithm, accepts bound constraints. Here, we force the first two points to stay at their initial positions, whereas there are no constraints on the other points. The `minimize` function accepts a `bounds` list containing, for each dimension, a pair of `[min, max]` values. (http://en.wikipedia.org/wiki/Limited-memory_BFGS#L-BFGS-B)


``` python
bounds = np.c_[P0[:2,:].ravel(), P0[:2,:].ravel()].tolist() + \
         [[None, None]] * (2*(n-2))
```



``` python
P1 = opt.minimize(energy, P0.ravel(),
                  method='L-BFGS-B',
                  bounds=bounds).x.reshape((-1, 2))
```


12. Let's display the stable configuration.


``` python
show_bar(P1);
plt.title("Equilibrium configuration");
```


This configuration looks realistic. The tension appears to be maximal on the top springs near the wall.

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).

