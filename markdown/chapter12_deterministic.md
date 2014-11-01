# deterministic


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 12.1. Plotting the bifurcation diagram of a chaotic dynamical system

1. We import NumPy and matplotlib.


``` python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```


2. We define the logistic function by:

$$f_r(x) = rx(1-x)$$

Our discrete dynamical system is defined by the recursive application of the logistic function:

$$x_{n+1}^{(r)} = f_r(x_n^{(r)}) = rx_n^{(r)}(1-x_n^{(r)})$$


``` python
def logistic(r, x):
    return r*x*(1-x)
```


3. We will simulate this system for 10000 values of $r$ linearly spaced between 2.5 and 4. Of course, we vectorize the simulation with NumPy.


``` python
n = 10000
r = np.linspace(2.5, 4.0, n)
```


4. We will simulate 1000 iterations of the logistic map, and we will keep the last 100 iterations to display the bifurcation diagram.


``` python
iterations = 1000
last = 100
```


5. We initialize our system with the same initial condition $x_0 = 10^{-5}$.


``` python
x = 1e-5 * np.ones(n)
```


6. We will also compute an approximation of the Lyapunov exponent, for every value of $r$. The Lyapunov exponent is defined by:

$$\lambda(r) = \lim_{n \to \infty} \frac{1}{n} \sum_{i=0}^{n-1} \log\left| \frac{df_r}{dx}\left(x_i^{(r)}\right) \right|$$


``` python
lyapunov = np.zeros(n)
```


7. Now, we simulate the system and we plot the bifurcation diagram. The simulation only involves the iterative evaluation of the function $f$ on our vector $x$. Then, to display the bifurcation diagram, we draw one pixel per point $x_n^{(r)}$ during the last 100 iterations.


``` python
plt.figure(figsize=(6,7));
plt.subplot(211);
for i in range(iterations):
    x = logistic(r, x)
    # We compute the partial sum of the Lyapunov exponent.
    lyapunov += np.log(abs(r-2*r*x))
    # We display the bifurcation diagram.
    if i >= (iterations - last):
        plt.plot(r, x, ',k', alpha=.04)
plt.xlim(2.5, 4);
plt.title("Bifurcation diagram");

# We display the Lyapunov exponent.
plt.subplot(212);
plt.plot(r[lyapunov<0], lyapunov[lyapunov<0] / iterations,
         ',k', alpha=.2);
plt.plot(r[lyapunov>=0], lyapunov[lyapunov>=0] / iterations,
         ',r', alpha=.5);
plt.xlim(2.5, 4);
plt.ylim(-2, 1);
plt.title("Lyapunov exponent");
plt.tight_layout();
```


The bifurcation diagram brings out the existence of a fixed point for $r<3$, then two and four equilibria... until a chaotic behavior when $r$ belongs to certain areas of the parameter space.
We observe an important property of the Lyapunov exponent: it is positive when the system is chaotic (in red here).

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 12.2. Simulating an elementary cellular automaton

1. We import NumPy and matplotlib.


``` python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```


2. We will use the following vector to obtain numbers written in binary representation.


``` python
u = np.array([[4], [2], [1]])
```


3. We write a function that performs one iteration on the grid, updating all cells at once according to the given rule in binary representation. The first step consists in stacking circularly shifted versions of the grid to get the LCR triplets of each cell (`y`). Then, we convert these triplets in 3-bit numbers (`z`). Finally, we compute the next state of every cell using the specified rule.


``` python
def step(x, rule_binary):
    """Compute a single stet of an elementary cellular
    automaton."""
    # The columns contains the L, C, R values
    # of all cells.
    y = np.vstack((np.roll(x, 1), x,
                   np.roll(x, -1))).astype(np.int8)
    # We get the LCR pattern numbers between 0 and 7.
    z = np.sum(y * u, axis=0).astype(np.int8)
    # We get the patterns given by the rule.
    return rule_binary[7-z]
```


4. We now write a function that simulates any elementary cellular automaton. First, we compute the binary representation of the rule (**Wolfram's code**). Then, we initialize the first row of the grid to random values. Finally, we apply the function `step` iteratively on the grid.


``` python
def generate(rule, size=80, steps=80):
    """Simulate an elementary cellular automaton given its rule
    (number between 0 and 255)."""
    # Compute the binary representation of the rule.
    rule_binary = np.array([int(_) 
                            for _ in np.binary_repr(rule, 8)],
                            dtype=np.int8)
    x = np.zeros((steps, size), dtype=np.int8)
    # Random initial state.
    x[0,:] = np.random.rand(size) < .5
    # Apply the step function iteratively.
    for i in range(steps-1):
        x[i+1,:] = step(x[i,:], rule_binary)
    return x
```


5. Now, we simulate and display 9 different automata.


``` python
plt.figure(figsize=(6, 6));
rules = [3, 18, 30, 
         90, 106, 110, 
         158, 154, 184]
for i, rule in enumerate(rules):
    x = generate(rule)
    plt.subplot(331+i)
    plt.imshow(x, interpolation='none', cmap=plt.cm.binary);
    plt.xticks([]); plt.yticks([]);
    plt.title(str(rule))
```


It has been shown that Rule 110 is **Turing complete** (or **universal**): in principle, this automaton can simulate any computer program.

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 12.3. Simulating an Ordinary Differential Equation with SciPy

1. Let's import NumPy, SciPy (`integrate` package), and matplotlib.


``` python
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
%matplotlib inline
```


2. We define a few parameters appearing in our model.


``` python
m = 1.  # particle's mass
k = 1.  # drag coefficient
g = 9.81  # gravity acceleration
```


3. We have two variables: `x` and `y` (two dimensions). We note $\mathbf{u}=(x,y)$. The ODE we are going to simulate is:

$$\ddot{\mathbf{u}} = -\frac{k}{m} \dot{\mathbf{u}} + \mathbf{g}$$

where $\mathbf{g}$ is the gravity acceleration vector. In order to simulate this second-order ODE with SciPy, we can convert it to a first-order ODE (another option would be to solve $\dot{\mathbf{u}}$ first before integrating the solution). To do this, we consider two 2D variables: $\mathbf{u}$ and $\dot{\mathbf{u}}$. We note $\mathbf{v} = (\mathbf{u}, \dot{\mathbf{u}})$. We can express $\dot{\mathbf{v}}$ as a function of $\mathbf{v}$. Now, we create the initial vector $\mathbf{v}_0$ at time $t=0$: it has four components.


``` python
# The initial position is (0, 0).
v0 = np.zeros(4)
# The initial speed vector is oriented
# to the top right.
v0[2] = 4.
v0[3] = 10.
```


4. We need to create a Python function $f$ that takes the current vector $\mathbf{v}(t_0)$ and a time $t_0$ as argument (with optional parameters), and that returns the derivative $\dot{\mathbf{v}}(t_0)$.


``` python
def f(v, t0, k):
    # v has four components: v=[u, u'].
    u, udot = v[:2], v[2:]
    # We compute the second derivative u'' of u.
    udotdot = -k/m * udot
    udotdot[1] -= g
    # We return v'=[u', u''].
    return np.r_[udot, udotdot]
```


3. Now, we simulate the system for different values of $k$. We use the SciPy function `odeint`, defined in the `scipy.integrate` package.


``` python
plt.figure(figsize=(6,3));
# We want to evaluate the system on 30 linearly
# spaced times between t=0 and t=3.
t = np.linspace(0., 3., 30)
# We simulate the system for different values of k.
for k in np.linspace(0., 1., 5):
    # We simulate the system and evaluate $v$ on the 
    # given times.
    v = spi.odeint(f, v0, t, args=(k,))
    # We plot the particle's trajectory.
    plt.plot(v[:,0], v[:,1], 'o-', mew=1, ms=8, mec='w',
                label='k={0:.1f}'.format(k));
plt.legend();
plt.xlim(0, 12);
plt.ylim(0, 6);
```


The most outward trajectory (blue) corresponds to drag-free motion (without air resistance). It is a parabola. In the other trajectories, we can observe the increasing effect of air resistance, parameterized with $k$.

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 12.4. Simulating a Partial Differential Equation: reaction-diffusion systems and Turing patterns

1. Let's import the packages.


``` python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```


2. We will simulate the following system of partial differential equations on the domain $E=[-1,1]^2$:

\begin{align*}
\frac{\partial u}{\partial t} &= a \Delta u + u - u^3 - v + k\\
\tau\frac{\partial v}{\partial t} &= b \Delta v + u - v\\
\end{align*}

The variable $u$ represents the concentration of a substance favoring skin pigmentation, whereas $v$ represents another substance that reacts with the first and impedes pigmentation.

At initialization time, we assume that $u$ and $v$ contain independent random numbers on every grid point. Besides, we take **Neumann boundary conditions**: we require the spatial derivatives of the variables with respect to the normal vectors to be null on the boundaries of the domain $E$.

Let's define the four parameters of the model.


``` python
a = 2.8e-4
b = 5e-3
tau = .1
k = -.005
```


3. We discretize time and space. The following condition ensures that the discretization scheme we use here is stable:

$$dt \leq \frac{dx^2}{2}$$


``` python
size = 80  # size of the 2D grid
dx = 2./size  # space step
```



``` python
T = 10.0  # total time
dt = .9 * dx**2/2  # time step
n = int(T/dt)
```


4. We initialize the variables $u$ and $v$. The matrices $U$ and $V$ contain the values of these variables on the vertices of the 2D grid. These variables are initialized with a uniform noise between $0$ and $1$.


``` python
U = np.random.rand(size, size)
V = np.random.rand(size, size)
```


5. Now, we define a function that computes the discrete Laplace operator of a 2D variable on the grid, using a five-point stencil finite difference method. This operator is defined by:

$$\Delta u(x,y) \simeq \frac{u(x+h,y)+u(x-h,y)+u(x,y+h)+u(x,y-h)-4u(x,y)}{dx^2}$$

We can compute the values of this operator on the grid using vectorized matrix operations. Because of side effects on the edges of the matrix, we need to remove the borders of the grid in the computation.


``` python
def laplacian(Z):
    Ztop = Z[0:-2,1:-1]
    Zleft = Z[1:-1,0:-2]
    Zbottom = Z[2:,1:-1]
    Zright = Z[1:-1,2:]
    Zcenter = Z[1:-1,1:-1]
    return (Ztop + Zleft + Zbottom + Zright - 4 * Zcenter) / dx**2
```


6. Now, we simulate the system of equations using the finite difference method. At each time step, we compute the right-hand sides of the two equations on the grid using discrete spatial derivatives (Laplacians). Then, we update the variables using a discrete time derivative.


``` python
# We simulate the PDE with the finite difference method.
for i in range(n):
    # We compute the Laplacian of u and v.
    deltaU = laplacian(U)
    deltaV = laplacian(V)
    # We take the values of u and v inside the grid.
    Uc = U[1:-1,1:-1]
    Vc = V[1:-1,1:-1]
    # We update the variables.
    U[1:-1,1:-1], V[1:-1,1:-1] = \
        Uc + dt * (a * deltaU + Uc - Uc**3 - Vc + k), \
        Vc + dt * (b * deltaV + Uc - Vc) / tau
    # Neumann conditions: derivatives at the edges
    # are null.
    for Z in (U, V):
        Z[0,:] = Z[1,:]
        Z[-1,:] = Z[-2,:]
        Z[:,0] = Z[:,1]
        Z[:,-1] = Z[:,-2]
```


7. Finally, we display the variable $u$ after a time $T$ of simulation.


``` python
plt.imshow(U, cmap=plt.cm.copper, extent=[-1,1,-1,1]);
```


Whereas the variables when completely random at initialization time, we observe the formation of patterns after a sufficiently long simulation time.

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).

