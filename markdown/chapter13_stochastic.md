# stochastic


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 13.1. Simulating a discrete-time Markov chain

1. Let's import NumPy and matplotlib.


``` python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
```


2. We consider a population that cannot comprise more than $N=100$ individuals. We also define birth and death rates.


``` python
N = 100  # maximum population size
a = .5/N  # birth rate
b = .5/N  # death rate
```


3. We will simulate a Markov chain on the finite space $\{0, 1, \ldots, N\}$. Each state represents a population size. The vector $x$ will contain the population size at each time step. We set the initial state to $x_0=25$, i.e. there are 25 individuals in the population at initialization time.


``` python
nsteps = 1000
x = np.zeros(nsteps)
x[0] = 25
```


4. We now simulate our chain. At each time step $t$, there is a new birth with probability $a \cdot x_t$, and independently, there is a new death with probability $b \cdot x_t$. These probabilities are proportional to the size of the population at that time. If the population size reaches $0$ or $N$, the evolution stops.


``` python
for t in range(nsteps - 1):
    if 0 < x[t] < N-1:
        # Is there a birth?
        birth = np.random.rand() <= a*x[t]
        # Is there a death?
        death = np.random.rand() <= b*x[t]
        # We update the population size.
        x[t+1] = x[t] + 1*birth - 1*death
    # The evolution stops if we reach $0$ or $N$.
    else:
        x[t+1] = x[t]
```


5. Let's look at the evolution of the population size.


``` python
plt.figure(figsize=(6,3));
plt.plot(x);
```


We see that, at every time, the population size can stays stable, increase by 1, or decrease by 1.

6. Now, we will simulate many independent trials of this Markov chain. We could run the previous simulation with a loop, but it would be very slow (two nested `for` loops). Instead, we *vectorize* the simulation by considering all independent trials at once. There is a single loop over time. At every time step, we update all trials simultaneously with vectorized operations on vectors. The vector `x` now contains the population size of all trials, at a particular time. At initialization time, the population sizes are set to random numbers between $0$ and $N$.


``` python
ntrials = 100
x = np.random.randint(size=ntrials,
                      low=0, high=N)
```


7. We define a function that performs the simulation. At every time step, we find the trials that undergo births and deaths by generating random vectors, and we update the population sizes with vector operations.


``` python
def simulate(x, nsteps):
    """Run the simulation."""
    for _ in range(nsteps - 1):
        # Which trials to update?
        upd = (0 < x) & (x < N-1)
        # In which trials do births occur?
        birth = 1*(np.random.rand(ntrials) <= a*x)
        # In which trials do deaths occur?
        death = 1*(np.random.rand(ntrials) <= b*x)
        # We update the population size for all trials.
        x[upd] += birth[upd] - death[upd]
```


8. Now, we will look at the histograms of the population size at different times. These histograms represent the probability distribution of the Markov chain, estimated with independent trials (Monte Carlo method).


``` python
bins = np.linspace(0, N, 25);
```



``` python
plt.figure(figsize=(12,3));
nsteps_list = [10, 1000, 10000]
for i, nsteps in enumerate(nsteps_list):
    plt.subplot(1, len(nsteps_list), i + 1);
    simulate(x, nsteps)
    plt.hist(x, bins=bins);
    plt.xlabel("Population size");
    if i == 0:
        plt.ylabel("Histogram");
    plt.title("{0:d} time steps".format(nsteps));
```


Whereas, initially, the population sizes look equally distributed between $0$ and $N$, they appear to converge to $0$ or $N$ after a sufficiently long time. This is because the states $0$ and $N$ are **absorbing**: once reached, the chain cannot leave those states. Furthermore, these states can be reached from any other state.

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 13.2. Simulating a Poisson process

1. Let's import NumPy and matplotlib.


``` python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
```


2. Let's specify the `rate`: the average number of events per second.


``` python
rate = 20.  # average number of events per second
```


3. First, we will simulate the process using small time bins of 1 millisecond.


``` python
dt = .001  # time step
n = int(1./dt)  # number of time steps
```


4. On every time bin, the probability that an event occurs is about $\textrm{rate} \times dt$, if $dt$ is small enough. Besides, since the Poisson process has no memory, the occurrence of an event is independent from one bin to another. Therefore, we can sample Bernoulli random variables in a vectorized way in order to simulate our process.


``` python
x = np.zeros(n)
x[np.random.rand(n) <= rate*dt] = 1
```


The vector `x` contains zeros and ones on all time bins, *one* corresponding to the occurrence of an event.


``` python
x[:10]
```


5. Let's display the simulated process. We draw a vertical line on each non-zero time bin.


``` python
plt.figure(figsize=(6,2));
plt.vlines(np.nonzero(x)[0], 0, 1);
plt.xticks([]); plt.yticks([]);
```


6. Another way of representing that same object consists in considering the associated **counting process** $N(t)$: the number of events that have occurred until time $t$. Here, we can display this process using the function `cumsum`.


``` python
plt.figure(figsize=(6,4));
plt.plot(np.linspace(0., 1., n), np.cumsum(x));
plt.xlabel("Time");
plt.ylabel("Counting process");
```


7. The other (and more efficient) way of simulating the homogeneous Poisson process is to use the property that the time interval between two successive events is an exponential random variable. Furthermore, these intervals are independent, so that we can sample these intervals in a vectorized way. Finally, we get our process by summing cumulatively all those intervals.


``` python
y = np.cumsum(np.random.exponential(1./rate, size=int(rate)))
```


The vector `y` contains another realization of our Poisson process, but the data structure is different. Every component of the vector is the time of an event.


``` python
y[:10]
```


8. Finally, let's display the simulated process.


``` python
plt.figure(figsize=(6,2));
plt.vlines(y, 0, 1);
plt.xticks([]); plt.yticks([]);
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 13.3. Simulating a Brownian motion

1. Let's import NumPy and matplotlib.


``` python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
```


2. We will simulate Brownian motions with 5000 time steps.


``` python
n = 5000
```


3. We simulate two independent one-dimensional Brownian processes to form a single two-dimensional Brownian process. The (discrete) Brownian motion makes independent Gaussian jumps at each time step (like a random walk). Therefore, we just have to compute the cumulative sum of independent normal random variables (one for each time step).


``` python
x = np.cumsum(np.random.randn(n))
y = np.cumsum(np.random.randn(n))
```


4. Now, to display the Brownian motion, we could just do `plot(x, y)`. However, the result would be monochromatic and a bit boring. We would like to use a gradient of color to illustrate the progression of the motion in time. Matplotlib forces us to use a small hack based on `scatter`. This function allows us to assign a different color to each point at the expense of dropping out line segments between points. To work around this issue, we interpolate linearly the process to give the illusion of a continuous line.


``` python
k = 10  # We add 10 intermediary points between two 
        # successive points.
# We interpolate x and y.
x2 = np.interp(np.arange(n*k), np.arange(n)*k, x)
y2 = np.interp(np.arange(n*k), np.arange(n)*k, y)
```



``` python
# Now, we draw our points with a gradient of colors.
plt.scatter(x2, y2, c=range(n*k), linewidths=0,
            marker='o', s=3, cmap=plt.cm.jet,)
plt.axis('equal');
plt.xticks([]); plt.yticks([]);
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 13.4. Simulating a stochastic differential equation

1. Let's import NumPy and matplotlib.


``` python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
```


2. Let's define a few parameters for our model.


``` python
sigma = 1.  # Standard deviation.
mu = 10.  # Mean.
tau = .05  # Time constant.
```


3. We also define a few simulation parameters.


``` python
dt = .001  # Time step.
T = 1.  # Total time.
n = int(T/dt)  # Number of time steps.
t = np.linspace(0., T, n)  # Vector of times.
```


4. We also define renormalized variables (to avoid recomputing these constants at every time step).


``` python
sigma_bis = sigma * np.sqrt(2. / tau)
sqrtdt = np.sqrt(dt)
```


5. We create a vector that will contain all successive values of our process during the simulation.


``` python
x = np.zeros(n)
```


6. Now, we simulate the process with the Euler-Maruyama method. It is really like the standard Euler method for ODEs, but with an extra stochastic term (which is just a scaled normal random variable). We will give the equation of the process along with the details of this method in *How it works...*.


``` python
for i in range(n-1):
    x[i+1] = x[i] + dt*(-(x[i]-mu)/tau) + \
             sigma_bis * sqrtdt * np.random.randn()
```


7. Let's display the evolution of the process.


``` python
plt.figure(figsize=(6,3));
plt.plot(t, x);
```


8. Now, we are going to take a look at the time evolution of the distribution of the process. To do that, we will simulate many independent realizations of the same process in a vectorized way. We define a vector `X` that will contain all realizations of the process at a given time (i.e. we do not keep the memory of all realizations at all times). This vector will be completely updated at every time step. We will show the estimated distribution (histograms) at several points in time.


``` python
ntrials = 10000
X = np.zeros(ntrials)
```



``` python
# We create bins for the histograms.
bins = np.linspace(-2., 14., 100);
plt.figure(figsize=(6,3));
for i in range(n):
    # We update the process independently for all trials.
    X += dt*(-(X-mu)/tau) + \
        sigma_bis*sqrtdt*np.random.randn(ntrials)
    # We display the histogram for a few points in time.
    if i in (5, 50, 900):
        hist, _ = np.histogram(X, bins=bins)
        plt.plot((bins[1:]+bins[:-1])/2, hist,
                 {5: '-', 50: '.', 900: '-.',}[i],
                 label="t={0:.2f}".format(i*dt));
    plt.legend();
```


The distribution of the process tends to a Gaussian distribution with mean $\mu=10$ and standard deviation $\sigma=1$. The process would be stationary if the initial distribution was also a Gaussian with the adequate parameters.

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).

