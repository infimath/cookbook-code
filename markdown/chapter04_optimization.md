# optimization


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 4.1. Evaluating the time taken by a statement in IPython


``` python
n = 100000
```



``` python
%timeit sum([1. / i**2 for i in range(1, n)])
```



``` python
%%timeit s = 0.
for i in range(1, n):
    s += 1. / i**2
```



``` python
import numpy as np
```



``` python
%timeit np.sum(1. / np.arange(1., n) ** 2)
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 4.2. Profiling your code easily with cProfile and IPython

Standard imports.


``` python
import numpy as np
import matplotlib.pyplot as plt
```



``` python
%matplotlib inline
```


This function generates an array with random, uniformly distributed +1 and -1.


``` python
def step(*shape):
    # Create a random n-vector with +1 or -1 values.
    return 2 * (np.random.random_sample(shape) < .5) - 1
```


We simulate $n$ random walks, and look at the histogram of the walks over time.


``` python
%%prun -s cumulative -q -l 10 -T prun0
# We profile the cell, sort the report by "cumulative time",
# limit it to 10 lines, and save it to a file "prun0".
n = 10000
iterations = 50
x = np.cumsum(step(iterations, n), axis=0)
bins = np.arange(-30, 30, 1)
y = np.vstack([np.histogram(x[i,:], bins)[0] for i in range(iterations)])
```



``` python
print(open('prun0', 'r').read())
```


The most expensive functions are respectively `histogram` (37 ms), `rand` (19 ms), and `cumsum` (5 ms).

We plot the array `y`, representing the distribution of the particles over time.


``` python
plt.figure(figsize=(6,6));
plt.imshow(y, cmap='hot');
```


We now run the same code with 10 times more iterations.


``` python
%%prun -s cumulative -q -l 10 -T prun1
n = 10000
iterations = 500
x = np.cumsum(step(iterations, n), axis=0)
bins = np.arange(-30, 30, 1)
y = np.vstack([np.histogram(x[i,:], bins)[0] for i in range(iterations)])
```



``` python
print(open('prun1', 'r').read())
```


The most expensive functions are this time respectively `histogram` (566 ms), `cumsum` (388 ms) and `rand` (241 ms). `cumsum`'s execution time was negligible in the first case, whereas it is not in this case (due to the higher number of iterations).

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 4.3. Profiling your code line by line with line_profiler

Standard imports.


``` python
import numpy as np
```


After installing `line_profiler`, we can export the IPython extension.


``` python
%load_ext line_profiler
```


For `%lprun` to work, we need to encapsulate the code in a function, and to save it in a Python script..


``` python
%%writefile simulation.py
import numpy as np

def step(*shape):
    # Create a random n-vector with +1 or -1 values.
    return 2 * (np.random.random_sample(shape) < .5) - 1

def simulate(iterations, n=10000):
    s = step(iterations, n)
    x = np.cumsum(s, axis=0)
    bins = np.arange(-30, 30, 1)
    y = np.vstack([np.histogram(x[i,:], bins)[0] for i in range(iterations)])
    return y
```


Now, we need to execute this script to load the function in the interactive namespace.


``` python
import simulation
```


Let's execute the function under the control of the line profiler.


``` python
%lprun -T lprof0 -f simulation.simulate simulation.simulate(50)
```



``` python
print(open('lprof0', 'r').read())
```


Let's run the simulation with 10 times more iterations.


``` python
%lprun -T lprof1 -f simulation.simulate simulation.simulate(iterations=500)
```



``` python
print(open('lprof1', 'r').read())
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 4.4. Profiling the memory usage of your code with memory_profiler

Standard imports.


``` python
import numpy as np
```


After installing `memory_profiler`, we can export the IPython extension.


``` python
%load_ext memory_profiler
```


For `%lprun` to work, we need to encapsulate the code in a function, and to save it in a Python script.


``` python
%%writefile simulation.py
import numpy as np

def step(*shape):
    # Create a random n-vector with +1 or -1 values.
    return 2 * (np.random.random_sample(shape) < .5) - 1

def simulate(iterations, n=10000):
    s = step(iterations, n)
    x = np.cumsum(s, axis=0)
    bins = np.arange(-30, 30, 1)
    y = np.vstack([np.histogram(x[i,:], bins)[0] for i in range(iterations)])
    return y
```


Now, we need to execute this script to load the function in the interactive namespace.


``` python
import simulation
```


Let's execute the function under the control of the line profiler.


``` python
%mprun -T mprof0 -f simulation.simulate simulation.simulate(50)
```



``` python
print(open('mprof0', 'r').read())
```


Let's run the simulation with 10 times more iterations.


``` python
%mprun -T mprof1 -f simulation.simulate simulation.simulate(iterations=500)
```



``` python
print(open('mprof1', 'r').read())
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 4.5. Understanding the internals of NumPy to avoid unnecessary array copying


``` python
import numpy as np
```


## Inspect the memory address of arrays


``` python
def id(x):
    # This function returns the memory
    # block address of an array.
    return x.__array_interface__['data'][0]
```



``` python
a = np.zeros(10); aid = id(a); aid
```



``` python
b = a.copy(); id(b) == aid
```


## In-place and copy operations


``` python
a *= 2; id(a) == aid
```



``` python
c = a * 2; id(c) == aid
```


## Benchmarking

In-place operation.


``` python
%%timeit a = np.zeros(10000000)
a *= 2
```


With memory copy.


``` python
%%timeit a = np.zeros(10000000)
b = a * 2
```


## Reshaping an array: copy or not?


``` python
a = np.zeros((10, 10)); aid = id(a); aid
```


Reshaping an array while preserving its order does not trigger a copy.


``` python
b = a.reshape((1, -1)); id(b) == aid
```


Transposing an array changes its order so that a reshape triggers a copy.


``` python
c = a.T.reshape((1, -1)); id(c) == aid
```


To return a flattened version (1D) of a multidimensional array, one can use `flatten` or `ravel`. The former always return a copy, whereas the latter only makes a copy if necessary.


``` python
d = a.flatten(); id(d) == aid
```



``` python
e = a.ravel(); id(e) == aid
```



``` python
%timeit a.flatten()
```



``` python
%timeit a.ravel()
```


## Broadcasting

When performing operations on arrays with different shapes, you don't necessarily have to make copies to make their shapes match. Broadcasting rules allow you to make computations on arrays with different but compatible shapes. Two dimensions are compatible if they are equal or one of them is 1. If the arrays have different number of dimensions, dimensions are added to the smaller array from the trailing dimensions to the leading ones.


``` python
n = 1000
```



``` python
a = np.arange(n)
ac = a[:, np.newaxis]
ar = a[np.newaxis, :]
```



``` python
%timeit np.tile(ac, (1, n)) * np.tile(ar, (n, 1))
```



``` python
%timeit ar * ac
```


## Exercise

Can you explain the performance discrepancy between the following two similar operations?


``` python
a = np.random.rand(5000, 5000)
```



``` python
%timeit a[0, :].sum()
```



``` python
%timeit a[:, 0].sum()
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 4.6. Using stride tricks with NumPy

Every array has a number of dimensions, a shape, a data type, and strides. Strides are integer numbers describing, for each dimension, the byte step in the contiguous block of memory. The address of an item in the array is a linear combination of its indices: the coefficients are the strides.


``` python
import numpy as np
```



``` python
id = lambda x: x.__array_interface__['data'][0]
```



``` python
x = np.zeros(10); x.strides
```


This vector contains float64 (8 bytes) items: one needs to go 8 bytes forward to go from one item to the next.


``` python
y = np.zeros((10, 10)); y.strides
```


In the first dimension (vertical), one needs to go 80 bytes (10 float64 items) forward to go from one item to the next, because the items are internally stored in row-major order. In the second dimension (horizontal), one needs to go 8 bytes forward to go from one item to the next.

### Broadcasting revisited

We create a new array pointing to the same memory block as `a`, but with a different shape. The strides are such that this array looks like it is a vertically tiled version of `a`. NumPy is *tricked*: it thinks `b` is a 2D `n * n` array with `n^2` elements, whereas the data buffer really contains only `n` elements.


``` python
n = 1000; a = np.arange(n)
```



``` python
b = np.lib.stride_tricks.as_strided(a, (n, n), (0, 4))
```



``` python
b
```



``` python
b.size, b.shape, b.nbytes
```



``` python
%timeit b * b.T
```


This first version does not involve any copy, as `b` and `b.T` are arrays pointing to the same data buffer in memory, but with different strides.


``` python
%timeit np.tile(a, (n, 1)) * np.tile(a[:, np.newaxis], (1, n))
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 4.7. Implementing an efficient rolling average algorithm with stride tricks

Stride tricks can be useful for local computations on arrays, when the computed value at a given position depends on the neighbor values. Examples include dynamical systems, filters, cellular automata, and so on. In this example, we will implement an efficient rolling average (a particular type of convolution-based linear filter) with NumPy stride tricks.

The idea is to start from a 1D vector, and make a "virtual" 2D array where each line is a shifted version of the previous line. When using stride tricks, this process does not involve any copy, so it is efficient.


``` python
import numpy as np
from numpy.lib.stride_tricks import as_strided
%precision 0
```



``` python
def id(x):
    # This function returns the memory
    # block address of an array.
    return x.__array_interface__['data'][0]
```



``` python
n = 5; k = 2
```



``` python
a = np.linspace(1, n, n); aid = id(a)
```


Let's change the strides of `a` to add shifted rows.


``` python
as_strided(a, (k, n), (a.itemsize, a.itemsize))
```



``` python
id(a), id(as_strided(a, (k, n)))
```


The last value indicates an out-of-bounds problem: stride tricks can be dangerous as memory access is not checked. Here, we should take edge effects into account by limiting the shape of the array.


``` python
as_strided(a, (k, n - k + 1), (a.itemsize,)*2)
```


Let's apply this technique to calculate the rolling average of a random increasing signal.

First version using array copies.


``` python
def shift1(x, k):
    return np.vstack([x[i:n-k+i+1] for i in range(k)])
```


Second version using stride tricks.


``` python
def shift2(x, k):
    return as_strided(x, (k, n - k + 1), (8, 8))
```



``` python
b = shift1(a, k); b, id(b) == aid
```



``` python
c = shift2(a, k); c, id(c) == aid
```


Let's generate a signal.


``` python
n, k = 100, 10
t = np.linspace(0., 1., n)
x = t + .1 * np.random.randn(n)
```


We compute the signal rolling average by creating the shifted version of the signal, and averaging along the vertical dimension.


``` python
y = shift2(x, k)
x_avg = y.mean(axis=0)
```


Let's plot the signal and its averaged version.


``` python
%matplotlib inline
```



``` python
import matplotlib.pyplot as plt
```



``` python
f = plt.figure()
plt.plot(x[:-k+1], '-k');
plt.plot(x_avg, '-r');
```


### Benchmarks

Let's benchmark the first version (creation of the shifted array, and computation of the mean), which involves array copy.


``` python
%timeit shift1(x, k)
```



``` python
%%timeit y = shift1(x, k)
z = y.mean(axis=0)
```


And the second version, using stride tricks.


``` python
%timeit shift2(x, k)
```



``` python
%%timeit y = shift2(x, k)
z = y.mean(axis=0)
```


In the first version, most of the time is spent in the array copy, whereas in the stride trick version, most of the time is instead spent in the computation of the average.

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 4.8. Making efficient selections in arrays with NumPy


``` python
import numpy as np
```



``` python
id = lambda x: x.__array_interface__['data'][0]
```


We create a large array.


``` python
n, d = 100000, 100
```



``` python
a = np.random.random_sample((n, d)); aid = id(a)
```


## Array views and fancy indexing

We take a selection using two different methods: with a view and with fancy indexing.


``` python
b1 = a[::10]
b2 = a[np.arange(0, n, 10)]
```



``` python
np.array_equal(b1, b2)
```


The view refers to the original data buffer, whereas fancy indexing yields a copy.


``` python
id(b1) == aid, id(b2) == aid
```


Fancy indexing is several orders of magnitude slower as it involves copying a large array. Fancy indexing is more general as it allows to select any portion of an array (using any list of indices), not just a strided selection.


``` python
%timeit a[::10]
```



``` python
%timeit a[np.arange(0, n, 10)]
```


## Alternatives to fancy indexing: list of indices

Given a list of indices, there are two ways of selecting the corresponding sub-array: fancy indexing, or the np.take function.


``` python
i = np.arange(0, n, 10)
```



``` python
b1 = a[i]
b2 = np.take(a, i, axis=0)
```



``` python
np.array_equal(b1, b2)
```



``` python
%timeit a[i]
```



``` python
%timeit np.take(a, i, axis=0)
```


Using np.take instead of fancy indexing is faster.

**Note**: Performance of fancy indexing has been improved in recent versions of NumPy; this trick is especially useful on older versions of NumPy.

## Alternatives to fancy indexing: mask of booleans

Let's create a mask of booleans, where each value indicates whether the corresponding row needs to be selected in x.


``` python
i = np.random.random_sample(n) < .5
```


The selection can be made using fancy indexing or the np.compress function.


``` python
b1 = a[i]
b2 = np.compress(i, a, axis=0)
```



``` python
np.array_equal(b1, b2)
```



``` python
%timeit a[i]
```



``` python
%timeit np.compress(i, a, axis=0)
```


Once again, the alternative method to fancy indexing is faster.

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 4.9. Processing huge NumPy arrays with memory mapping


``` python
import numpy as np
```


## Writing a memory-mapped array

We create a memory-mapped array with a specific shape.


``` python
nrows, ncols = 1000000, 100
```



``` python
f = np.memmap('memmapped.dat', dtype=np.float32, 
              mode='w+', shape=(nrows, ncols))
```


Let's feed the array with random values, one column at a time because our system memory is limited!


``` python
for i in range(ncols):
    f[:,i] = np.random.rand(nrows)
```


We save the last column of the array.


``` python
x = f[:,-1]
```


Now, we flush memory changes to disk by removing the object.


``` python
del f
```


## Reading a memory-mapped file

Reading a memory-mapped array from disk involves the same memmap function but with a different file mode. The data type and the shape need to be specified again, as this information is not stored in the file.


``` python
f = np.memmap('memmapped.dat', dtype=np.float32, shape=(nrows, ncols))
```



``` python
np.array_equal(f[:,-1], x)
```



``` python
del f
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 4.10. Manipulating large arrays with HDF5 and PyTables


``` python
import numpy as np
import tables as tb
```


## Creating an HDF5 file

Let's create a new empty HDF5 file.


``` python
f = tb.open_file('myfile.h5', 'w')
```


We create a new top-level group named "experiment1".


``` python
f.create_group('/', 'experiment1')
```


Let's also add some metadata to this group.


``` python
f.set_node_attr('/experiment1', 'date', '2014-09-01')
```


In this group, we create a 1000*1000 array named "array1".


``` python
x = np.random.rand(1000, 1000)
f.create_array('/experiment1', 'array1', x)
```


Finally, we need to close the file to commit the changes on disk.


``` python
f.close()
```


## Reading a HDF5 file


``` python
f = tb.open_file('myfile.h5', 'r')
```


We can retrieve an attribute by giving the group path and the attribute name.


``` python
f.get_node_attr('/experiment1', 'date')
```


We can access any item in the file using attributes. IPython's tab completion is incredibly useful in this respect when exploring a file interactively.


``` python
y = f.root.experiment1.array1
type(y)
```


The array can be used as a NumPy array, but an important distinction is that it is stored on disk instead of system memory. Performing a computation on this array triggers a preliminary loading of the array in memory, so that it is more efficient to only access views on this array.


``` python
np.array_equal(x[0,:], y[0,:])
```


It is also possible to get a node from its absolute path, which is useful when this path is only known at runtime.


``` python
f.get_node('/experiment1/array1')
```



``` python
f.close()
```


Clean-up.


``` python
import os
os.remove('myfile.h5')
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 4.11. Manipulating large heterogeneous tables with HDF5 and PyTables


``` python
import numpy as np
import tables as tb
```


We create a new HDF5 file.


``` python
f = tb.open_file('myfile.h5', 'w')
```


We will create a HDF5 table with two columns: the name of a city (a string with 64 characters at most), and its population (a 32 bit integer).


``` python
dtype = np.dtype([('city', 'S64'), ('population', 'i4')])
```


Now, we create the table in '/table1'.


``` python
table = f.create_table('/', 'table1', dtype)
```


Let's add a few rows.


``` python
table.append([('Brussels', 1138854),
              ('London',   8308369),
              ('Paris',    2243833)])
```


After adding rows, we need to flush the table to commit the changes on disk.


``` python
table.flush()
```


Data can be obtained from the table with a lot of different ways in PyTables. The easiest but less efficient way is to load the entire table in memory, which returns a NumPy array.


``` python
table[:]
```


It is also possible to load a particular column (and all rows).


``` python
table.col('city')
```


When dealing with a large number of rows, we can make a SQL-like query in the table to load all rows that satisfy particular conditions.


``` python
[row['city'] for row in table.where('population>2e6')]
```


Finally, we can access particular rows knowing their indices.


``` python
table[1]
```


Clean-up.


``` python
f.close()
import os
os.remove('myfile.h5')
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).

