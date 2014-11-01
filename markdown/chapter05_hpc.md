# hpc


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 5.1. Accelerating pure Python code with Numba and Just-In-Time compilation

In this example, we first write a pure Python version of a function that generates a Mandelbrot fractal. Then, we use Numba to compile it dynamically to native code.


``` python
import numpy as np
```


We initialize the simulation and generate the grid
in the complex plane.


``` python
size = 200
iterations = 100
```


## Pure Python version

The following function generates the fractal.


``` python
def mandelbrot_python(m, size, iterations):
    for i in range(size):
        for j in range(size):
            c = -2 + 3./size*j + 1j*(1.5-3./size*i)
            z = 0
            for n in range(iterations):
                if np.abs(z) <= 10:
                    z = z*z + c
                    m[i, j] = n
                else:
                    break
```



``` python
m = np.zeros((size, size))
mandelbrot_python(m, size, iterations)
```



``` python
import matplotlib.pyplot as plt
%matplotlib inline
plt.imshow(np.log(m), cmap=plt.cm.hot,);
plt.xticks([]); plt.yticks([]);
```



``` python
%%timeit m = np.zeros((size, size))
mandelbrot_python(m, size, iterations)
```


## Numba version

We first import Numba.


``` python
import numba
from numba import jit, complex128
```


Now, we just add the `@jit` decorator to the exact same function.


``` python
@jit(locals=dict(c=complex128, z=complex128))
def mandelbrot_numba(m, size, iterations):
    for i in range(size):
        for j in range(size):
            c = -2 + 3./size*j + 1j*(1.5-3./size*i)
            z = 0
            for n in range(iterations):
                if abs(z) <= 10:
                    z = z*z + c
                    m[i, j] = n
                else:
                    break
```



``` python
m = np.zeros((size, size))
mandelbrot_numba(m, size, iterations)
```



``` python
%%timeit m = np.zeros((size, size))
mandelbrot_numba(m, size, iterations)
```


The Numba version is 250 times faster than the pure Python version here!

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 5.2. Accelerating array computations with Numexpr

Let's import NumPy and Numexpr.


``` python
import numpy as np
import numexpr as ne
```


We generate three large vectors.


``` python
x, y, z = np.random.rand(3, 1000000)
```


Now, we evaluate the time taken by NumPy to calculate a complex algebraic expression involving our vectors.


``` python
%timeit x + (y**2 + (z*x + 1)*3)
```


And now, the same calculation performed by Numexpr. We need to give the formula as a string as Numexpr will parse it and compile it.


``` python
%timeit ne.evaluate('x + (y**2 + (z*x + 1)*3)')
```


Numexpr also makes use of multicore processors. Here, we have 4 physical cores and 8 virtual threads with hyperthreading. We can specify how many cores we want numexpr to use.


``` python
ne.ncores
```



``` python
for i in range(1, 5):
    ne.set_num_threads(i)
    %timeit ne.evaluate('x + (y**2 + (z*x + 1)*3)')
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 5.3. Wrapping a C library in Python with ctypes

This example shows:

  * How to write and compile C code defining functions that are accessible from Python, and
  * How to call C functions from Python using the native ctypes module.
  
This notebook has been written for Windows systems and Microsoft's C compiler (shipped with Visual Studio).

**Note**: on Windows, for the C compiler to run, you need to execute a sequence of magic incantations before launching the IPython notebook. See the `_launch_notebook.bat` file in this repository.

Let's write the generation of the Mandelbrot fractal in C.


``` python
%%writefile mandelbrot.c

// Needed when creating a DLL.
#define EXPORT __declspec(dllexport)

#include "stdio.h"
#include "stdlib.h"

// This function will be available in the DLL.
EXPORT void __stdcall mandelbrot(int size,
                                 int iterations,
                                 int *col) 
{
    // Variable declarations.
    int i, j, n, index;
    double cx, cy;
    double z0, z1, z0_tmp, z0_2, z1_2;
    
    // Loop within the grid.
    for (i = 0; i < size; i++)
    {
        cy = -1.5 + (double)i / size * 3;
        for (j = 0; j < size; j++)
        {
            // We initialize the loop of the system.
            cx = -2.0 + (double)j / size * 3;
            index = i * size + j;
            // Let's run the system.
            z0 = 0.0;
            z1 = 0.0;
            for (n = 0; n < iterations; n++)
            {
                z0_2 = z0 * z0;
                z1_2 = z1 * z1;
                if (z0_2 + z1_2 <= 100)
                {
                    // Update the system.
                    z0_tmp = z0_2 - z1_2 + cx;
                    z1 = 2 * z0 * z1 + cy;
                    z0 = z0_tmp;
                    col[index] = n;
                }
                else
                {
                    break;
                }
            }
        }
    }
}
```


Now, let's build this C source file into a DLL with Microsoft Visual Studio's `cl.exe`. The `/LD` option specifies that a DLL has to be created.


``` python
!cl /LD mandelbrot.c
```


## Wrapping the C library with NumPy and ctypes

Let's access the library with ctypes.


``` python
import ctypes
```



``` python
lb = ctypes.CDLL('mandelbrot.dll')
```



``` python
lib = ctypes.WinDLL(None, handle=lb._handle)
```



``` python
# Access the mandelbrot function.
mandelbrot = lib.mandelbrot
```


NumPy and ctypes allow us to wrap the C function defined in the DLL.


``` python
from numpy.ctypeslib import ndpointer
```



``` python
# Define the types of the output and arguments of this function.
mandelbrot.restype = None
mandelbrot.argtypes = [ctypes.c_int, ctypes.c_int,
                       ndpointer(ctypes.c_int)]
```


Now, we can execute the mandelbrot function.


``` python
import numpy as np
# We initialize an empty array.
size = 200
iterations = 100
col = np.empty((size, size), dtype=np.int32)
# We execute the C function, which will update the array.
mandelbrot(size, iterations, col)
```


The simulation has finished, let's display the fractal.


``` python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```



``` python
plt.imshow(np.log(col), cmap=plt.cm.hot,);
plt.xticks([]);
plt.yticks([]);
```



``` python
%timeit mandelbrot(size, iterations, col)
```


We free the library handle at the end.


``` python
lb._handle
```



``` python
from ctypes.wintypes import HMODULE
ctypes.windll.kernel32.FreeLibrary.argtypes = [HMODULE]
ctypes.windll.kernel32.FreeLibrary(lb._handle);
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 5.4. Accelerating Python code with Cython

We use Cython to accelerate the generation of the Mandelbrot fractal.


``` python
import numpy as np
```


We initialize the simulation and generate the grid
in the complex plane.


``` python
size = 200
iterations = 100
```


## Pure Python


``` python
def mandelbrot_python(m, size, iterations):
    for i in range(size):
        for j in range(size):
            c = -2 + 3./size*j + 1j*(1.5-3./size*i)
            z = 0
            for n in range(iterations):
                if np.abs(z) <= 10:
                    z = z*z + c
                    m[i, j] = n
                else:
                    break
```



``` python
%%timeit -n1 -r1 m = np.zeros((size, size))
mandelbrot_python(m, size, iterations)
```


## Cython versions

We first import Cython.


``` python
%load_ext cythonmagic
```


### Take 1

First, we just add the %%cython magic.


``` python
%%cython -a
import numpy as np

def mandelbrot_cython(m, size, iterations):
    for i in range(size):
        for j in range(size):
            c = -2 + 3./size*j + 1j*(1.5-3./size*i)
            z = 0
            for n in range(iterations):
                if np.abs(z) <= 10:
                    z = z*z + c
                    m[i, j] = n
                else:
                    break
```



``` python
%%timeit -n1 -r1 m = np.zeros((size, size), dtype=np.int32)
mandelbrot_cython(m, size, iterations)
```


Virtually no speedup.

### Take 2

Now, we add type information, using memory views for NumPy arrays.


``` python
%%cython -a
import numpy as np

def mandelbrot_cython(int[:,::1] m, 
                      int size, 
                      int iterations):
    cdef int i, j, n
    cdef complex z, c
    for i in range(size):
        for j in range(size):
            c = -2 + 3./size*j + 1j*(1.5-3./size*i)
            z = 0
            for n in range(iterations):
                if z.real**2 + z.imag**2 <= 100:
                    z = z*z + c
                    m[i, j] = n
                else:
                    break
```



``` python
%%timeit -n1 -r1 m = np.zeros((size, size), dtype=np.int32)
mandelbrot_cython(m, size, iterations)
```


Interesting speedup!

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 5.5. Ray tracing: pure Python

In this example, we will render a sphere with a diffuse and specular material. The principle is to model a scene with a light source and a camera, and use the physical properties of light propagation to calculate the light intensity and color of every pixel of the screen.


``` python
import numpy as np
import matplotlib.pyplot as plt
```



``` python
%matplotlib inline
```



``` python
w, h = 200, 200  # Size of the screen in pixels.

def normalize(x):
    # This function normalizes a vector.
    x /= np.linalg.norm(x)
    return x

def intersect_sphere(O, D, S, R):
    # Return the distance from O to the intersection 
    # of the ray (O, D) with the sphere (S, R), or 
    # +inf if there is no intersection.
    # O and S are 3D points, D (direction) is a 
    # normalized vector, R is a scalar.
    a = np.dot(D, D)
    OS = O - S
    b = 2 * np.dot(D, OS)
    c = np.dot(OS, OS) - R*R
    disc = b*b - 4*a*c
    if disc > 0:
        distSqrt = np.sqrt(disc)
        q = (-b - distSqrt) / 2.0 if b < 0 \
            else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return np.inf

def trace_ray(O, D):
    # Find first point of intersection with the scene.
    t = intersect_sphere(O, D, position, radius)
    # No intersection?
    if t == np.inf:
        return
    # Find the point of intersection on the object.
    M = O + D * t
    N = normalize(M - position)
    toL = normalize(L - M)
    toO = normalize(O - M)
    # Ambient light.
    col = ambient
    # Lambert shading (diffuse).
    col += diffuse * max(np.dot(N, toL), 0) * color
    # Blinn-Phong shading (specular).
    col += specular_c * color_light * \
        max(np.dot(N, normalize(toL + toO)), 0) \
           ** specular_k
    return col

def run():
    img = np.zeros((h, w, 3))
    # Loop through all pixels.
    for i, x in enumerate(np.linspace(-1., 1., w)):
        for j, y in enumerate(np.linspace(-1., 1., h)):
            # Position of the pixel.
            Q[0], Q[1] = x, y
            # Direction of the ray going through the optical center.
            D = normalize(Q - O)
            # Launch the ray and get the color of the pixel.
            col = trace_ray(O, D)
            if col is None:
                continue
            img[h - j - 1, i, :] = np.clip(col, 0, 1)
    return img

# Sphere properties.
position = np.array([0., 0., 1.])
radius = 1.
color = np.array([0., 0., 1.])
diffuse = 1.
specular_c = 1.
specular_k = 50

# Light position and color.
L = np.array([5., 5., -10.])
color_light = np.ones(3)
ambient = .05

# Camera.
O = np.array([0., 0., -1.])  # Position.
Q = np.array([0., 0., 0.])  # Pointing to.

img = run()
plt.imshow(img);
plt.xticks([]); plt.yticks([]);
```



``` python
%timeit run()
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 5.5. Ray tracing: naive Cython

In this example, we will render a sphere with a diffuse and specular material. The principle is to model a scene with a light source and a camera, and use the physical properties of light propagation to calculate the light intensity and color of every pixel of the screen.


``` python
import numpy as np
import matplotlib.pyplot as plt
```



``` python
%matplotlib inline
```



``` python
%load_ext cythonmagic
```



``` python
%%cython
import numpy as np
cimport numpy as np

w, h = 200, 200  # Size of the screen in pixels.

def normalize(x):
    # This function normalizes a vector.
    x /= np.linalg.norm(x)
    return x

def intersect_sphere(O, D, S, R):
    # Return the distance from O to the intersection 
    # of the ray (O, D) with the sphere (S, R), or 
    # +inf if there is no intersection.
    # O and S are 3D points, D (direction) is a 
    # normalized vector, R is a scalar.
    a = np.dot(D, D)
    OS = O - S
    b = 2 * np.dot(D, OS)
    c = np.dot(OS, OS) - R*R
    disc = b*b - 4*a*c
    if disc > 0:
        distSqrt = np.sqrt(disc)
        q = (-b - distSqrt) / 2.0 if b < 0 \
            else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return np.inf

def trace_ray(O, D):
    # Find first point of intersection with the scene.
    t = intersect_sphere(O, D, position, radius)
    # No intersection?
    if t == np.inf:
        return
    # Find the point of intersection on the object.
    M = O + D * t
    N = normalize(M - position)
    toL = normalize(L - M)
    toO = normalize(O - M)
    # Ambient light.
    col = ambient
    # Lambert shading (diffuse).
    col += diffuse * max(np.dot(N, toL), 0) * color
    # Blinn-Phong shading (specular).
    col += specular_c * color_light * \
        max(np.dot(N, normalize(toL + toO)), 0) \
           ** specular_k
    return col

def run():
    img = np.zeros((h, w, 3))
    # Loop through all pixels.
    for i, x in enumerate(np.linspace(-1., 1., w)):
        for j, y in enumerate(np.linspace(-1., 1., h)):
            # Position of the pixel.
            Q[0], Q[1] = x, y
            # Direction of the ray going through the optical center.
            D = normalize(Q - O)
            depth = 0
            # Launch the ray and get the color of the pixel.
            col = trace_ray(O, D)
            if col is None:
                continue
            img[h - j - 1, i, :] = np.clip(col, 0, 1)
    return img

# Sphere properties.
position = np.array([0., 0., 1.])
radius = 1.
color = np.array([0., 0., 1.])
diffuse = 1.
specular_c = 1.
specular_k = 50

# Light position and color.
L = np.array([5., 5., -10.])
color_light = np.ones(3)
ambient = .05

# Camera.
O = np.array([0., 0., -1.])  # Position.
Q = np.array([0., 0., 0.])  # Pointing to.
```



``` python
img = run()
plt.imshow(img);
plt.xticks([]); plt.yticks([]);
```



``` python
%timeit -n1 -r1 run()
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 5.5. Ray tracing: Cython array buffers

In this example, we will render a sphere with a diffuse and specular material. The principle is to model a scene with a light source and a camera, and use the physical properties of light propagation to calculate the light intensity and color of every pixel of the screen.


``` python
import numpy as np
import matplotlib.pyplot as plt
```



``` python
%matplotlib inline
```



``` python
%load_ext cythonmagic
```


## Take 1


``` python
%%cython
import numpy as np
cimport numpy as np
from numpy import dot
from libc.math cimport sqrt

DBL = np.double
ctypedef np.double_t DBL_C
INT = np.int
ctypedef np.int_t INT_C
cdef INT_C w, h

w, h = 200, 200  # Size of the screen in pixels.

def normalize(np.ndarray[DBL_C, ndim=1] x):
    # This function normalizes a vector.
    x /= np.linalg.norm(x)
    return x

def intersect_sphere(np.ndarray[DBL_C, ndim=1] O, np.ndarray[DBL_C, ndim=1] D, 
                     np.ndarray[DBL_C, ndim=1] S, DBL_C R):
    # Return the distance from O to the intersection 
    # of the ray (O, D) with the sphere (S, R), or 
    # +inf if there is no intersection.
    # O and S are 3D points, D (direction) is a 
    # normalized vector, R is a scalar.
    
    cdef DBL_C a, b, c, disc, distSqrt, q, t0, t1
    cdef np.ndarray[DBL_C, ndim=1] OS
    
    a = dot(D, D)
    OS = O - S
    b = 2 * dot(D, OS)
    c = dot(OS, OS) - R*R
    disc = b*b - 4*a*c
    if disc > 0:
        distSqrt = np.sqrt(disc)
        q = (-b - distSqrt) / 2.0 if b < 0 \
            else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return np.inf

def trace_ray(np.ndarray[DBL_C, ndim=1] O, np.ndarray[DBL_C, ndim=1] D,
               np.ndarray[DBL_C, ndim=1] position,
               np.ndarray[DBL_C, ndim=1] color,
               np.ndarray[DBL_C, ndim=1] L,
               np.ndarray[DBL_C, ndim=1] color_light):
        
    cdef DBL_C t
    cdef np.ndarray[DBL_C, ndim=1] M, N, toL, toO, col
    
    # Find first point of intersection with the scene.
    t = intersect_sphere(O, D, position, radius)
    # No intersection?
    if t == np.inf:
        return
    # Find the point of intersection on the object.
    M = O + D * t
    N = normalize(M - position)
    toL = normalize(L - M)
    toO = normalize(O - M)
    # Ambient light.
    col = ambient * np.ones(3)
    # Lambert shading (diffuse).
    col += diffuse * max(dot(N, toL), 0) * color
    # Blinn-Phong shading (specular).
    col += specular_c * color_light * \
        max(dot(N, normalize(toL + toO)), 0) \
           ** specular_k
    return col

def run():
    cdef np.ndarray[DBL_C, ndim=3] img
    img = np.zeros((h, w, 3))
    cdef INT_C i, j
    cdef DBL_C x, y
    cdef np.ndarray[DBL_C, ndim=1] O, Q, D, col, position, color, L, color_light

    # Sphere properties.
    position = np.array([0., 0., 1.])
    color = np.array([0., 0., 1.])
    L = np.array([5., 5., -10.])
    color_light = np.ones(3)
        
    # Camera.
    O = np.array([0., 0., -1.])  # Position.
    Q = np.array([0., 0., 0.])  # Pointing to.
        
    # Loop through all pixels.
    for i, x in enumerate(np.linspace(-1., 1., w)):
        for j, y in enumerate(np.linspace(-1., 1., h)):
            # Position of the pixel.
            Q[0], Q[1] = x, y
            # Direction of the ray going through the optical center.
            D = normalize(Q - O)
            # Launch the ray and get the color of the pixel.
            col = trace_ray(O, D, position, color, L, color_light)
            if col is None:
                continue
            img[h - j - 1, i, :] = np.clip(col, 0, 1)
    return img

cdef DBL_C radius, ambient, diffuse, specular_k, specular_c

# Sphere and light properties.
radius = 1.
diffuse = 1.
specular_c = 1.
specular_k = 50.
ambient = .05       
```



``` python
img = run()
plt.imshow(img);
plt.xticks([]); plt.yticks([]);
```



``` python
%timeit -n1 -r1 run()
```


## Take 2

In this version, we rewrite normalize in pure C.


``` python
%%cython
import numpy as np
cimport numpy as np
from numpy import dot
from libc.math cimport sqrt

DBL = np.double
ctypedef np.double_t DBL_C
INT = np.int
ctypedef np.int_t INT_C
cdef INT_C w, h

w, h = 200, 200  # Size of the screen in pixels.

# normalize is now a pure C function that does not make
# use NumPy for the computations
cdef normalize(np.ndarray[DBL_C, ndim=1] x):
    cdef DBL_C n
    n = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
    x[0] /= n
    x[1] /= n
    x[2] /= n
    return x

def intersect_sphere(np.ndarray[DBL_C, ndim=1] O, np.ndarray[DBL_C, ndim=1] D, 
                     np.ndarray[DBL_C, ndim=1] S, DBL_C R):
    # Return the distance from O to the intersection 
    # of the ray (O, D) with the sphere (S, R), or 
    # +inf if there is no intersection.
    # O and S are 3D points, D (direction) is a 
    # normalized vector, R is a scalar.
    
    cdef DBL_C a, b, c, disc, distSqrt, q, t0, t1
    cdef np.ndarray[DBL_C, ndim=1] OS
    
    a = dot(D, D)
    OS = O - S
    b = 2 * dot(D, OS)
    c = dot(OS, OS) - R*R
    disc = b*b - 4*a*c
    if disc > 0:
        distSqrt = np.sqrt(disc)
        q = (-b - distSqrt) / 2.0 if b < 0 \
            else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return np.inf

def trace_ray(np.ndarray[DBL_C, ndim=1] O, np.ndarray[DBL_C, ndim=1] D,
               np.ndarray[DBL_C, ndim=1] position,
               np.ndarray[DBL_C, ndim=1] color,
               np.ndarray[DBL_C, ndim=1] L,
               np.ndarray[DBL_C, ndim=1] color_light):
        
    cdef DBL_C t
    cdef np.ndarray[DBL_C, ndim=1] M, N, toL, toO, col
    
    # Find first point of intersection with the scene.
    t = intersect_sphere(O, D, position, radius)
    # No intersection?
    if t == np.inf:
        return
    # Find the point of intersection on the object.
    M = O + D * t
    N = normalize(M - position)
    toL = normalize(L - M)
    toO = normalize(O - M)
    # Ambient light.
    col = ambient * np.ones(3)
    # Lambert shading (diffuse).
    col += diffuse * max(dot(N, toL), 0) * color
    # Blinn-Phong shading (specular).
    col += specular_c * color_light * \
        max(dot(N, normalize(toL + toO)), 0) \
           ** specular_k
    return col

def run():
    cdef np.ndarray[DBL_C, ndim=3] img
    img = np.zeros((h, w, 3))
    cdef INT_C i, j
    cdef DBL_C x, y
    cdef np.ndarray[DBL_C, ndim=1] O, Q, D, col, position, color, L, color_light

    # Sphere properties.
    position = np.array([0., 0., 1.])
    color = np.array([0., 0., 1.])
    L = np.array([5., 5., -10.])
    color_light = np.ones(3)
        
    # Camera.
    O = np.array([0., 0., -1.])  # Position.
    Q = np.array([0., 0., 0.])  # Pointing to.
        
    # Loop through all pixels.
    for i, x in enumerate(np.linspace(-1., 1., w)):
        for j, y in enumerate(np.linspace(-1., 1., h)):
            # Position of the pixel.
            Q[0], Q[1] = x, y
            # Direction of the ray going through the optical center.
            D = normalize(Q - O)
            # Launch the ray and get the color of the pixel.
            col = trace_ray(O, D, position, color, L, color_light)
            if col is None:
                continue
            img[h - j - 1, i, :] = np.clip(col, 0, 1)
    return img

cdef DBL_C radius, ambient, diffuse, specular_k, specular_c

# Sphere and light properties.
radius = 1.
diffuse = 1.
specular_c = 1.
specular_k = 50.
ambient = .05       
```



``` python
img = run()
plt.imshow(img);
plt.xticks([]); plt.yticks([]);
```



``` python
%timeit -n1 -r1 run()
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 5.5. Ray tracing: Cython with tuples


``` python
import numpy as np
import matplotlib.pyplot as plt
```



``` python
%matplotlib inline
```



``` python
import cython
```



``` python
%load_ext cythonmagic
```


We don't use NumPy anymore for computations on 3D vectors, we use tuples which have less overhead for small vectors. We need to reimplement all element-wise computations with tuples, but that's fine since our vectors always have 3 elements only.


``` python
%%cython
import numpy as np
cimport numpy as np
DBL = np.double
ctypedef np.double_t DBL_C
from libc.math cimport sqrt

cdef int w, h
w, h = 200, 200

cdef dot(tuple x, tuple y):
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]

cdef normalize(tuple x):
    cdef double n
    n = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
    return (x[0] / n, x[1] / n, x[2] / n)

cdef max(double x, double y):
    return x if x > y else y

cdef min(double x, double y):
    return x if x < y else y

cdef clip_(double x, double m, double M):
    return min(max(x, m), M)

cdef clip(tuple x, double m, double M):
    return (clip_(x[0], m, M), clip_(x[1], m, M), clip_(x[2], m, M),)

cdef add(tuple x, tuple y):
    return (x[0] + y[0], x[1] + y[1], x[2] + y[2])

cdef subtract(tuple x, tuple y):
    return (x[0] - y[0], x[1] - y[1], x[2] - y[2])

cdef minus(tuple x):
    return (-x[0], -x[1], -x[2])

cdef multiply(tuple x, tuple y):
    return (x[0] * y[0], x[1] * y[1], x[2] * y[2])
    
cdef multiply_s(tuple x, double c):
    return (x[0] * c, x[1] * c, x[2] * c)
    
cdef intersect_sphere(tuple O, 
                      tuple D, 
                      tuple S, 
                      double R):
    # Return the distance from O to the intersection of the ray (O, D) with the 
    # sphere (S, R), or +inf if there is no intersection.
    # O and S are 3D points, D (direction) is a normalized vector, R is a scalar.
    cdef double a, b, c, disc, distSqrt, q, t0, t1
    cdef tuple OS
    
    a = dot(D, D)
    OS = subtract(O, S)
    b = 2 * dot(D, OS)
    c = dot(OS, OS) - R * R
    disc = b * b - 4 * a * c
    if disc > 0:
        distSqrt = sqrt(disc)
        q = (-b - distSqrt) / 2.0 if b < 0 else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return float('inf')

cdef trace_ray(tuple O, tuple D,):
    
    cdef double t, radius, diffuse, specular_k, specular_c, DF, SP
    cdef tuple M, N, L, toL, toO, col_ray, \
        position, color, color_light, ambient

    # Sphere properties.
    position = (0., 0., 1.)
    radius = 1.
    color = (0., 0., 1.)
    diffuse = 1.
    specular_c = 1.
    specular_k = 50.
    
    # Light position and color.
    L = (5., 5., -10.)
    color_light = (1., 1., 1.)
    ambient = (.05, .05, .05)
    
    # Find first point of intersection with the scene.
    t = intersect_sphere(O, D, position, radius)
    # Return None if the ray does not intersect any object.
    if t == float('inf'):
        return
    # Find the point of intersection on the object.
    M = (O[0] + D[0] * t, O[1] + D[1] * t, O[2] + D[2] * t)
    N = normalize(subtract(M, position))
    toL = normalize(subtract(L, M))
    toO = normalize(subtract(O, M))
    DF = diffuse * max(dot(N, toL), 0)
    SP = specular_c * max(dot(N, normalize(add(toL, toO))), 0) ** specular_k
    
    return add(ambient, add(multiply_s(color, DF), multiply_s(color_light, SP)))

def run():
    cdef DBL_C[:,:,:] img = np.zeros((h, w, 3))
    cdef tuple img_
    cdef int i, j
    cdef double x, y
    cdef tuple O, Q, D, col_ray
        
    # Camera.
    O = (0., 0., -1.)  # Position.
        
    # Loop through all pixels.
    for i in range(w):
        for j in range(h):
            x = -1. + 2*float(i)/w
            y = -1. + 2*float(j)/h
            Q = (x, y, 0.)
            D = normalize(subtract(Q, O))
            col_ray = trace_ray(O, D)
            if col_ray is None:
                continue
            img_ = clip(col_ray, 0., 1.)
            img[h - j - 1, i, 0] = img_[0]
            img[h - j - 1, i, 1] = img_[1]
            img[h - j - 1, i, 2] = img_[2]
    return img
```



``` python
img = run()
plt.imshow(img);
plt.xticks([]); plt.yticks([]);
```



``` python
%timeit run()
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 5.5. Ray tracing: Cython with structs


``` python
import numpy as np
import matplotlib.pyplot as plt
```



``` python
%matplotlib inline
```



``` python
import cython
```



``` python
%load_ext cythonmagic
```


We now use a pure C structure to represent a 3D vector. We also implement all operations we need by hand in pure C.


``` python
%%cython
cimport cython
import numpy as np
cimport numpy as np
DBL = np.double
ctypedef np.double_t DBL_C
from libc.math cimport sqrt

cdef int w, h

cdef struct Vec3:
    double x, y, z
        
cdef Vec3 vec3(double x, double y, double z):
    cdef Vec3 v
    v.x = x
    v.y = y
    v.z = z
    return v

cdef double dot(Vec3 x, Vec3 y):
    return x.x * y.x + x.y * y.y + x.z * y.z

cdef Vec3 normalize(Vec3 x):
    cdef double n
    n = sqrt(x.x * x.x + x.y * x.y + x.z * x.z)
    return vec3(x.x / n, x.y / n, x.z / n)

cdef double max(double x, double y):
    return x if x > y else y

cdef double min(double x, double y):
    return x if x < y else y

cdef double clip_(double x, double m, double M):
    return min(max(x, m), M)

cdef Vec3 clip(Vec3 x, double m, double M):
    return vec3(clip_(x.x, m, M), clip_(x.y, m, M), clip_(x.z, m, M),)

cdef Vec3 add(Vec3 x, Vec3 y):
    return vec3(x.x + y.x, x.y + y.y, x.z + y.z)

cdef Vec3 subtract(Vec3 x, Vec3 y):
    return vec3(x.x - y.x, x.y - y.y, x.z - y.z)

cdef Vec3 minus(Vec3 x):
    return vec3(-x.x, -x.y, -x.z)

cdef Vec3 multiply(Vec3 x, Vec3 y):
    return vec3(x.x * y.x, x.y * y.y, x.z * y.z)
    
cdef Vec3 multiply_s(Vec3 x, double c):
    return vec3(x.x * c, x.y * c, x.z * c)
    
cdef double intersect_sphere(Vec3 O, 
                      Vec3 D, 
                      Vec3 S, 
                      double R):
    # Return the distance from O to the intersection of the ray (O, D) with the 
    # sphere (S, R), or +inf if there is no intersection.
    # O and S are 3D points, D (direction) is a normalized vector, R is a scalar.
    cdef double a, b, c, disc, distSqrt, q, t0, t1
    cdef Vec3 OS
    
    a = dot(D, D)
    OS = subtract(O, S)
    b = 2 * dot(D, OS)
    c = dot(OS, OS) - R * R
    disc = b * b - 4 * a * c
    if disc > 0:
        distSqrt = sqrt(disc)
        q = (-b - distSqrt) / 2.0 if b < 0 else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return 1000000

cdef Vec3 trace_ray(Vec3 O, Vec3 D,):
    
    cdef double t, radius, diffuse, specular_k, specular_c, DF, SP
    cdef Vec3 M, N, L, toL, toO, col_ray, \
        position, color, color_light, ambient

    # Sphere properties.
    position = vec3(0., 0., 1.)
    radius = 1.
    color = vec3(0., 0., 1.)
    diffuse = 1.
    specular_c = 1.
    specular_k = 50.
    
    # Light position and color.
    L = vec3(5., 5., -10.)
    color_light = vec3(1., 1., 1.)
    ambient = vec3(.05, .05, .05)
    
    # Find first point of intersection with the scene.
    t = intersect_sphere(O, D, position, radius)
    # Return None if the ray does not intersect any object.
    if t == 1000000:
        col_ray.x = 1000000
        return col_ray
    # Find the point of intersection on the object.
    M = vec3(O.x + D.x * t, O.y + D.y * t, O.z + D.z * t)
    N = normalize(subtract(M, position))
    toL = normalize(subtract(L, M))
    toO = normalize(subtract(O, M))
    DF = diffuse * max(dot(N, toL), 0)
    SP = specular_c * max(dot(N, normalize(add(toL, toO))), 0) ** specular_k
    
    return add(ambient, add(multiply_s(color, DF), multiply_s(color_light, SP)))

def run(int w, int h):
    cdef DBL_C[:,:,:] img = np.zeros((h, w, 3))
    cdef Vec3 img_
    cdef int i, j
    cdef double x, y
    cdef Vec3 O, Q, D, col_ray
    cdef double w_ = float(w)
    cdef double h_ = float(h)
    
    col_ray = vec3(0., 0., 0.)
    
    # Camera.
    O = vec3(0., 0., -1.)  # Position.
        
    # Loop through all pixels.
    for i in range(w):
        Q = vec3(0., 0., 0.)
        for j in range(h):
            x = -1. + 2*(i)/w_
            y = -1. + 2*(j)/h_
            Q.x = x
            Q.y = y
            col_ray = trace_ray(O, normalize(subtract(Q, O)))
            if col_ray.x == 1000000:
                continue
            img_ = clip(col_ray, 0., 1.)
            img[h - j - 1, i, 0] = img_.x
            img[h - j - 1, i, 1] = img_.y
            img[h - j - 1, i, 2] = img_.z
    return img
```



``` python
w, h = 200, 200
```



``` python
img = run(w, h)
plt.imshow(img);
plt.xticks([]); plt.yticks([]);
```



``` python
%timeit run(w, h)
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 5.6. Releasing the GIL to take advantage of multi-core processors with Cython and OpenMP


``` python
import numpy as np
import matplotlib.pyplot as plt
```



``` python
%matplotlib inline
```



``` python
import cython
```



``` python
%load_ext cythonmagic
```


This is Cython pushed to its limits: our code that was initially in pure Python is now in almost pure C, with very few Python API calls. Yet, we use the nice Python syntax. We explicitly release the GIL in all functions as they do not use Python, so that we can enable multithread computations on multicore processors with OpenMP.


``` python
%%cython --compile-args=-fopenmp --link-args=-fopenmp --force
from cython.parallel import prange
cimport cython
import numpy as np
cimport numpy as np
DBL = np.double
ctypedef np.double_t DBL_C
from libc.math cimport sqrt

cdef int w, h

cdef struct Vec3:
    double x, y, z
        
cdef Vec3 vec3(double x, double y, double z) nogil:
    cdef Vec3 v
    v.x = x
    v.y = y
    v.z = z
    return v

cdef double dot(Vec3 x, Vec3 y) nogil:
    return x.x * y.x + x.y * y.y + x.z * y.z

cdef Vec3 normalize(Vec3 x) nogil:
    cdef double n
    n = sqrt(x.x * x.x + x.y * x.y + x.z * x.z)
    return vec3(x.x / n, x.y / n, x.z / n)

cdef double max(double x, double y) nogil:
    return x if x > y else y

cdef double min(double x, double y) nogil:
    return x if x < y else y

cdef double clip_(double x, double m, double M) nogil:
    return min(max(x, m), M)

cdef Vec3 clip(Vec3 x, double m, double M) nogil:
    return vec3(clip_(x.x, m, M), clip_(x.y, m, M), clip_(x.z, m, M),)

cdef Vec3 add(Vec3 x, Vec3 y) nogil:
    return vec3(x.x + y.x, x.y + y.y, x.z + y.z)

cdef Vec3 subtract(Vec3 x, Vec3 y) nogil:
    return vec3(x.x - y.x, x.y - y.y, x.z - y.z)

cdef Vec3 minus(Vec3 x) nogil:
    return vec3(-x.x, -x.y, -x.z)

cdef Vec3 multiply(Vec3 x, Vec3 y) nogil:
    return vec3(x.x * y.x, x.y * y.y, x.z * y.z)
    
cdef Vec3 multiply_s(Vec3 x, double c) nogil:
    return vec3(x.x * c, x.y * c, x.z * c)
    
cdef double intersect_sphere(Vec3 O, 
                      Vec3 D, 
                      Vec3 S, 
                      double R) nogil:
    # Return the distance from O to the intersection of the ray (O, D) with the 
    # sphere (S, R), or +inf if there is no intersection.
    # O and S are 3D points, D (direction) is a normalized vector, R is a scalar.
    cdef double a, b, c, disc, distSqrt, q, t0, t1
    cdef Vec3 OS
    
    a = dot(D, D)
    OS = subtract(O, S)
    b = 2 * dot(D, OS)
    c = dot(OS, OS) - R * R
    disc = b * b - 4 * a * c
    if disc > 0:
        distSqrt = sqrt(disc)
        q = (-b - distSqrt) / 2.0 if b < 0 else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return 1000000

cdef Vec3 trace_ray(Vec3 O, Vec3 D,) nogil:
    
    cdef double t, radius, diffuse, specular_k, specular_c, DF, SP
    cdef Vec3 M, N, L, toL, toO, col_ray, \
        position, color, color_light, ambient

    # Sphere properties.
    position = vec3(0., 0., 1.)
    radius = 1.
    color = vec3(0., 0., 1.)
    diffuse = 1.
    specular_c = 1.
    specular_k = 50.
    
    # Light position and color.
    L = vec3(5., 5., -10.)
    color_light = vec3(1., 1., 1.)
    ambient = vec3(.05, .05, .05)
    
    # Find first point of intersection with the scene.
    t = intersect_sphere(O, D, position, radius)
    # Return None if the ray does not intersect any object.
    if t == 1000000:
        col_ray.x = 1000000
        return col_ray
    # Find the point of intersection on the object.
    M = vec3(O.x + D.x * t, O.y + D.y * t, O.z + D.z * t)
    N = normalize(subtract(M, position))
    toL = normalize(subtract(L, M))
    toO = normalize(subtract(O, M))
    DF = diffuse * max(dot(N, toL), 0)
    SP = specular_c * max(dot(N, normalize(add(toL, toO))), 0) ** specular_k
    
    return add(ambient, add(multiply_s(color, DF), multiply_s(color_light, SP)))

@cython.boundscheck(False)
@cython.wraparound(False)
def run(int w, int h):
    cdef DBL_C[:,:,:] img = np.zeros((h, w, 3))
    cdef Vec3 img_
    cdef int i, j
    cdef double x, y
    cdef Vec3 O, Q, D, col_ray
    cdef double w_ = float(w)
    cdef double h_ = float(h)
    
    col_ray = vec3(0., 0., 0.)
    
    # Camera.
    O = vec3(0., 0., -1.)  # Position.
        
    # Loop through all pixels.
    with nogil:
        for i in prange(w):
            Q = vec3(0., 0., 0.)
            for j in range(h):
                x = -1. + 2*(i)/w_
                y = -1. + 2*(j)/h_
                Q.x = x
                Q.y = y
                col_ray = trace_ray(O, normalize(subtract(Q, O)))
                if col_ray.x == 1000000:
                    continue
                img_ = clip(col_ray, 0., 1.)
                img[h - j - 1, i, 0] = img_.x
                img[h - j - 1, i, 1] = img_.y
                img[h - j - 1, i, 2] = img_.z
    return img
```



``` python
w, h = 200, 200
```



``` python
img = run(w, h)
plt.imshow(img);
plt.xticks([]); plt.yticks([]);
```



``` python
%timeit run(w, h)
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 5.6. Releasing the GIL to take advantage of multi-core processors with Cython and OpenMP


``` python
import numpy as np
import matplotlib.pyplot as plt
```



``` python
%matplotlib inline
```



``` python
import cython
```



``` python
%load_ext cythonmagic
```


This is Cython pushed to its limits: our code that was initially in pure Python is now in almost pure C, with very few Python API calls. Yet, we use the nice Python syntax. We explicitly release the GIL in all functions as they do not use Python, so that we can enable multithread computations on multicore processors with OpenMP.


``` python
%%cython --compile-args=/openmp --link-args=/openmp --force
from cython.parallel import prange
cimport cython
import numpy as np
cimport numpy as np
DBL = np.double
ctypedef np.double_t DBL_C
from libc.math cimport sqrt

cdef int w, h

cdef struct Vec3:
    double x, y, z
        
cdef Vec3 vec3(double x, double y, double z) nogil:
    cdef Vec3 v
    v.x = x
    v.y = y
    v.z = z
    return v

cdef double dot(Vec3 x, Vec3 y) nogil:
    return x.x * y.x + x.y * y.y + x.z * y.z

cdef Vec3 normalize(Vec3 x) nogil:
    cdef double n
    n = sqrt(x.x * x.x + x.y * x.y + x.z * x.z)
    return vec3(x.x / n, x.y / n, x.z / n)

cdef double max(double x, double y) nogil:
    return x if x > y else y

cdef double min(double x, double y) nogil:
    return x if x < y else y

cdef double clip_(double x, double m, double M) nogil:
    return min(max(x, m), M)

cdef Vec3 clip(Vec3 x, double m, double M) nogil:
    return vec3(clip_(x.x, m, M), clip_(x.y, m, M), clip_(x.z, m, M),)

cdef Vec3 add(Vec3 x, Vec3 y) nogil:
    return vec3(x.x + y.x, x.y + y.y, x.z + y.z)

cdef Vec3 subtract(Vec3 x, Vec3 y) nogil:
    return vec3(x.x - y.x, x.y - y.y, x.z - y.z)

cdef Vec3 minus(Vec3 x) nogil:
    return vec3(-x.x, -x.y, -x.z)

cdef Vec3 multiply(Vec3 x, Vec3 y) nogil:
    return vec3(x.x * y.x, x.y * y.y, x.z * y.z)
    
cdef Vec3 multiply_s(Vec3 x, double c) nogil:
    return vec3(x.x * c, x.y * c, x.z * c)
    
cdef double intersect_sphere(Vec3 O, 
                      Vec3 D, 
                      Vec3 S, 
                      double R) nogil:
    # Return the distance from O to the intersection of the ray (O, D) with the 
    # sphere (S, R), or +inf if there is no intersection.
    # O and S are 3D points, D (direction) is a normalized vector, R is a scalar.
    cdef double a, b, c, disc, distSqrt, q, t0, t1
    cdef Vec3 OS
    
    a = dot(D, D)
    OS = subtract(O, S)
    b = 2 * dot(D, OS)
    c = dot(OS, OS) - R * R
    disc = b * b - 4 * a * c
    if disc > 0:
        distSqrt = sqrt(disc)
        q = (-b - distSqrt) / 2.0 if b < 0 else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return 1000000

cdef Vec3 trace_ray(Vec3 O, Vec3 D,) nogil:
    
    cdef double t, radius, diffuse, specular_k, specular_c, DF, SP
    cdef Vec3 M, N, L, toL, toO, col_ray, \
        position, color, color_light, ambient

    # Sphere properties.
    position = vec3(0., 0., 1.)
    radius = 1.
    color = vec3(0., 0., 1.)
    diffuse = 1.
    specular_c = 1.
    specular_k = 50.
    
    # Light position and color.
    L = vec3(5., 5., -10.)
    color_light = vec3(1., 1., 1.)
    ambient = vec3(.05, .05, .05)
    
    # Find first point of intersection with the scene.
    t = intersect_sphere(O, D, position, radius)
    # Return None if the ray does not intersect any object.
    if t == 1000000:
        col_ray.x = 1000000
        return col_ray
    # Find the point of intersection on the object.
    M = vec3(O.x + D.x * t, O.y + D.y * t, O.z + D.z * t)
    N = normalize(subtract(M, position))
    toL = normalize(subtract(L, M))
    toO = normalize(subtract(O, M))
    DF = diffuse * max(dot(N, toL), 0)
    SP = specular_c * max(dot(N, normalize(add(toL, toO))), 0) ** specular_k
    
    return add(ambient, add(multiply_s(color, DF), multiply_s(color_light, SP)))

@cython.boundscheck(False)
@cython.wraparound(False)
def run(int w, int h):
    cdef DBL_C[:,:,:] img = np.zeros((h, w, 3))
    cdef Vec3 img_
    cdef int i, j
    cdef double x, y
    cdef Vec3 O, Q, D, col_ray
    cdef double w_ = float(w)
    cdef double h_ = float(h)
    
    col_ray = vec3(0., 0., 0.)
    
    # Camera.
    O = vec3(0., 0., -1.)  # Position.
        
    # Loop through all pixels.
    with nogil:
        for i in prange(w):
            Q = vec3(0., 0., 0.)
            for j in range(h):
                x = -1. + 2*(i)/w_
                y = -1. + 2*(j)/h_
                Q.x = x
                Q.y = y
                col_ray = trace_ray(O, normalize(subtract(Q, O)))
                if col_ray.x == 1000000:
                    continue
                img_ = clip(col_ray, 0., 1.)
                img[h - j - 1, i, 0] = img_.x
                img[h - j - 1, i, 1] = img_.y
                img[h - j - 1, i, 2] = img_.z
    return img
```



``` python
w, h = 200, 200
```



``` python
img = run(w, h)
plt.imshow(img);
plt.xticks([]); plt.yticks([]);
```



``` python
%timeit run(w, h)
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 5.7. Writing massively parallel code for NVIDIA graphics cards (GPUs) with CUDA

Let's import PyCUDA.


``` python
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
```


Now, we initialize the NumPy array that will contain the fractal.


``` python
size = 200
iterations = 100
col = np.empty((size, size), dtype=np.int32)
```


We allocate memory for this array on the GPU.


``` python
col_gpu = cuda.mem_alloc(col.nbytes)
```


We write the CUDA kernel in a string. The mandelbrot function accepts the figure size, the number of iterations, and a pointer to the memory buffer as arguments. It updates the col buffer with the escape value in the fractal for each pixel.


``` python
code = """
__global__ void mandelbrot(int size,
                           int iterations,
                           int *col)

{
    // Get the row and column index of the current thread.
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int index = i * size + j;
    
    // Declare and initialize the variables.
    double cx, cy;
    double z0, z1, z0_tmp, z0_2, z1_2;
    cx = -2.0 + (double)j / size * 3;
    cy = -1.5 + (double)i / size * 3;

    // Main loop.
    z0 = z1 = 0.0;
    for (int n = 0; n < iterations; n++)
    {
        z0_2 = z0 * z0;
        z1_2 = z1 * z1;
        if (z0_2 + z1_2 <= 100)
        {
            // Need to update z0 and z1 in parallel.
            z0_tmp = z0_2 - z1_2 + cx;
            z1 = 2 * z0 * z1 + cy;
            z0 = z0_tmp;
            col[index] = n;
        }
        else break;
    }
}
"""
```


Now, we compile the CUDA program.


``` python
prg = SourceModule(code)
mandelbrot = prg.get_function("mandelbrot")
```


We define the block size and the grid size, specifying how the threads will be parallelized with respect to the data.


``` python
block_size = 10
block = (block_size, block_size, 1)
grid = (size // block_size, size // block_size, 1)
```


We call the compiled function.


``` python
mandelbrot(np.int32(size), np.int32(iterations), col_gpu,
           block=block, grid=grid)
```


Once the function has completed, we copy the contents of the CUDA buffer back to the NumPy array col.


``` python
cuda.memcpy_dtoh(col, col_gpu)
```


Let's display the fractal.


``` python
import matplotlib.pyplot as plt
%matplotlib inline
plt.imshow(np.log(col), cmap=plt.cm.hot,);
plt.xticks([]);
plt.yticks([]);
```


Let's evaluate the time taken by this function.


``` python
%%timeit col_gpu = cuda.mem_alloc(col.nbytes); cuda.memcpy_htod(col_gpu, col)
mandelbrot(np.int32(size), np.int32(iterations), col_gpu,
           block=block, grid=grid)
cuda.memcpy_dtoh(col, col_gpu)
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 5.8. Writing massively parallel code for heterogeneous platforms with OpenCL

Let's import PyOpenCL.


``` python
import pyopencl as cl
import numpy as np
```


This object defines some flags related to memory management on the device.


``` python
mf = cl.mem_flags
```


We create an OpenCL context and a command queue.


``` python
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
```


Now, we initialize the NumPy array that will contain the fractal.


``` python
size = 200
iterations = 100
col = np.empty((size, size), dtype=np.int32)
```


We allocate memory for this array on the GPU.


``` python
col_buf = cl.Buffer(ctx, 
                    mf.WRITE_ONLY,
                    col.nbytes)
```


We write the OpenCL kernel in a string. The mandelbrot function accepts pointers to the buffers as arguments, as well as the figure size. It updates the col buffer with the escape value in the fractal for each pixel.


``` python
code = """
__kernel void mandelbrot(int size,
                         int iterations,
                         global int *col)
{
    // Get the row and column index of the current thread.
    int i = get_global_id(1);
    int j = get_global_id(0);
    int index = i * size + j;
    
    // Declare and initialize the variables.
    double cx, cy;
    double z0, z1, z0_tmp, z0_2, z1_2;
    cx = -2.0 + (double)j / size * 3;
    cy = -1.5 + (double)i / size * 3;

    // Main loop.
    z0 = z1 = 0.0;
    for (int n = 0; n < iterations; n++)
    {
        z0_2 = z0 * z0;
        z1_2 = z1 * z1;
        if (z0_2 + z1_2 <= 100)
        {
            // Need to update z0 and z1 in parallel.
            z0_tmp = z0_2 - z1_2 + cx;
            z1 = 2 * z0 * z1 + cy;
            z0 = z0_tmp;
            col[index] = n;
        }
        else break;
    }
}
"""
```


Now, we compile the OpenCL program.


``` python
prg = cl.Program(ctx, code).build()
```


We call the compiled function, passing the command queue, the grid size, the number of iterations, and the buffer as arguments.


``` python
prg.mandelbrot(queue, col.shape, None, np.int32(size), np.int32(iterations), col_buf).wait()
```


Once the function has completed, we copy the contents of the OpenCL buffer back to the NumPy array col.


``` python
cl.enqueue_copy(queue, col, col_buf);
```


Let's display the fractal.


``` python
import matplotlib.pyplot as plt
%matplotlib inline
plt.imshow(np.log(col), cmap=plt.cm.hot,);
plt.xticks([]);
plt.yticks([]);
```


Let's evaluate the time taken by this function.


``` python
%%timeit
prg.mandelbrot(queue, col.shape, None, np.int32(size), np.int32(iterations), col_buf).wait()
cl.enqueue_copy(queue, col, col_buf);
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 5.9. Distributing Python code across multiple cores with IPython

First, we launch 4 IPython engines with `ipcluster start -n 4` in a console.

Then, we create a client that will act as a proxy to the IPython engines. The client automatically detects the running engines.


``` python
from IPython.parallel import Client
rc = Client()
```


Let's check the number of running engines.


``` python
rc.ids
```


To run commands in parallel over the engines, we can use the %px magic or the %%px cell magic.


``` python
%%px
import os
print("Process {0:d}.".format(os.getpid()))
```


We can specify which engines to run the commands on using the --targets or -t option.


``` python
%%px -t 1,2
# The os module has already been imported in the previous cell.
print("Process {0:d}.".format(os.getpid()))
```


By default, the %px magic executes commands in blocking mode: the cell returns when the commands have completed on all engines. It is possible to run non-blocking commands with the --noblock or -a option. In this case, the cell returns immediately, and the task's status and the results can be polled asynchronously from the IPython interactive session.


``` python
%%px -a
import time
time.sleep(5)
```


The previous command returned an ASyncResult instance that we can use to poll the task's status.


``` python
print(_.elapsed, _.ready())
```


The %pxresult blocks until the task finishes.


``` python
%pxresult
```



``` python
print(_.elapsed, _.ready())
```


IPython provides convenient functions for most common use-cases, like a parallel map function.


``` python
v = rc[:]
res = v.map(lambda x: x*x, range(10))
```



``` python
print(res.get())
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.

# 5.10. Interacting with asynchronous parallel tasks in IPython

You need to start IPython engines (see previous recipe). The simplest option is to launch them from the *Clusters* tab in the notebook dashboard. In this recipe, we use four engines.

1. Let's import a few modules.


``` python
import time
import sys
from IPython import parallel
from IPython.display import clear_output, display
from IPython.html import widgets
```


2. We create a Client.


``` python
rc = parallel.Client()
```


3. Now, we create a load balanced view on the IPython engines.


``` python
view = rc.load_balanced_view()
```


4. We define a simple function for our parallel tasks.


``` python
def f(x):
    import time
    time.sleep(.1)
    return x*x
```


5. We will run this function on 100 integer numbers in parallel.


``` python
numbers = list(range(100))
```


6. We execute `f` on our list `numbers` in parallel across all of our engines, using `map_async()`. This function returns immediately an `AsyncResult` object. This object allows us to retrieve interactively information about the tasks.


``` python
ar = view.map_async(f, numbers)
```


7. This object has a `metadata` attribute, a list of dictionaries for all engines. We can get the date of submission and completion, the status, the standard output and error, and other information.


``` python
ar.metadata[0]
```


8. Iterating over the `AsyncResult` instance works normally; the iteration progresses in real-time while the tasks are being completed.


``` python
for _ in ar:
    print(_, end=', ')
```


9. Now, we create a simple progress bar for our asynchronous tasks. The idea is to create a loop polling for the tasks' status at every second. An `IntProgressWidget` widget is updated in real-time and shows the progress of the tasks.


``` python
def progress_bar(ar):
    # We create a progress bar.
    w = widgets.IntProgressWidget()
    # The maximum value is the number of tasks.
    w.max = len(ar.msg_ids)
    # We display the widget in the output area.
    display(w)
    # Repeat every second:
    while not ar.ready():
        # Update the widget's value with the
        # number of tasks that have finished
        # so far.
        w.value = ar.progress
        time.sleep(1)
    w.value = w.max
```



``` python
ar = view.map_async(f, numbers)
```



``` python
progress_bar(ar)
```


10. Finally, it is easy to debug a parallel task on an engine. We can launch a Qt client on the remote kernel by calling `%qtconsole` within a `%%px` cell magic.


``` python
%%px -t 0
%qtconsole
```


The Qt console allows us to inspect the remote namespace for debugging or analysis purposes.

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.

# 5.11. Using MPI with IPython

For this recipe, you need a MPI installation and the mpi4py package.

1. We first need to create a MPI profile with:


``` python
!ipython profile create --parallel --profile=mpi
```


2. Then, we need to open `~/.ipython/profile_mpi/ipcluster_config.py` and add the line `c.IPClusterEngines.engine_launcher_class = 'MPI'`.

3. Once the MPI profile has been created and configured, we can launch the engines with: `ipcluster start -n 4 --engines MPI --profile=mpi` in a terminal.

4. Now, to actually use the engines, we create a MPI client in the notebook.


``` python
import numpy as np
from IPython.parallel import Client
```



``` python
c = Client(profile='mpi')
```


5. Let's create a view on all engines.


``` python
view = c[:]
```


6. In this example, we compute the sum of all integers between 0 and 15 in parallel over two cores. We first distribute the array with the 16 values across the engines (each engine gets a subarray).


``` python
view.scatter('a', np.arange(16., dtype='float'))
```


7. We compute the total sum in parallel using MPI's `allreduce` function. Every node makes the same computation and returns the same result.


``` python
%%px
from mpi4py import MPI
import numpy as np
print(MPI.COMM_WORLD.allreduce(np.sum(a), op=MPI.SUM))
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.

# 5.12. Trying the Julia language in the notebook

For this recipe, you need to install Julia and IJulia. You'll find the installation instructions in the book.

1. We can't avoid the customary *Hello World* example. The `println()` function displays a string and adds a line break at the end.


``` python
println("Hello world!")
```


2. We create a polymorphic function `f` that computes the expression $z*z+c$. We will notably evaluate this function on arrays, so we use elementwise operators with a dot (`.`) prefix.


``` python
f(z, c) = z.*z .+ c
```


3. Let's evaluate `f` on scalar complex numbers (the imaginary number $i$ is `1im`).


``` python
f(2.0 + 1.0im, 1.0)
```


4. Now, we create a `(2, 2)` matrix. Components are separated by a space, rows are separated by a semicolon (`;`). The type of this `Array` is automatically inferred from its components. The `Array` type is a built-in data type in Julia, similar, but not identical, to NumPy's `ndarray` type.


``` python
z = [-1.0 - 1.0im  1.0 - 1.0im;
     -1.0 + 1.0im  1.0 + 1.0im]
```


5. We can index arrays with brackets `[]`. A notable difference with Python is that indexing starts from 1 instead of 0. MATLAB has the same convention. Besides, the keyword `end` refers to the last item in that dimension.


``` python
z[1,end]
```


6. We can evaluate `f` on the matrix `z` and a scalar `c` (polymorphism).


``` python
f(z, 0)
```


7. Now, we create a function `julia` that computes a Julia set. Optional named arguments are separated from positional arguments by a semicolon (`;`). Julia's syntax for flow control is close from Python's, except that colons are dropped, indentation doesn't count, and block `end` keywords are mandatory.


``` python
function julia(z, c; maxiter=200)
    for n = 1:maxiter
        if abs2(z) > 4.0
            return n-1
        end
        z = f(z, c)
    end
    return maxiter
end
```


8. We can use Python packages from Julia. First, we have to install the `PyCall` package by using Julia's built-in package manager (`Pkg`). Once the package is installed, we can use it in the interactive session with `using PyCall`.


``` python
Pkg.add("PyCall")
using PyCall
```


9. We can import Python packages with the `@pyimport` **macro** (a metaprogramming feature in Julia). This macro is the equivalent of Python's `import` command.


``` python
@pyimport numpy as np
```


10. The `np` namespace is now available in the Julia interactive session. NumPy arrays are automatically converted to Julia `Array`s.


``` python
z = np.linspace(-1., 1., 100)
```


11. We can use list comprehensions to evaluate the function `julia` on many arguments.


``` python
m = [julia(z[i], 0.5) for i=1:100]
```


12. Let's try the Gadfly plotting package. This library offers a high-level plotting interface inspired by Grammar of Graphics. In the notebook, plots are interactive thanks to the **d3.js** library.


``` python
Pkg.add("Gadfly")
using Gadfly
```



``` python
plot(x=1:100, y=m, Geom.point, Geom.line)
```


13. Now, we compute a Julia set by using two nested loops. In general, and unlike Python, there is no significant performance penalty using `for` loops instead of vectorized operations in Julia. High-performance code can be written either with vectorized operations or `for` loops.


``` python
@time m = [julia(complex(r, i), complex(-0.06, 0.67)) 
           for i = 1:-.001:-1,
               r = -1.5:.001:1.5];
```


14. Finally, we use the `PyPlot` package to draw matplotlib figures in Julia.


``` python
Pkg.add("PyPlot")
using PyPlot
```



``` python
imshow(m, cmap="RdGy", 
       extent=[-1.5, 1.5, -1, 1]);
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).

