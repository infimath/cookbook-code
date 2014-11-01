# basic


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 1.1. Introducing the IPython notebook

1. We assume that you have installed a Python distribution with IPython, and that you are now in an IPython notebook. Type in a cell the following command, and press `Shift+Enter` to validate it:


``` python
print("Hello world!")
```


A notebook contains a linear succession of **cells** and **output areas**. A cell contains Python code, in one or multiple lines. The output of the code is shown in the corresponding output area.

2. Now, we do a simple arithmetic operation.


``` python
2+2
```


The result of the operation is shown in the output area. Let's be more precise. The output area not only displays text that is printed by any command in the cell, it also displays a text representation of the last returned object. Here, the last returned object is the result of `2+2`, i.e. `4`.

3. In the next cell, we can recover the value of the last returned object with the `_` (underscore) special variable. In practice, it may be more convenient to assign objects to named variables, like in `myresult = 2+2`.


``` python
_ * 3
```


4. IPython not only accepts Python code, but also shell commands. Those are defined by the operating system (Windows, Linux, Mac OS X, etc.). We first type `!` in a cell before typing the shell command. Here, we get the list of notebooks in the current directory.


``` python
!ls *.ipynb
```


5. IPython comes with a library of **magic commands**. Those commands are convenient shortcuts to common actions. They all start with `%` (percent character). You can get the list of all magic commands with `%lsmagic`.


``` python
%lsmagic
```


Cell magic have a `%%` prefix: they apply to an entire cell in the notebook.

6. For example, the `%%writefile` cell magic lets you create a text file easily. This magic command accepts a filename as argument. All remaining lines in the cell are directly written to this text file. Here, we create a file `test.txt` and we write `Hello world!` in it.


``` python
%%writefile test.txt
Hello world!
```



``` python
# Let's check what this file contains.
with open('test.txt', 'r') as f:
    print(f.read())
```


7. As you can see in the output of `%lsmagic`, there are many magic commands in IPython. You can find more information about any command by adding a `?` after it. For example, here is how we get help about the `%run` magic command:


``` python
# You can omit this, it is just to force help output
# to print in the standard output, rather than 
# in the pager. This might change in future versions
# of IPython.
from __future__ import print_function
from IPython.core import page
page.page = print
```



``` python
%run?
```


8. We covered the basics of IPython and the notebook. Let's now turn to the rich display and interactive features of the notebook. Until now, we only created **code cells**, i.e. cells that contain... code. There are other types of cells, notably **Markdown cells**. Those contain rich text formatted with **Markdown**, a popular plain text formatting syntax. This format supports normal text, headers, bold, italics, hypertext links, images, mathematical equations in LaTeX, code, HTML elements, and other features, as shown below.

### New paragraph

This is *rich* **text** with [links](http://ipython.org), equations:

$$\hat{f}(\xi) = \int_{-\infty}^{+\infty} f(x)\, \mathrm{e}^{-i \xi x}$$

code with syntax highlighting:

```python
print("Hello world!")
```

and images:

![This is an image](http://ipython.org/_static/IPy_header.png)

By combining code cells and Markdown cells, you can create a standalone interactive document that combines computations (code), text and graphics.

9. That was it for Markdown cells. IPython also comes with a sophisticated display system that lets you insert rich web elements in the notebook. Here, we show how to add HTML, SVG (Scalable Vector Graphics) and even Youtube videos in a notebook.

First, we need to import some classes.


``` python
from IPython.display import HTML, SVG, YouTubeVideo
```


We create an HTML table dynamically with Python, and we display it in the (HTML-based) notebook.


``` python
HTML('''
<table style="border: 2px solid black;">
''' + 
''.join(['<tr>' + 
         ''.join(['<td>{row},{col}</td>'.format(
            row=row, col=col
            ) for col in range(5)]) +
         '</tr>' for row in range(5)]) +
'''
</table>
''')
```


Similarly here, we create a SVG graphics dynamically.


``` python
SVG('''<svg width="600" height="80">''' + 
''.join(['''<circle cx="{x}" cy="{y}" r="{r}"
        fill="red" stroke-width="2" stroke="black">
        </circle>'''.format(
            x=(30+3*i)*(10-i), y=30, r=3.*float(i)
        ) for i in range(10)]) + 
'''</svg>''')
```


Finally, we display a Youtube video by giving its identifier to `YoutubeVideo`.


``` python
YouTubeVideo('j9YpkSX7NNM')
```


10. Now, we illustrate the latest interactive features in IPython 2.0+. This version brings graphical widgets in the notebook that can interact with Python objects. We will create a drop-down menu allowing us to display one among several videos.


``` python
from collections import OrderedDict
from IPython.display import display, clear_output
from IPython.html.widgets import DropdownWidget
```



``` python
# We create a DropdownWidget, with a dictionary containing
# the keys (video name) and the values (Youtube identifier) 
# of every menu item.
dw = DropdownWidget(values=OrderedDict([
                        ('SciPy 2012', 'iwVvqwLDsJo'), 
                        ('PyCon 2012', '2G5YTlheCbw'),
                        ('SciPy 2013', 'j9YpkSX7NNM')]))
# We create a callback function that displays the requested
# Youtube video.
def on_value_change(name, val):
    clear_output()
    display(YouTubeVideo(val))
# Every time the user selects an item, the function
# `on_value_change` is called, and the `val` argument
# contains the value of the selected item.
dw.on_trait_change(on_value_change, 'value')
# We choose a default value.
dw.value = dw.values['SciPy 2013']
# Finally, we display the widget.
display(dw)
```


The interactive features of IPython 2.0 bring a whole new dimension in the notebook, and we can expect many developments in the months and years to come.

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 1.2. Getting started with exploratory data analysis in IPython

We will download and process a dataset about attendance on Montreal's bicycle tracks. This example is largely inspired by a presentation from [Julia Evans](http://nbviewer.ipython.org/github/jvns/talks/blob/master/mtlpy35/pistes-cyclables.ipynb).

1. The very first step is to import the scientific packages we will be using in this recipe, namely NumPy, Pandas, and matplotlib. We also instruct matplotlib to render the figures as PNG images in the notebook.


``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```


2. Now, we create a new Python variable called `url` that contains the address to a CSV (**Comma-separated values**) data file. This standard text-based file format is used to store tabular data.


``` python
url = "http://donnees.ville.montreal.qc.ca/storage/f/2014-01-20T20%3A48%3A50.296Z/2013.csv"
```


3. Pandas defines a `read_csv` function that can read any CSV file. Here, we give it the URL to the file. Pandas will automatically download and parse the file, and return a `DataFrame` object. We need to specify a few options to make sure the dates are parsed correctly.


``` python
df = pd.read_csv(url, index_col='Date', parse_dates=True, dayfirst=True)
```


4. The `df` variable contains a `DataFrame` object, a specific Pandas data structure that contains 2D tabular data. The `head(n)` method displays the first `n` rows of this table.


``` python
df.head(2)
```


Every row contains the number of bicycles on every track of the city, for every day of the year.

5. We can get some summary statistics of the table with the `describe` method.


``` python
df.describe()
```


6. Let's display some figures! We will plot the daily attendance of two tracks. First, we select the two columns `'Berri1'` and `'PierDup'`. Then, we call the `plot` method.


``` python
# The styling '-' and '--' is just to make the figure
# readable in the black & white printed version of this book.
df[['Berri1', 'PierDup']].plot(figsize=(8,4),
                               style=['-', '--']);
```


7. Now, we move to a slightly more advanced analysis. We will look at the attendance of all tracks as a function of the weekday. We can get the week day easily with Pandas: the `index` attribute of the `DataFrame` contains the dates of all rows in the table. This index has a few date-related attributes, including `weekday`.


``` python
df.index.weekday
```


However, we would like to have names (Monday, Tuesday, etc.) instead of numbers between 0 and 6. This can be done easily. First, we create an array `days` with all weekday names. Then, we index it by `df.index.weekday`. This operation replaces every integer in the index by the corresponding name in `days`. The first element, `Monday`, has index 0, so every 0 in `df.index.weekday` is replaced by `Monday`, and so on. We assign this new index to a new column `Weekday` in the `DataFrame`.


``` python
days = np.array(['Monday', 'Tuesday', 'Wednesday', 
                 'Thursday', 'Friday', 'Saturday', 
                 'Sunday'])
df['Weekday'] = days[df.index.weekday]
```


8. To get the attendance as a function of the weekday, we need to group the table by the weekday. The `groupby` method lets us do just that. Once grouped, we can sum all rows in every group.


``` python
df_week = df.groupby('Weekday').sum()
```



``` python
df_week
```


9. We can now display this information in a figure. We first need to reorder the table by the weekday using `ix` (indexing operation). Then, we plot the table, specifying the line width and the figure size.


``` python
df_week.ix[days].plot(lw=3, figsize=(6,4));
plt.ylim(0);  # Set the bottom axis to 0.
```


10. Finally, let's illustrate the new interactive capabilities of the notebook in IPython 2.0. We will plot a *smoothed* version of the track attendance as a function of time (**rolling mean**). The idea is to compute the mean value in the neighborhood of any day. The larger the neighborhood, the smoother the curve. We will create an interactive slider in the notebook to vary this parameter in real-time in the plot.


``` python
from IPython.html.widgets import interact
@interact
def plot(n=(1, 30)):
    plt.figure(figsize=(8,4));
    pd.rolling_mean(df['Berri1'], n).dropna().plot();
    plt.ylim(0, 8000);
    plt.show();
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 1.3. Introducing the multidimensional array in NumPy for fast array computations

1. Let's import the built-in `random` Python module and NumPy.


``` python
import random
import numpy as np
```


We use the `%precision` magic (defined in IPython) to show only 3 decimals in the Python output. This is just to alleviate the text.


``` python
%precision 3
```


2. We generate two Python lists `x` and `y`, each one containing one million random numbers between 0 and 1.


``` python
n = 1000000
x = [random.random() for _ in range(n)]
y = [random.random() for _ in range(n)]
```



``` python
x[:3], y[:3]
```


3. Let's compute the element-wise sum of all these numbers: the first element of `x` plus the first element of `y`, and so on. We use a `for` loop in a list comprehension.


``` python
z = [x[i] + y[i] for i in range(n)]
z[:3]
```


4. How long does this computation take? IPython defines a handy `%timeit` magic command to quickly evaluate the time taken by a single command.


``` python
%timeit [x[i] + y[i] for i in range(n)]
```


5. Now, we will perform the same operation with NumPy. NumPy works on multidimensional arrays, so we need to convert our lists to arrays. The `np.array()` function does just that.


``` python
xa = np.array(x)
ya = np.array(y)
```



``` python
xa[:3]
```


The arrays `xa` and `ya` contain the *exact* same numbers than our original lists `x` and `y`. Whereas those lists where instances of a built-in class `list`, our arrays are instances of a NumPy class `ndarray`. Those types are implemented very differently in Python and NumPy. We will see that, in this example, using arrays instead of lists leads to drastic performance improvements.

6. Now, to compute the element-wise sum of these arrays, we don't need to do a `for` loop anymore. In NumPy, adding two arrays means adding the elements of the arrays component by component.


``` python
za = xa + ya
za[:3]
```


We see that the list `z` and the array `za` contain the same elements (the sum of the numbers in `x` and `y`).

7. Let's compare the performance of this NumPy operation with the native Python loop.


``` python
%timeit xa + ya
```


We observe that this operation is more than one order of magnitude faster in NumPy than in pure Python!

8. Now, we will compute something else: the sum of all elements in `x` or `xa`. Although this is not an element-wise operation, NumPy is still highly efficient here. The pure Python version uses the built-in `sum` function on an iterable. The NumPy version uses the `np.sum()` function on a NumPy array.


``` python
%timeit sum(x)  # pure Python
%timeit np.sum(xa)  # NumPy
```


We also observe an impressive speedup here.

9. Finally, let's perform a last operation: computing the arithmetic distance between any pair of numbers in our two lists (we only consider the first 1000 elements to keep computing times reasonable). First, we implement this in pure Python with two nested `for` loops.


``` python
d = [abs(x[i] - y[j]) 
     for i in range(1000) for j in range(1000)]
```



``` python
d[:3]
```


10. Now, we use a NumPy implementation, bringing out two slightly more advanced notions. First, we consider a **two-dimensional array** (or matrix). This is how we deal with *two* indices *i* and *j*. Second, we use **broadcasting** to perform an operation between a 2D array and a 1D array. We will give more details in *How it works...*


``` python
da = np.abs(xa[:1000,None] - ya[:1000])
```



``` python
da
```



``` python
%timeit [abs(x[i] - y[j]) for i in range(1000) for j in range(1000)]
%timeit np.abs(xa[:1000, None] - ya[:1000])
```


Here again, observe observe the significant speedups.

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 1.4. Creating an IPython extension with custom magic commands

1. Let's import a few functions from the IPython magic system.


``` python
from IPython.core.magic import (register_line_magic, 
                                register_cell_magic)
```


2. Defining a new line magic is particularly simple. First, let's create a function that accepts the contents of the line (except the initial `%`-prefixed magic command). The name of this function is the name of the magic. Then, let's decorate this function with `@register_line_magic`. We're done!


``` python
@register_line_magic
def hello(line):
    if line == 'french':
        print("Salut tout le monde!")
    else:
        print("Hello world!")
```



``` python
%hello
```



``` python
%hello french
```


3. Let's create a slightly more useful cell magic `%%csv` that parses a CSV string and returns a Pandas DataFrame object. This time, the function takes as argument the first line (what follows `%%csv`), and the contents of the cell (everything in the cell except the first line).


``` python
import pandas as pd
#from StringIO import StringIO  # Python 2
from io import StringIO  # Python 3

@register_cell_magic
def csv(line, cell):
    # We create a string buffer containing the
    # contents of the cell.
    sio = StringIO(cell)
    # We use Pandas' read_csv function to parse
    # the CSV string.
    return pd.read_csv(sio)
```



``` python
%%csv
col1,col2,col3
0,1,2
3,4,5
7,8,9
```


We can access the returned object with `_`.


``` python
df = _
df.describe()
```


4. The method we described is useful in an interactive session. If you want to use the same magic in multiple notebooks, or if you want to distribute it, you need to create an **IPython extension** that implements your custom magic command. Let's show how to do that. The first step is to create a Python script (`csvmagic.py` here) that implements the magic.


``` python
%%writefile csvmagic.py
import pandas as pd
#from StringIO import StringIO  # Python 2
from io import StringIO  # Python 3

def csv(line, cell):
    sio = StringIO(cell)
    return pd.read_csv(sio)

def load_ipython_extension(ipython):
    """This function is called when the extension is loaded.
    It accepts an IPython InteractiveShell instance.
    We can register the magic with the `register_magic_function`
    method of the shell instance."""
    ipython.register_magic_function(csv, 'cell')
```


5. Once the extension is created, we need to import it in the IPython session. The `%load_ext` magic command takes the name of a Python module and imports it, calling immediately `load_ipython_extension`. Here, loading this extension automatically registers our magic function `%%csv`. The Python module needs to be importable. Here, it is in the current directory. In other situations, it has to be in the Python path. It can also be stored in `~\.ipython\extensions` which is automatically put in the Python path.


``` python
%load_ext csvmagic
```



``` python
%%csv
col1,col2,col3
0,1,2
3,4,5
7,8,9
```


Finally, to ensure that this magic is automatically defined in our IPython profile, we can instruct IPython to load this extension at startup. To do this, let's open the file `~/.ipython/profile_default/ipython_config.py` and let's put `'csvmagic'` in the `c.InteractiveShellApp.extensions` list. The `csvmagic` module needs to be importable. It is common to create a *Python package* implementing an IPython extension, which itself defines custom magic commands.

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.

# 1.5. Mastering IPython's configuration system


``` python
%%writefile random_magics.py
# NOTE: We create the `random_magics.py` file here so that 
# you don't have to do it...
from IPython.utils.traitlets import Int, Float, Unicode, Bool
from IPython.core.magic import (Magics, magics_class, line_magic)
import numpy as np

@magics_class
class RandomMagics(Magics):
    text = Unicode(u'{n}', config=True)
    max = Int(1000, config=True)
    seed = Int(0, config=True)
    
    def __init__(self, shell):
        super(RandomMagics, self).__init__(shell)
        self._rng = np.random.RandomState(self.seed or None)
        
    @line_magic
    def random(self, line):
        return self.text.format(n=self._rng.randint(self.max))
    
def load_ipython_extension(ipython):
    ipython.register_magics(RandomMagics)
```


1. We create an IPython extension in a file `random_magics.py`. Let's start by importing a few objects:

2. We create a `RandomMagics` class deriving from `Magics`. This class contains a few configurable parameters.

3. We need to call the parent's constructor. Then, we initialize a random number generator with a seed.

4. Then, we create a line magic `%random` that displays a random number.

5. Finally, we register that magics when the extension is loaded.

6. Let's test our extension!


``` python
%load_ext random_magics
```



``` python
%random
```



``` python
%random
```


7. Our magics command has a few configurable parameters. These variables are meant to be configured by the user in the IPython configuration file, or in the console when starting IPython. To configure these variables in the terminal, we can type in a system shell the following command:

In that session, we get the following behavior:

8. To configure the variables in the IPython configuration file, we have to open the file `~/.ipython/profile_cookbook/ipython_config.py` and add the following line:

After launching IPython, we get the following behavior:


``` python
%random
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.

# 1.6. Creating a simple kernel for IPython

This recipe has been tested on the development version of IPython 3. It should work on the final version of IPython 3 with no or minimal changes. We give all references about wrapper kernels and messaging protocols at the end of this recipe.

Besides, the code given here works with Python 3. It can be ported to Python 2 with minimal efforts.


``` python
%%writefile plotkernel.py
# NOTE: We create the `plotkernel.py` file here so that 
# you don't have to do it...
from IPython.kernel.zmq.kernelbase import Kernel
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import urllib, base64

def _to_png(fig):
    """Return a base64-encoded PNG from a 
    matplotlib figure."""
    imgdata = BytesIO()
    fig.savefig(imgdata, format='png')
    imgdata.seek(0)
    return urllib.parse.quote(
        base64.b64encode(imgdata.getvalue()))

_numpy_namespace = {n: getattr(np, n) 
                    for n in dir(np)}
def _parse_function(code):
    """Return a NumPy function from a string 'y=f(x)'."""
    return lambda x: eval(code.split('=')[1].strip(),
                          _numpy_namespace, {'x': x})

class PlotKernel(Kernel):
    implementation = 'Plot'
    implementation_version = '1.0'
    language = 'python'  # will be used for
                         # syntax highlighting
    language_version = ''
    banner = "Simple plotting"
    
    def do_execute(self, code, silent,
                   store_history=True,
                   user_expressions=None,
                   allow_stdin=False):

        # We create the plot with matplotlib.
        fig = plt.figure(figsize=(6,4), dpi=100)
        x = np.linspace(-5., 5., 200)
        functions = code.split('\n')
        for fun in functions:
            f = _parse_function(fun)
            y = f(x)
            plt.plot(x, y)
        plt.xlim(-5, 5)

        # We create a PNG out of this plot.
        png = _to_png(fig)

        if not silent:
            # We send the standard output to the client.
            self.send_response(self.iopub_socket,
                'stream', {
                    'name': 'stdout', 
                    'data': 'Plotting {n} function(s)'. \
                                format(n=len(functions))})

            # We prepare the response with our rich data
            # (the plot).
            content = {
                'source': 'kernel',

                # This dictionary may contain different
                # MIME representations of the output.
                'data': {
                    'image/png': png
                },

                # We can specify the image size
                # in the metadata field.
                'metadata' : {
                      'image/png' : {
                        'width': 600,
                        'height': 400
                      }
                    }
            }        

            # We send the display_data message with the
            # contents.
            self.send_response(self.iopub_socket,
                'display_data', content)

        # We return the exection results.
        return {'status': 'ok',
                'execution_count': self.execution_count,
                'payload': [],
                'user_expressions': {},
               }

if __name__ == '__main__':
    from IPython.kernel.zmq.kernelapp import IPKernelApp
    IPKernelApp.launch_instance(kernel_class=PlotKernel)
```


1. First, we create a file `plotkernel.py`. This file will contain the implementation of our custom kernel. Let's import a few modules.

```
from IPython.kernel.zmq.kernelbase import Kernel
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import urllib, base64```

2. We write a function that returns a PNG base64-encoded representation of a matplotlib figure.

```
def _to_png(fig):
    """Return a base64-encoded PNG from a 
    matplotlib figure."""
    imgdata = BytesIO()
    fig.savefig(imgdata, format='png')
    imgdata.seek(0)
    return urllib.parse.quote(
        base64.b64encode(imgdata.getvalue()))```

3. Now, we write a function that parses a code string which has the form `y = f(x)`, and returns a NumPy function. Here, `f` is an arbitrary Python expression that can use NumPy functions.

```
_numpy_namespace = {n: getattr(np, n) 
                    for n in dir(np)}
def _parse_function(code):
    """Return a NumPy function from a string 'y=f(x)'."""
    return lambda x: eval(code.split('=')[1].strip(),
                          _numpy_namespace, {'x': x})```

4. For our new wrapper kernel, we create a class deriving from `Kernel`. There are a few metadata fields we need to provide.

```
class PlotKernel(Kernel):
    implementation = 'Plot'
    implementation_version = '1.0'
    language = 'python'  # will be used for
                         # syntax highlighting
    language_version = ''
    banner = "Simple plotting"
    ```

5. In this class, we implement a `do_execute()` method that takes code as input, and sends responses to the client.

```
def do_execute(self, code, silent,
                   store_history=True,
                   user_expressions=None,
                   allow_stdin=False):

        # We create the plot with matplotlib.
        fig = plt.figure(figsize=(6,4), dpi=100)
        x = np.linspace(-5., 5., 200)
        functions = code.split('\n')
        for fun in functions:
            f = _parse_function(fun)
            y = f(x)
            plt.plot(x, y)
        plt.xlim(-5, 5)

        # We create a PNG out of this plot.
        png = _to_png(fig)

        if not silent:
            # We send the standard output to the client.
            self.send_response(self.iopub_socket,
                'stream', {
                    'name': 'stdout', 
                    'data': 'Plotting {n} function(s)'. \
                                format(n=len(functions))})

            # We prepare the response with our rich data
            # (the plot).
            content = {
                'source': 'kernel',

                # This dictionary may contain different
                # MIME representations of the output.
                'data': {
                    'image/png': png
                },

                # We can specify the image size
                # in the metadata field.
                'metadata' : {
                      'image/png' : {
                        'width': 600,
                        'height': 400
                      }
                    }
            }        

            # We send the display_data message with the
            # contents.
            self.send_response(self.iopub_socket,
                'display_data', content)

        # We return the exection results.
        return {'status': 'ok',
                'execution_count': self.execution_count,
                'payload': [],
                'user_expressions': {},
               }```

6. Finally, we add the following lines at the end of the file.

```
if __name__ == '__main__':
    from IPython.kernel.zmq.kernelapp import IPKernelApp
    IPKernelApp.launch_instance(kernel_class=PlotKernel)```

7. Our kernel is ready! The next step is to indicate to IPython that this new kernel is available. To do this, we need to create a **kernel spec** `kernel.json` file and put it in `~/.ipython/kernels/plot/`. This file contains the following lines:

```
{
 "argv": ["python", "-m",
          "plotkernel", "-f",
          "{connection_file}"],
 "display_name": "Plot",
 "language": "python"
}```

The `plotkernel.py` file needs to be importable by Python. For example, you could simply put it in the current directory.

8. In IPython 3, you can launch a notebook with this kernel from the IPython notebook dashboard. However, this feature is not available at the time of writing. An alternative (that is probably going to be deprecated by the time IPython 3 is released) is to run the following command in a terminal:

```
ipython notebook --KernelManager.kernel_cmd="['python', '-m', 'plotkernel', '-f', '{connection_file}']"
```

9. Finally, in a new notebook backed by our custom plot kernel, we can simply write mathematical equations `y=f(x)`. The corresponding graph appears in the output area.

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).

