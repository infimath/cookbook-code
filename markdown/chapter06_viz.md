# viz


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 6.1. Making nicer matplotlib figures with prettyplotlib

1. Let's first import NumPy and matplotlib.


``` python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```


2. We first draw several curves with matplotlib.


``` python
plt.figure(figsize=(6,4));
np.random.seed(12)
for i in range(8):
    x = np.arange(1000)
    y = np.random.randn(1000).cumsum()
    plt.plot(x, y, label=str(i))
plt.legend();
```


3. Now, we create the exact same plot with prettplotlib. We can basically replace the `matplotlib.pyplot` namespace with `prettyplotlib`.


``` python
import prettyplotlib as ppl
plt.figure(figsize=(6,4));
np.random.seed(12)
for i in range(8):
    x = np.arange(1000)
    y = np.random.randn(1000).cumsum()
    ppl.plot(x, y, label=str(i))
ppl.legend();
```


The figure appears clearer, and the colors are more visually pleasant.

4. Let's show another example with an image. We first use matplotlib's `pcolormesh` function to display a 2D array as an image.


``` python
plt.figure(figsize=(4,3));
np.random.seed(12)
plt.pcolormesh(np.random.rand(16, 16));
plt.colorbar();
```


The default *rainbow* color map is known to mislead data visualization.

5. Now, we use prettyplotlib to display the exact same image.


``` python
plt.figure(figsize=(4,3));
np.random.seed(12);
ppl.pcolormesh(np.random.rand(16, 16));
```


This visualization is much clearer, in that high or low values are better brought out than with the rainbow color map.

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 6.2. Creating beautiful statistical plots with seaborn

1. Let's import NumPy, matplotlib, and seaborn.


``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


2. We generate a random dataset (following this example on seaborn's website: http://nbviewer.ipython.org/github/mwaskom/seaborn/blob/master/examples/linear_models.ipynb)


``` python
x1 = np.random.randn(80)
x2 = np.random.randn(80)
x3 = x1 * x2
y1 = .5 + 2 * x1 - x2 + 2.5 * x3 + 3 * np.random.randn(80)
y2 = .5 + 2 * x1 - x2 + 2.5 * np.random.randn(80)
y3 = y2 + np.random.randn(80)
```


2. Seaborn implements many easy-to-use statistical plotting functions. For example, here is how to create a violin plot (showing the distribution of several sets of points).


``` python
plt.figure(figsize=(4,3));
sns.violinplot([x1,x2, x3]);
```


4. Seaborn also implement all-in-one statistical visualization functions. For example, one can use a single function (`regplot`) to perform *and* display a linear regression between two variables.


``` python
plt.figure(figsize=(4,3));
sns.regplot(x2, y2);
```


5. Seaborn has built-in support for Pandas data structures. Here, we display the pairwise correlations between all variables defined in a `DataFrame`.


``` python
df = pd.DataFrame(dict(x1=x1, x2=x2, x3=x3, 
                       y1=y1, y2=y2, y3=y3))
sns.corrplot(df);
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 6.3. Creating interactive Web visualizations with Bokeh

1. Let's import NumPy and Bokeh. We need to call `output_notebook` in order to tell Bokeh to render plots in the IPython notebook.


``` python
import numpy as np
import bokeh.plotting as bkh
bkh.output_notebook()
```


2. Let's create some random data.


``` python
x = np.linspace(0., 1., 100)
y = np.cumsum(np.random.randn(100))
```


3. Let's draw a curve.


``` python
bkh.line(x, y, line_width=5)
bkh.show()
```


An interactive plot is rendered in the notebook. We can pan and zoom by clicking on the buttons above the plot.

4. Let's move to another example. We first load a sample dataset (*Iris flowers*). We also generate some colors based on the species of the flowers.


``` python
from bokeh.sampledata.iris import flowers
colormap = {'setosa': 'red',
            'versicolor': 'green',
            'virginica': 'blue'}
flowers['color'] = flowers['species'].map(lambda x: colormap[x])
```


5. Now, we render an interactive scatter plot.


``` python
bkh.scatter(flowers["petal_length"], 
            flowers["petal_width"],
            color=flowers["color"], 
            fill_alpha=0.25, size=10,)
bkh.show()
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 6.4. Visualizing a NetworkX graph in the IPython notebook with d3.js

1. Let's import the packages.


``` python
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
%matplotlib inline
```


2. We load a famous social graph published in 1977, called **Zachary's Karate club graph**. This graph represents the friendships between members of a Karate Club. The club's president and the instructor were involved in a dispute, resulting in a split-up of this group. Here, we simply display the graph with matplotlib (using `networkx.draw()`).


``` python
g = nx.karate_club_graph()
plt.figure(figsize=(6,4));
nx.draw(g)
```


3. Now, we're going to display this graph in the notebook with d3.js. The first step is to bring this graph to Javascript. We choose here to export the graph in JSON. Note that d3.js generally expects each edge to be an object with a `source` and a `target`. Also, we specify which side each member has taken (`club` attribute).


``` python
from networkx.readwrite import json_graph
data = json_graph.node_link_data(g)
with open('graph.json', 'w') as f:
    json.dump(data, f, indent=4)
```


4. The next step is to create an HTML object that will contain the visualization. Here, we create a `<div>` element in the notebook. We also specify a few CSS styles for nodes and links (also called edges).


``` python
%%html
<div id="d3-example"></div>
<style>
.node {stroke: #fff; stroke-width: 1.5px;}
.link {stroke: #999; stroke-opacity: .6;}
</style>
```


5. The last step is trickier. We write the Javascript code to load the graph from the JSON file, and display it with d3.js. Knowing the basics of d3.js is required here (see the documentation of d3.js). We also give detailled explanations in the code comments below. (http://d3js.org)


``` python
%%javascript
// We load the d3.js library from the Web.
require.config({paths: {d3: "http://d3js.org/d3.v3.min"}});
require(["d3"], function(d3) {
    // The code in this block is executed when the 
    // d3.js library has been loaded.
    
    // First, we specify the size of the canvas containing
    // the visualization (size of the <div> element).
    var width = 300,
        height = 300;

    // We create a color scale.
    var color = d3.scale.category10();

    // We create a force-directed dynamic graph layout.
    var force = d3.layout.force()
        .charge(-120)
        .linkDistance(30)
        .size([width, height]);

    // In the <div> element, we create a <svg> graphic
    // that will contain our interactive visualization.
    var svg = d3.select("#d3-example").select("svg")
    if (svg.empty()) {
        svg = d3.select("#d3-example").append("svg")
                    .attr("width", width)
                    .attr("height", height);
    }
        
    // We load the JSON file.
    d3.json("graph.json", function(error, graph) {
        // In this block, the file has been loaded
        // and the 'graph' object contains our graph.
        
        // We load the nodes and links in the force-directed
        // graph.
        force.nodes(graph.nodes)
            .links(graph.links)
            .start();

        // We create a <line> SVG element for each link
        // in the graph.
        var link = svg.selectAll(".link")
            .data(graph.links)
            .enter().append("line")
            .attr("class", "link");

        // We create a <circle> SVG element for each node
        // in the graph, and we specify a few attributes.
        var node = svg.selectAll(".node")
            .data(graph.nodes)
            .enter().append("circle")
            .attr("class", "node")
            .attr("r", 5)  // radius
            .style("fill", function(d) {
                // The node color depends on the club.
                return color(d.club); 
            })
            .call(force.drag);

        // The name of each node is the node number.
        node.append("title")
            .text(function(d) { return d.name; });

        // We bind the positions of the SVG elements
        // to the positions of the dynamic force-directed graph,
        // at each time step.
        force.on("tick", function() {
            link.attr("x1", function(d) { return d.source.x; })
                .attr("y1", function(d) { return d.source.y; })
                .attr("x2", function(d) { return d.target.x; })
                .attr("y2", function(d) { return d.target.y; });

            node.attr("cx", function(d) { return d.x; })
                .attr("cy", function(d) { return d.y; });
        });
    });
});
```


When we execute this cell, the HTML object created in the previous cell is updated. The graph is animated and interactive: we can click on nodes, see their labels, and move them within the canvas.

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 6.5. Converting matplotlib figures to d3.js visualizations with mpld3

1. First, we load NumPy and matplotlib as usual.


``` python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```


2. Then, we enable the mpld3 figures in the notebook with a single function call.


``` python
from mpld3 import enable_notebook
enable_notebook()
```


3. Now, let's create a scatter plot with matplotlib.


``` python
X = np.random.normal(0, 1, (100, 3))
color = np.random.random(100)
size = 500 * np.random.random(100)
plt.figure(figsize=(6,4))
plt.scatter(X[:,0], X[:,1], c=color,
            s=size, alpha=0.5, linewidths=2)
plt.grid(color='lightgray', alpha=0.7)
```


The matplotlib figure is rendered with d3.js instead of the standard matplotlib backend. In particular, the figure is interactive (pan and zoom).

4. Now, we create a more complex example with multiple subplots representing different 2D projections of a 3D dataset. We use the `sharex` and `sharey` keywords in matplotlib's `subplots` function to automatically bind the x and y axes of the different figures. Panning and zooming in any of the subplots automatically updates all the other subplots.


``` python
fig, ax = plt.subplots(3, 3, figsize=(6, 6),
                       sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.3)
X[::2,2] += 3
for i in range(3):
    for j in range(3):
        ax[i,j].scatter(X[:,i], X[:,j], c=color,
            s=.1*size, alpha=0.5, linewidths=2)
        ax[i,j].grid(color='lightgray', alpha=0.7)
```


This use case is perfectly handled by mpld3: the d3.js subplots are also dynamically linked together.

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.

# 6.6. Getting started with Vispy for high-performance interactive data visualizations

Vispy depends on NumPy and PyOpenGL. A backend library is necessary (PyQt4 or PySide for example).

This recipe has been tested with the [development version of Vispy](https://github.com/vispy/vispy). You should clone the GitHub repository and install Vispy with `python setup.py install`.

The API used in this recipe might change in future versions.

1. Let's import NumPy, **vispy.app** (to display a canvas) and **vispy.gloo** (object-oriented interface to OpenGL).


``` python
import numpy as np
from vispy import app
from vispy import gloo
```


2. In order to display a window, we need to create a **Canvas**.


``` python
c = app.Canvas(keys='interactive')
```


3. When using `vispy.gloo`, we need to write **shaders**. These programs written in a C-like language run on the GPU and give us full flexibility for our visualizations. Here, we create a trivial **vertex shader** that directly displays 2D data points (stored in the `a_position` variable) in the canvas. We give more details in *There's more*.


``` python
vertex = """
attribute vec2 a_position;
void main (void)
{
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""
```


4. The other shader we need to create is the **fragment shader**. It lets us control the pixels' color. Here, we display all data points in black.


``` python
fragment = """
void main()
{
    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
}
"""
```


5. Next, we create an **OpenGL Program**. This object contains shaders and links the shader variables to NumPy data.


``` python
program = gloo.Program(vertex, fragment)
```


6. We link the variable `a_position` to a `(1000, 2)` NumPy array containing the coordinates of 1000 data points. In the default coordinate system, the coordinates of the four canvas corners are `(+/-1, +/-1)`.


``` python
program['a_position'] = np.c_[
        np.linspace(-1.0, +1.0, 1000),
        np.random.uniform(-0.5, +0.5, 1000)]
```


7. We create a callback function for when the window is being resized. Updating the **OpenGL viewport** lets us ensure that Vispy uses the entire canvas.


``` python
@c.connect
def on_resize(event):
    gloo.set_viewport(0, 0, *event.size)
```


8. We create a callback function when the canvas needs to be refreshed. This `on_draw` function renders the entire scene. First, we clear the window in white (it is necessary to do that at every frame). Then, we draw a succession of line segments using our OpenGL program.


``` python
@c.connect
def on_draw(event):
    gloo.clear((1,1,1,1))
    program.draw('line_strip')
```


9. Finally, we show the canvas and we run the application.


``` python
c.show()
app.run();
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).

