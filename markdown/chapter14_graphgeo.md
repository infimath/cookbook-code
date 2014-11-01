# graphgeo


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 14.1. Manipulating and visualizing graphs with NetworkX

In this recipe, we show how to create, manipulate and visualize graphs with NetworkX.

You can find the installation instructions of NetworkX on the official documentation. (http://networkx.github.io/documentation/latest/install.html)

In brief, you can just execute `pip install networkx`. On Windows, you can also use Chris Gohlke's installer. (http://www.lfd.uci.edu/~gohlke/pythonlibs/#networkx)

1. Let's import NumPy, NetworkX, and matplotlib.


``` python
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
%matplotlib inline
```


2. There are many different ways to create a graph. Here, we create a list of edges.


``` python
n = 10  # Number of nodes in the graph.
# Each node is connected to the two next nodes,
# in a circular fashion.
adj = [(i, (i+1)%n) for i in range(n)]
adj += [(i, (i+2)%n) for i in range(n)]
```


3. We instantiate a `Graph` object here, giving the list of edges as argument.


``` python
g = nx.Graph(adj)
```


4. Let's check the list of nodes and edges of the graph, and its adjacency matrix.


``` python
print(g.nodes())
```



``` python
print(g.edges())
```



``` python
print(nx.adjacency_matrix(g))
```


5. Now, we will display this graph. NetworkX comes with a variety of drawing functions. A graph being an abstract mathematical object only describing relations between items, there is no canonical way to display one. One needs to either specify the positions of the nodes explicitly, or an algorithm to compute an "interesting" layout. Here, we use the `draw_circular` function that simply positions nodes linearly on a circle.


``` python
plt.figure(figsize=(4,4));
nx.draw_circular(g)
```


6. Graphs can be modified easily. Here, we add a new node connected to all existing nodes. We also specify a `color` attribute to this node. In NetworkX, every node and edge comes with a Python dictionary containing arbitrary attributes. This feature is extremely useful in practice.


``` python
g.add_node(n, color='#fcff00')
# We add an edge from every existing 
# node to the new node.
for i in range(n):
    g.add_edge(i, n)
```


7. Now, let's draw the modified graph again. This time, we specify the nodes' positions and colors explicitly.


``` python
plt.figure(figsize=(4,4));
# We define custom node positions on a circle
# except the last node which is at the center.
t = np.linspace(0., 2*np.pi, n)
pos = np.zeros((n+1, 2))
pos[:n,0] = np.cos(t)
pos[:n,1] = np.sin(t)
# A node's color is specified by its 'color'
# attribute, or a default color if this attribute
# doesn't exist.
color = [g.node[i].get('color', '#88b0f3')
         for i in range(n+1)]
# We now draw the graph with matplotlib.
nx.draw_networkx(g, pos=pos, node_color=color)
plt.axis('off');
```


8. Let's also use an automatic layout algorithm.


``` python
plt.figure(figsize=(4,4));
nx.draw_spectral(g, node_color=color)
plt.axis('off');
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 14.2. Analyzing a social network with NetworkX

First, you need the *twitter* Python package for this recipe. You can install it with `pip install twitter`. (https://pypi.python.org/pypi/twitter)

Then, you need to obtain authentication codes in order to access Twitter data. The procedure is free. In addition to a Twitter account, you also need to create an *Application* on the Twitter Developers website. Then, you will be able to retrieve the OAuth authentication codes that are required for this recipe. (https://dev.twitter.com/apps)

**Note that access to the Twitter API is not unlimited**. Most methods can only be called a few times within a given time window. Unless you study small networks or look at small portions of large networks, you will need to throttle your requests. In this recipe, we only consider a small portion of the network, so that the API limit should not be reached. Otherwise, you will have to wait a few minutes before the next time window starts. (https://dev.twitter.com/docs/rate-limiting/1.1/limits)

1. Let's import a few packages.


``` python
import math
import json
import twitter
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
%matplotlib inline
```


2.  You need to create a `twitter.txt` text file in the current folder with the four authentication keys. You will find those in your Twitter Developers Application page, *OAuth tool* section. (https://dev.twitter.com/apps)


``` python
(CONSUMER_KEY, CONSUMER_SECRET, 
OAUTH_TOKEN, OAUTH_TOKEN_SECRET) = open('twitter.txt', 'r').read().splitlines()
```


3. We now create the `Twitter` instance that will give us access to the Twitter API.


``` python
auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET,
                           CONSUMER_KEY, CONSUMER_SECRET)
tw = twitter.Twitter(auth=auth)
```


4. We use the latest 1.1 version of the Twitter API in this recipe. The `twitter` library defines a direct mapping between the REST API and the attributes of the `Twitter` instance. Here, we execute the `account/verify_credentials` REST request to obtain the user id of the authenticated user (me here, or you if you execute this notebook yourself!).


``` python
me = tw.account.verify_credentials()
```



``` python
myid = me['id']
```


5. Let's define a simple function that returns the identifiers of all followers of a given user (the authenticated user by default).


``` python
def get_followers_ids(uid=None):
    # Retrieve the list of followers' ids of the specified user.
    return tw.followers.ids(user_id=uid)['ids']
```



``` python
# We get the list of my followers.
my_followers_ids = get_followers_ids()
```


6. Now, we define a function that retrieves the full profile of Twitter users. Since the `users/lookup` batch request is limited to 100 users per call, and that only a small number of calls are allowed within a time window, we only look at a subset of all the followers.


``` python
def get_users_info(users_ids, max=500):
    n = min(max, len(users_ids))
    # Get information about those users, using batch requests.
    users = [tw.users.lookup(user_id=users_ids[100*i:100*(i+1)])
             for i in range(int(math.ceil(n/100.)))]
    # We flatten this list of lists.
    users = [item for sublist in users for item in sublist]
    return {user['id']: user for user in users}
```



``` python
users_info = get_users_info(my_followers_ids)
```



``` python
# Let's save this dictionary on the disk.
with open('my_followers.json', 'w') as f:
    json.dump(users_info, f, indent=1)
```


7. We also start to define the graph with the followers, as an adjacency list (technically, a dictionary of lists). This is called the **ego graph**. This graph represents all *following* connections between my followers.


``` python
adjacency = {myid: my_followers_ids}
```


8. Now, we are going to take a look at the part of the ego graph related to Python. Specifically, we will consider the followers of the 10 most followed users whom descriptions contain "Python".


``` python
my_followers_python = [user for user in users_info.values()
                       if 'python' in user['description'].lower()]
```



``` python
my_followers_python_best = sorted(my_followers_python, 
        key=lambda u: u['followers_count'])[::-1][:10]
```


The request for retrieving the followers of a given user is rate-limited. Let's check how many calls remaining we have.


``` python
tw.application.rate_limit_status(resources='followers') \
             ['resources']['followers']['/followers/ids']
```



``` python
for user in my_followers_python_best:
    # The call to get_followers_ids is rate-limited.
    adjacency[user['id']] = list(set(get_followers_ids(
        user['id'])).intersection(my_followers_ids))
```


9. Now that our graph is defined as an adjacency list in a dictionary, we will load it in NetworkX.


``` python
g = nx.Graph(adjacency)
```



``` python
# We only restrict the graph to the users for which we 
# were able to retrieve the profile.
g = g.subgraph(users_info.keys())
```



``` python
# We also save this graph on disk.
with open('my_graph.json', 'w') as f:
    json.dump(nx.to_dict_of_lists(g), f, indent=1)
```



``` python
# We remove isolated nodes for simplicity.
g.remove_nodes_from([k for k, d in g.degree().items()
                     if d <= 1])
```



``` python
# Since I am connected to all nodes, by definition,
# we can remove me for simplicity.
g.remove_nodes_from([myid])
```


10. Let's take a look at the graph's statistics.


``` python
len(g.nodes()), len(g.edges())
```


11. We are now going to plot this graph. We will use different sizes and colors for the nodes, according to the number of followers and the number of tweets for each user. Most followed users will be bigger. Most active users will be redder.


``` python
# Update the dictionary.
deg = g.degree()
for user in users_info.values():
    fc = user['followers_count']
    sc = user['statuses_count']
    # Is this user a Pythonista?
    user['python'] = 'python' in user['description'].lower()
    # We compute the node size as a function of the 
    # number of followers.
    user['node_size'] = math.sqrt(1 + 10 * fc)
    # The color is function of its activity on Twitter.
    user['node_color'] = 10 * math.sqrt(1 + sc)
    # We only display the name of the most followed users.
    user['label'] = user['screen_name'] if fc > 2000 else ''
```


12. Finally, we use the `draw` function to display the graph. We need to specify the node sizes and colors as lists, and the labels as a dictionary.


``` python
node_size = [users_info[uid]['node_size'] for uid in g.nodes()]
```



``` python
node_color = [users_info[uid]['node_color'] for uid in g.nodes()]
```



``` python
labels = {uid: users_info[uid]['label'] for uid in g.nodes()}
```



``` python
plt.figure(figsize=(6,6));
nx.draw(g, cmap=plt.cm.OrRd, alpha=.8,
        node_size=node_size, node_color=node_color,
        labels=labels, font_size=4, width=.1);
```


## See also

* Manipulate and visualize graphs with NetworkX

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 14.3. Resolving dependencies in a Directed Acyclic Graph with a topological sort

You need the `python-apt` package in order to build the package dependency graph. (https://pypi.python.org/pypi/python-apt/)

We also assume that this notebook is executed on a Debian system (like Ubuntu). If you don't have such a system, you can download the data *Debian* directly on the book's website. Extract it in the current directory, and start this notebook directly at step 7. (http://ipython-books.github.io)

1. We import the `apt` module and we build the list of packages.


``` python
import json
import apt
cache = apt.Cache()
```


2. The `graph` dictionary will contain the adjacency list of a small portion of the dependency graph.


``` python
graph = {}
```


3. We define a function that returns the list of dependencies of a package.


``` python
def get_dependencies(package):
    if package not in cache:
        return []
    pack = cache[package]
    ver = pack.candidate or pack.versions[0]
    # We flatten the list of dependencies,
    # and we remove the duplicates.
    return sorted(set([item.name 
            for sublist in ver.dependencies 
            for item in sublist]))
```


4. We now define a *recursive* function that builds the dependency graph for a particular package. This function updates the `graph` variable.


``` python
def get_dep_recursive(package):
    if package not in cache:
        return []
    if package not in graph:
        dep = get_dependencies(package)
        graph[package] = dep
    for dep in graph[package]:
        if dep not in graph:
            graph[dep] = get_dep_recursive(dep)
    return graph[package]
```


5. Let's build the dependency graph for IPython.


``` python
get_dep_recursive('ipython');
```


6. Finally, let's save the adjacency list in JSON.


``` python
with open('data/apt.json', 'w') as f:
    json.dump(graph, f, indent=1)
```


7. We import a few packages.


``` python
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
%matplotlib inline

```


8. Let's load the adjacency list from the JSON file.


``` python
with open('data/apt.json', 'r') as f:
    graph = json.load(f)
```


9. Now, we create a directed graph (`DiGraph` in NetworkX) from our adjacency list. We reverse the graph to get a more natural ordering.


``` python
g = nx.DiGraph(graph).reverse()
```


10. A topological sort only exists when the graph is a **directed acyclic graph** (DAG). This means that there is no cycle in the graph, i.e. no circular dependency here. Is our graph a DAG?


``` python
nx.is_directed_acyclic_graph(g)
```


11. What are the packages responsible for the cycles? We can find it out with the `simple_cycles` function.


``` python
set([cycle[0] for cycle in nx.simple_cycles(g)])
```


12. Here, we can try to remove these packages. In an actual package manager, these cycles need to be carefully taken into account.


``` python
g.remove_nodes_from(_)
```



``` python
nx.is_directed_acyclic_graph(g)
```


13. The graph is now a DAG. Let's display it first.


``` python
ug = g.to_undirected()
deg = ug.degree()
```



``` python
plt.figure(figsize=(4,4));
# The size of the nodes depends on the number of dependencies.
nx.draw(ug, font_size=6, 
        node_size=[20*deg[k] for k in ug.nodes()]);
```


14. Finally, we can perform the topological sort, thereby obtaining a linear installation order satisfying all dependencies.


``` python
nx.topological_sort(g)
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 14.4. Computing connected components in an image

1. Let's import the packages.


``` python
import itertools
import numpy as np
import networkx as nx
import matplotlib.colors as col
import matplotlib.pyplot as plt
%matplotlib inline

```


2. We create a $10 \times 10$ image where each pixel can take one of three possible labels (or colors).


``` python
n = 10
```



``` python
img = np.random.randint(size=(n, n), 
                        low=0, high=3)
```


3. Now, we create the underlying 2D grid graph encoding the structure of the image. Each node is a pixel, and a node is connected to its nearest neighbors. NetworkX defines a function `grid_2d_graph` for generating this graph.


``` python
g = nx.grid_2d_graph(n, n)
```


4. Let's create two functions to display the image and the corresponding graph.


``` python
def show_image(img, **kwargs):
    plt.imshow(img, origin='lower',interpolation='none',
               **kwargs);
    plt.axis('off');
```



``` python
def show_graph(g, **kwargs):
    nx.draw(g,
            pos={(i, j): (j, i) for (i, j) in g.nodes()},
            node_color=[img[i, j] for (i, j) in g.nodes()],
            linewidths=1, edge_color='w', with_labels=False,
            node_size=30, **kwargs);
```



``` python
cmap = plt.cm.Blues
```


5. Here is the original image superimposed with the underlying graph.


``` python
plt.figure(figsize=(5,5));
show_image(img, cmap=cmap, vmin=-1);
show_graph(g, cmap=cmap, vmin=-1);
```


6. We are now going to find all the contiguous regions of the image in dark blue containing more than three pixels. The first step is to consider the *subgraph* corresponding to all dark blue pixels.


``` python
g2 = g.subgraph(zip(*np.nonzero(img==2)))
```



``` python
plt.figure(figsize=(5,5));
show_image(img, cmap=cmap, vmin=-1);
show_graph(g2, cmap=cmap, vmin=-1);
```


7. We see that the requested contiguous regions correspond to the *connected components* containing more than three nodes in the subgraph. We can use the `connected_components` function of NetworkX to find those components.


``` python
components = [np.array(comp)
              for comp in nx.connected_components(g2)
              if len(comp)>=3]
len(components)
```


8. Finally, let's assign a new color to each of these components, and let's display the new image.


``` python
# We copy the image, and assign a new label
# to each found component.
img_bis = img.copy()
for i, comp in enumerate(components):
    img_bis[comp[:,0], comp[:,1]] = i + 3
```



``` python
# We create a new discrete color map extending
# the previous map with new colors.
colors = [cmap(.5), cmap(.75), cmap(1.), 
          '#f4f235', '#f4a535', '#f44b35']
cmap2 = col.ListedColormap(colors, 'indexed')
```



``` python
plt.figure(figsize=(5,5));
show_image(img_bis, cmap=cmap2);
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 14.5. Computing the Voronoi diagram of a set of points

You will need the Smopy module to display the OpenStreetMap map of Paris. You can install this package with `pip install smopy`.

Download the *Metro* dataset on the book's website and extract it in the current directory. (https://ipython-books.github.io)

1. Let's import NumPy, Pandas, matplotlib, and SciPy, which implements a Voronoi diagram algorithm.


``` python
import numpy as np
import pandas as pd
import scipy.spatial as spatial
import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib as mpl
import smopy
%matplotlib inline

```


2. Let's load the dataset with Pandas.


``` python
df = pd.read_csv('data/ratp.csv', sep='#', header=None)
```



``` python
df[df.columns[1:]].tail(3)
```


3. The DataFrame contains the coordinates, name, city, district and type of stations from RATP (the society that manages public transportations in Paris). Let's select the metro stations only.


``` python
metro = df[(df[5] == 'metro')]
```



``` python
metro[metro.columns[1:]].tail(3)
```


4. We are going to extract the district number of the stations that are located in Paris. With Pandas, we can use vectorized string operations using the `str` attribute of the corresponding column.


``` python
# We only extract the district from stations in Paris.
paris = metro[4].str.startswith('PARIS').values
```



``` python
# We create a vector of integers with the district number of
# the corresponding station, or 0 if the station is not
# in Paris.
districts = np.zeros(len(paris), dtype=np.int32)
districts[paris] = metro[4][paris].str.slice(6, 8).astype(np.int32)
districts[~paris] = 0
ndistricts = districts.max() + 1
```


5. We also extract the coordinates of all metro stations.


``` python
lon = metro[1]
lat = metro[2]
```


6. Now, let's retrieve the map of Paris with OpenStreetMap. We specify the map's boundaries with the extreme latitude and longitude coordinates of all our metro stations. We use the lightweight *smopy* module for generating the map.


``` python
box = (lat[paris].min(), lon[paris].min(), 
       lat[paris].max(), lon[paris].max())
m = smopy.Map(box, z=12)
```


7. We now compute the Voronoi diagram of the stations using SciPy. A `Voronoi` object is created with the points coordinates, and contains several attributes we will use for display.


``` python
vor = spatial.Voronoi(np.c_[lat, lon])
```


8. We need to define a custom function to display the Voronoi diagram. Although SciPy comes with its own display function, it does not take infinite points into account. This function can be used as is every time you need to display a Voronoi diagram.


``` python
def voronoi_finite_polygons_2d(vor, radius=None):
    """Reconstruct infinite Voronoi regions in a 
    2D diagram to finite regions.
    Source: http://stackoverflow.com/questions/20515554/colorize-voronoi-diagram
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()
    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, 
                                  vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue
        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue
            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        # Sort region counterclockwise.
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        new_regions.append(new_region.tolist())
    return new_regions, np.asarray(new_vertices)
```


9. The `voronoi_finite_polygons_2d` function returns a list of regions and a list of vertices. Every region is a list of vertex indices. The coordinates of all vertices are stored in `vertices`. From these structures, we can create a list of *cells*. Every cell represents a polygon as an array of vertex coordinates. We also use the `to_pixels` method of the `smopy.Map` instance which converts latitude and longitude geographical coordinates to pixels in the image.


``` python
regions, vertices = voronoi_finite_polygons_2d(vor)
cells = [m.to_pixels(vertices[region]) for region in regions]
```


10. Now we compute the color of every polygon.


``` python
cmap = plt.cm.Set3
# We generate colors for districts using a color map.
colors_districts = cmap(np.linspace(0., 1., ndistricts))[:,:3]
# The color of every polygon, grey by default.
colors = .25 * np.ones((len(districts), 3))
# We give each polygon in Paris the color of its district.
colors[paris] = colors_districts[districts[paris]]
```


11. Finally, we can display the map with the Voronoi diagram, using the `show_mpl` method of the `Map` instance.


``` python
ax = m.show_mpl();
ax.add_collection(mpl.collections.PolyCollection(cells,
                  facecolors=colors, edgecolors='k',
                  alpha=.35,));
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 14.6. Manipulating geospatial data with Shapely and basemap

In order to run this recipe, you will need the following packages:

* [GDAL/OGR](http://www.gdal.org/ogr/)
* [fiona](http://toblerity.org/fiona/README.html)
* [Shapely](http://toblerity.org/shapely/project.html)
* [descartes](https://pypi.python.org/pypi/descartes)
* [basemap](http://matplotlib.org/basemap/)

On Windows, you can find binary installers for all those packages except descartes on Chris Gohlke's webpage. (http://www.lfd.uci.edu/~gohlke/pythonlibs/)

Installing descartes is easy: `pip install descartes`.

On other systems, you can find installation instructions on the projects' websites. GDAL/OGR is a C++ library that is required by fiona. The other packages are regular Python packages.

Finally, you need to download the *Africa* dataset on the book's website. (http://ipython-books.github.io)

1. Let's import all the packages.


``` python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as col
from mpl_toolkits.basemap import Basemap
import fiona
import shapely.geometry as geom
from descartes import PolygonPatch
%matplotlib inline

```


2. We load the Shapefile dataset with fiona. This dataset notably contains the borders of all countries in the world.


``` python
# natural earth data
countries = fiona.open("data/ne_10m_admin_0_countries.shp")
```


3. We select the countries in Africa.


``` python
africa = [c for c in countries 
          if c['properties']['CONTINENT'] == 'Africa']
```


4. Now, we create a basemap map showing the African continent.


``` python
m = Basemap(llcrnrlon=-23.03,
            llcrnrlat=-37.72,
            urcrnrlon=55.20,
            urcrnrlat=40.58)
```


5. We need to convert the geographical coordinates of the countries' borders in map coordinates, so that we can display then in basemap.


``` python
def _convert(poly, m):
    if isinstance(poly, list):
        return [_convert(_, m) for _ in poly]
    elif isinstance(poly, tuple):
        return m(*poly)
```



``` python
for _ in africa:
    _['geometry']['coordinates'] = _convert(
        _['geometry']['coordinates'], m)
```


6. The next step is to create matplotlib `PatchCollection` objects from the Shapefile dataset loaded with fiona. We use Shapely and descartes to do this.


``` python
def get_patch(shape, **kwargs):
    """Return a matplotlib PatchCollection from a geometry 
    object loaded with fiona."""
    # Simple polygon.
    if isinstance(shape, geom.Polygon):
        return col.PatchCollection([PolygonPatch(shape, **kwargs)],
                                   match_original=True)
    # Collection of polygons.
    elif isinstance(shape, geom.MultiPolygon):
        return col.PatchCollection([PolygonPatch(c, **kwargs)
                                    for c in shape],
                                   match_original=True)
```



``` python
def get_patches(shapes, fc=None, ec=None, **kwargs):
    """Return a list of matplotlib PatchCollection objects
    from a Shapefile dataset."""
    # fc and ec are the face and edge colors of the countries.
    # We ensure these are lists of colors, with one element
    # per country.
    if not isinstance(fc, list):
        fc = [fc] * len(shapes)
    if not isinstance(ec, list):
        ec = [ec] * len(shapes)
    # We convert each polygon to a matplotlib PatchCollection
    # object.
    return [get_patch(geom.shape(s['geometry']), 
                      fc=fc_, ec=ec_, **kwargs) 
            for s, fc_, ec_ in zip(shapes, fc, ec)]
```


7. We also define a function to get countries colors depending on a specific field in the Shapefile dataset. Indeed, our dataset not only contains countries borders, but also a few administrative, economical and geographical properties for each country. Here, we will choose the color according to the population and GDP of the countries.


``` python
def get_colors(field, cmap):
    """Return one color per country, depending on a specific
    field in the dataset."""
    values = [country['properties'][field] for country in africa]
    values_max = max(values)
    return [cmap(v / values_max) for v in values]
```


8. Finally, we display the maps. We display the coast lines with basemap, and the countries with our Shapefile dataset.


``` python
plt.figure(figsize=(8,6));
# Display the countries color-coded with their population.
ax = plt.subplot(121);
m.drawcoastlines();
patches = get_patches(africa, 
                      fc=get_colors('POP_EST', plt.cm.Reds), 
                      ec='k')
for p in patches:
    ax.add_collection(p)
plt.title("Population");
# Display the countries color-coded with their population.
ax = plt.subplot(122);
m.drawcoastlines();
patches = get_patches(africa, 
                      fc=get_colors('GDP_MD_EST', plt.cm.Blues), 
                      ec='k')
for p in patches:
    ax.add_collection(p)
plt.title("GDP");
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 14.7. Creating a route planner for road network

You need NetworkX and Smopy for this recipe. In order for NetworkX to read Shapefile datasets, you might need GDAL/OGR. You can find more information in the previous recipe.

You also need to download the *Road* dataset on the book's website. (http://ipython-books.github.io)

1. Let's import the packages.


``` python
import networkx as nx
import numpy as np
import pandas as pd
import json
import smopy
import matplotlib.pyplot as plt
%matplotlib inline

```


2. We load the data (a Shapefile dataset) with NetworkX. This dataset contains detailed information about the primary roads in California. NetworkX's `read_shp` function returns a graph, where each node is a geographical position, and each edge contains information about the road linking the two nodes. The data comes from the [United States Census Bureau website](http://www.census.gov/geo/maps-data/data/tiger.html).


``` python
g = nx.read_shp("data/tl_2013_06_prisecroads.shp")
```


3. This graph is not necessarily connected, but we need a connected graph in order to compute shortest paths. Here, we take the largest connected subgraph using the `connected_component_subgraphs` function.


``` python
sgs = list(nx.connected_component_subgraphs(
           g.to_undirected()))
largest = np.argmax([len(sg) 
                     for sg in sgs])
sg = sgs[largest]
len(sg)
```


4. We define two positions (latitude and longitude). We will find the shortest path between these two positions.


``` python
pos0 = (36.6026, -121.9026)
pos1 = (34.0569, -118.2427)
```


5. Each edge in the graph contains information about the road, including a list of points along this road. We first create a function that returns this array of coordinates, for any edge in the graph.


``` python
def get_path(n0, n1):
    """If n0 and n1 are connected nodes in the graph, this function
    return an array of point coordinates along the road linking
    these two nodes."""
    return np.array(json.loads(sg[n0][n1]['Json'])['coordinates'])
```


6. We will notably use the road path to compute its length. We first need to define a function that computes the distance between any two points in geographical coordinates.


``` python
# http://stackoverflow.com/questions/8858838/need-help-calculating-geographical-distance
EARTH_R = 6372.8
def geocalc(lat0, lon0, lat1, lon1):
    """Return the distance (in km) between two points in 
    geographical coordinates."""
    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0)
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    dlon = lon0 - lon1
    y = np.sqrt(
        (np.cos(lat1) * np.sin(dlon)) ** 2
         + (np.cos(lat0) * np.sin(lat1) 
         - np.sin(lat0) * np.cos(lat1) * np.cos(dlon)) ** 2)
    x = np.sin(lat0) * np.sin(lat1) + \
        np.cos(lat0) * np.cos(lat1) * np.cos(dlon)
    c = np.arctan2(y, x)
    return EARTH_R * c
```


7. We can now define a function that returns the length of a path.


``` python
def get_path_length(path):
    return np.sum(geocalc(path[1:,0], path[1:,1],
                          path[:-1,0], path[:-1,1]))
```


8. Now, we update our graph by computing the distance between any two connected nodes. We add this information in the `distance` attribute of the edges.


``` python
# Compute the length of the road segments.
for n0, n1 in sg.edges_iter():
    path = get_path(n0, n1)
    distance = get_path_length(path)
    sg.edge[n0][n1]['distance'] = distance
```


9. The last step before we can find the shortest path in the graph, is to find the two nodes in the graph that are closest to the two requested positions.


``` python
nodes = np.array(sg.nodes())
# Get the closest nodes in the graph.
pos0_i = np.argmin(np.sum((nodes[:,::-1] - pos0)**2, axis=1))
pos1_i = np.argmin(np.sum((nodes[:,::-1] - pos1)**2, axis=1))
```


10. Now, we use NetworkX's `shortest_path` function to compute the shortest path between our two positions. We specify that the weight of every edge is the length of the road between them.


``` python
# Compute the shortest path.
path = nx.shortest_path(sg, 
                        source=tuple(nodes[pos0_i]), 
                        target=tuple(nodes[pos1_i]),
                        weight='distance')
len(path)
```


11. The itinerary has been computed. The `path` variable contains the list of edges that form the shortest path between our two positions. Now, we can get information about the itinerary with Pandas. The dataset has a few fields of interest, including the name and type (State, Interstate, etc.) of the roads.


``` python
roads = pd.DataFrame([sg.edge[path[i]][path[i + 1]] 
                      for i in range(len(path) - 1)], 
                     columns=['FULLNAME', 'MTFCC', 
                              'RTTYP', 'distance'])
roads
```


Here is the total length of this itinerary.


``` python
roads['distance'].sum()
```


12. Finally, let display the itinerary on the map. We first retrieve the map with Smopy.


``` python
map = smopy.Map(pos0, pos1, z=7, margin=.1)
```


13. Our path contains connected nodes in the graph. Every edge between two nodes is characterized by a list of points (constituting a part of the road). Therefore, we need to define a function that concatenates the positions along every edge in the path. A difficulty is that we need to concatenate the positions in the right order along our path. We choose the order based on the fact that the last point in an edge needs to be close to the first point in the next edge.


``` python
def get_full_path(path):
    """Return the positions along a path."""
    p_list = []
    curp = None
    for i in range(len(path)-1):
        p = get_path(path[i], path[i+1])
        if curp is None:
            curp = p
        if np.sum((p[0]-curp)**2) > np.sum((p[-1]-curp)**2):
            p = p[::-1,:]
        p_list.append(p)
        curp = p[-1]
    return np.vstack(p_list)
```


14. We convert the path in pixels in order to display it on the Smopy map.


``` python
linepath = get_full_path(path)
x, y = map.to_pixels(linepath[:,1], linepath[:,0])
```


15. Finally, we display the map, with our two positions and the computed itinerary between them.


``` python
plt.figure(figsize=(6,6));
map.show_mpl();
# Plot the itinerary.
plt.plot(x, y, '-k', lw=1.5);
# Mark our two positions.
plt.plot(x[0], y[0], 'ob', ms=10);
plt.plot(x[-1], y[-1], 'or', ms=10);
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).

