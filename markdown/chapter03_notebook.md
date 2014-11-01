# notebook


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 3.1. Teaching programming in the notebook with IPython blocks

You need to install ipythonblocks for this recipe. You can just type in a terminal `pip install ipythonblocks`. Note that you can also execute this shell command from the IPython notebook by prefixing this command with `!`.


``` python
!pip install ipythonblocks
```


For the last part of this recipe, you also need to install Pillow: you will find more instructions in Chapter 11. (http://python-imaging.github.io)

Finally, you need to download the *Portrait* image on the [book's website](http://ipython-books.github.io) and extract it in the current directory. You can also play with your own images!

1. First, we import some modules.


``` python
import time
from IPython.display import clear_output
from ipythonblocks import BlockGrid, colors
```


2. Now, we create a **block grid** with 5 columns and 5 rows, and we fill each block in purple.


``` python
grid = BlockGrid(width=5, height=5, fill=colors['Purple'])
grid.show()
```


3. We can access individual blocks with 2D indexing. This illustrates the indexing syntax in Python. We can also access an entire row or line with `:` (colon). Each block is represented by an RGB color. The library comes with a handy dictionary of colors, assigning RGB tuples to standard color names.


``` python
grid[0,0] = colors['Lime']
grid[-1,0] = colors['Lime']
grid[:,-1] = colors['Lime']
grid.show()
```


4. Now, we are going to illustrate **matrix multiplication**, a fundamental notion in linear algebra. We will represent two $(n,n)$ matrices $A$ (in cyan) and $B$ (lime) aligned with $C=A \cdot B$ (yellow). To do this, we use a small trick consisting in creating a big white grid of size $(2n+1,2n+1)$. The matrices $A$, $B$ and $C$ are just *views* on parts of the grid.


``` python
n = 5
grid = BlockGrid(width=2*n+1, 
                 height=2*n+1, 
                 fill=colors['White'])
A = grid[n+1:,:n]
B = grid[:n,n+1:]
C = grid[n+1:,n+1:]
A[:,:] = colors['Cyan']
B[:,:] = colors['Lime']
C[:,:] = colors['Yellow']
grid.show()
```


5. Let's turn to matrix multiplication itself. We perform a loop over all rows and columns, and we highlight the corresponding rows and columns in $A$ and $B$ that are multiplied together during the matrix product. We combine IPython's `clear_output()` method with `grid.show()` and `time.sleep()` (pause) to implement the animation.


``` python
for i in range(n):
    for j in range(n):
        # We reset the matrix colors.
        A[:,:] = colors['Cyan']
        B[:,:] = colors['Lime']
        C[:,:] = colors['Yellow']
        # We highlight the adequate rows
        # and columns in red.
        A[i,:] = colors['Red']
        B[:,j] = colors['Red']
        C[i,j] = colors['Red']
        # We animate the grid in the loop.
        clear_output()
        grid.show()
        time.sleep(.25)
```


6. Finally, we will display an image with IPython blocks. We import the JPG image with `Image.open()` and we retrieve the data with `getdata()`.


``` python
from PIL import Image
imdata = Image.open('data/photo.jpg').getdata()
```


Now, we can create a `BlockGrid` with the appropriate number of rows and columns, and set each block's color to the corresponding pixel's color in the image. We use a small block size, and we remove the lines between the blocks.


``` python
rows, cols = imdata.size
grid = BlockGrid(width=rows, height=cols,
                 block_size=4, lines_on=False)
for block, rgb in zip(grid, imdata):
    block.rgb = rgb
grid.show()
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 3.2. Converting an IPython notebook to other formats with nbconvert

You need pandoc, a LateX distribution, and the Notebook dataset on the book's website. On Windows, you also need pywin32 (`conda install pywin32` if you use Anaconda).

1. Let's open the test notebook in the `data` folder. A notebook is just a plain text file (JSON), so we open it in text mode (`r` mode).


``` python
with open('data/test.ipynb', 'r') as f:
    contents = f.read()
print(len(contents))
```



``` python
print(contents[:345] + '...' + contents[-33:])
```


2. Now that we have loaded the notebook as a string, let's parse it with the `json` module.


``` python
import json
nb = json.loads(contents)
```


3. Let's have a look at the keys in the notebook dictionary.


``` python
print(nb.keys())
print('nbformat ' + str(nb['nbformat']) + 
      '.' + str(nb['nbformat_minor']))
```


The version of the notebook format is indicated in `nbformat` and `nbformat_minor`.

3. The main field is `worksheets`: there is only one by default. A worksheet contains a list of cells, and some metadata.


``` python
nb['worksheets'][0].keys()
```


4. Each cell has a type, optional metadata, some contents (text or code), possibly one or several outputs, and other information. Let's look at a Markdown cell and a code cell.


``` python
nb['worksheets'][0]['cells'][1]
```



``` python
nb['worksheets'][0]['cells'][2]
```


5. Once parsed, the notebook is represented as a Python dictionary. Manipulating it is therefore quite convenient in Python. Here, we count the number of Markdown and code cells.


``` python
cells = nb['worksheets'][0]['cells']
nm = len([cell for cell in cells
          if cell['cell_type'] == 'markdown'])
nc = len([cell for cell in cells
          if cell['cell_type'] == 'code'])
print(("There are {nm} Markdown cells and "
       "{nc} code cells.").format(
        nm=nm, nc=nc))
```


6. Let's have a closer look at the image output of the cell with the matplotlib figure.


``` python
png = cells[2]['outputs'][0]['png']
cells[2]['outputs'][0]['png'] = png[:20] + '...' + png[-20:]
cells[2]['outputs'][0]
```


In general, there can be zero, one, or multiple outputs. Besides, each output can have multiple representations. Here, the matplotlib figure has a PNG representation (the base64-encoded image) and a text representation (the internal representation of the figure).

7. Now, we are going to use nbconvert to convert our text notebook to other formats. This tool can be used from the command-line. Note that the API of nbconvert may change in future versions. Here, we convert the notebook to an HTML document.


``` python
!ipython nbconvert --to html data/test.ipynb
```


8. Let's display this document in an `<iframe>` (a small window showing an external HTML document within the notebook).


``` python
from IPython.display import IFrame
IFrame('test.html', 600, 200)
```


9. We can also convert the notebook to LaTeX and PDF. In order to specify the title and author of the document, we need to *extend* the default LaTeX template. First, we create a file `mytemplate.tplx` that extends the default `article.tplx` template provided by nbconvert. We precise the contents of the author and title blocks here.


``` python
%%writefile mytemplate.tplx
((*- extends 'article.tplx' -*))

((* block author *))
\author{Cyrille Rossant}
((* endblock author *))

((* block title *))
\title{My document}
((* endblock title *))
```


10. Then, we can run nbconvert by specifying our custom template.


``` python
!ipython nbconvert --to latex --template mytemplate data/test.ipynb
!pdflatex test.tex
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 3.3. Adding custom controls in the notebook toolbar

The CSS and Javascript of the HTML notebook can be customized through the files in `~/.ipython/profile_default/static/custom`, where `~` is your `HOME` directory, and `default` is your IPython profile. In this short recipe, we will use this feature to add a new button in the notebook toolbar on top of every notebook. Specifically, this button renumbers linearly all code cells.

1. First, we are going to inject Javascript code directly in the notebook. This is useful for testing purposes, or if you don't want your changes to be permanent. The Javascript code will be loaded with that notebook only. To do this, we can just use the `%%javascript` cell magic.


``` python
%%javascript
// This function allows us to add buttons 
// to the notebook toolbar.
IPython.toolbar.add_buttons_group([
{
    // The button's label.
    'label': 'renumber all code cells',
    
    // The button's icon.
    // See a list of Font-Awesome icons here:
    // http://fortawesome.github.io/Font-Awesome/icons/
    'icon': 'icon-list-ol',
    
    // The callback function.
    'callback': function () {
        
        // We retrieve the lists of all cells.
        var cells = IPython.notebook.get_cells();
        
        // We only keep the code cells.
        cells = cells.filter(function(c)
            {
                return c instanceof IPython.CodeCell; 
            })
        
        // We set the input prompt of all code cells.
        for (var i = 0; i < cells.length; i++) {
            cells[i].set_input_prompt(i + 1);
        }
    }
}]);
```


Running this code cell adds a button in the toolbar. Clicking on this button automatically updates the prompt numbers of all code cells.

2. To make these changes permanent, i.e. to add this button on every notebook open within the current profile, we can open the file `~/.ipython/profile_default/static/custom/custom.js` and add the following code:

```javascript
$([IPython.events]).on('app_initialized.NotebookApp',
                       function(){

    // Copy of the Javascript code above (step 1).
    IPython.toolbar.add_buttons_group([
    {
        // The button's label.
        'label': 'renumber all code cells',

        // The button's icon.
        // See a list of Font-Awesome icons here:
        // http://fortawesome.github.io/Font-Awesome/icons/
        'icon': 'icon-list-ol',

        // The callback function.
        'callback': function () {

            // We retrieve the lists of all cells.
            var cells = IPython.notebook.get_cells();

            // We only keep the code cells.
            cells = cells.filter(function(c)
                {
                    return c instanceof IPython.CodeCell; 
                })

            // We set the input prompt of all code cells.
            for (var i = 0; i < cells.length; i++) {
                cells[i].set_input_prompt(i + 1);
            }
        }
    }]);
});
```

The code put here will be automatically loaded as soon as a notebook page is loaded.

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 3.4. Customizing the CSS style in the notebook

You will need the *CSS* dataset on the book's website. (http://ipython-books.github.com)

You are expected to know a bit of CSS3 for this recipe. You can find many tutorials online (see references at the end of this recipe).

1. First, we create a new IPython profile to avoid messing with our regular profile.


``` python
!ipython profile create custom_css
```


2. Now, we retrieve in Python the path to this profile (this should be `~/.ipython`) and to the `custom.css` file (empty by default).


``` python
dir = !ipython locate profile custom_css
dir = dir[0]
```



``` python
import os
csspath = os.path.realpath(os.path.join(
            dir, 'static/custom/custom.css'))
```



``` python
csspath
```


3. We can now edit this file here. We change the background color, the font size of code cells, the border of some cells, and we highlight the selected cells in edit mode.


``` python
%%writefile {csspath}

body {
    /* Background color for the whole notebook. */
    background-color: #f0f0f0;
}

/* Level 1 headers. */
h1 {
    text-align: right;
    color: red;
}

/* Code cells. */
div.input_area > div.highlight > pre {
    font-size: 10px;
}

/* Output images. */
div.output_area img {
    border: 3px #ababab solid;
    border-radius: 8px;
}

/* Selected cells. */
div.cell.selected {
    border: 3px #ababab solid;
    background-color: #ddd;
}

/* Code cells in edit mode. */
div.cell.edit_mode {
    border: 3px red solid;
    background-color: #faa;
}
```


4. Opening a notebook with the `custom_css` profile leads to the following style:

5. We can also use this stylesheet with nbconvert. We just have to convert a notebook to a static HTML document, and copy the `custom.css` file in the same folder. Here, we use a test notebook that has been downloaded on this book's website (see *Getting started*).


``` python
!cp {csspath} custom.css
!ipython nbconvert --to html data/test.ipynb
```


Here is how this HTML document look like:


``` python
from IPython.display import IFrame
IFrame('test.html', 600, 650)
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 3.5. Using interactive widgets: a piano in the notebook

You need to download the *Piano* dataset on the book's website. (http://ipython-books.github.io)

This dataset contains synthetized piano notes obtained on `archive.org` (CC0 1.0 Universal licence). (https://archive.org/details/SynthesizedPianoNotes)

1. Let's import a few modules.


``` python
import numpy as np
import os
from IPython.display import Audio, display, clear_output
from IPython.html import widgets
from functools import partial
```


2. To create a piano, we will draw one button per note. The corresponding note plays when the user clicks on the button. This is implemented by displaying an `<audio>` element.


``` python
dir = 'data/synth'
```



``` python
# This is the list of notes.
notes = 'C,C#,D,D#,E,F,F#,G,G#,A,A#,B,C'.split(',')
```



``` python
def play(note, octave=0):
    """This function displays an HTML Audio element
    that plays automatically when it appears."""
    f = os.path.join(dir, 
         "piano_{i}.mp3".format(i=note+12*octave))
    clear_output()
    display(Audio(filename=f, autoplay=True))
```


3. We are going to place all buttons within a **container widget**. In IPython 2.0+, widgets can be organized hierarchically. One common use case is to organize several widgets in a given layout. Here, `piano` will contain 12 buttons for the 12 notes.


``` python
piano = widgets.ContainerWidget()
```


4. We create our first widget: a slider control that specifies the octave (0 or 1 here).


``` python
octave_slider = widgets.IntSliderWidget()
octave_slider.max = 1
octave_slider
```


5. Now, we create the buttons. There are several steps. First, we instantiate a `ButtonWidget` object. Then, we specify a callback function that plays the corresponding note (given by an index) at a given octave (given by the current value of the octave slider). Finally, we set the CSS of each button, notably the white or black color.


``` python
buttons = []
for i, note in enumerate(notes):
    button = widgets.ButtonWidget(description=note)
    
    def on_button_clicked(i, _):
        play(i+1, octave_slider.value)
        
    button.on_click(partial(on_button_clicked, i))
    
    button.set_css({'width': '30px', 
                    'height': '60px',
                    'padding': '0',
                    'color': ('black', 
                              'white')['#' in note],
                    'background': ('white', 'black')['#' in note],
                    'border': '1px solid black',
                    'float': 'left'})
    
    buttons.append(button)
```


6. Finally, we arrange all widgets with the containers. The `piano` container contains the buttons, and the main container (`container`) contains the slider and the piano.


``` python
piano.children = buttons
```



``` python
container = widgets.ContainerWidget()
container.children = [octave_slider,
                      piano]
```


By default, widgets are organized vertically within a container. Here, the octave slider will be above the piano.


``` python
display(container)
piano.remove_class('vbox')
piano.add_class('hbox')
```


Within the piano, we want all notes to be arranged horizontally. We do this by replacing the default `vbox` CSS class by the `hbox` class.

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 3.6. Creating a custom Javascript widget in the notebook: a spreadsheet editor for Pandas

You need IPython 2.0+ for this recipe. Besides, you need the [Handsontable](http://handsontable.com) Javascript library. Below are the instructions to load this Javascript library in the IPython notebook.

1. Go [here](https://github.com/warpech/jquery-handsontable/tree/master/dist).
2. Download `jquery.handsontable.full.css` and `jquery.handsontable.full.js`, and put these two files in `~\.ipython\profile_default\static\custom\`.
3. In this folder, add the following line in `custom.js`:
`require(['/static/custom/jquery.handsontable.full.js']);`
4. In this folder, add the following line in `custom.css`:
`@import "/static/custom/jquery.handsontable.full.css"`

Now, refresh the notebook!

1. Let's import a few functions and classes.


``` python
from IPython.html import widgets
from IPython.display import display
from IPython.utils.traitlets import Unicode
```


2. We create a new widget. The `value` trait will contain the JSON representation of the entire table. This trait will be synchronized between Python and Javascript thanks to IPython 2.0's widget machinery.


``` python
class HandsonTableWidget(widgets.DOMWidget):
    _view_name = Unicode('HandsonTableView', sync=True)
    value = Unicode(sync=True)
```


3. Now we write the Javascript code for the widget. The three important functions that are responsible for the synchronization are:

  * `render` for the widget initialization
  * `update` for Python to Javascript update
  * `handle_table_change` for Javascript to Python update


``` python
%%javascript
var table_id = 0;
require(["widgets/js/widget"], function(WidgetManager){    
    // Define the HandsonTableView
    var HandsonTableView = IPython.DOMWidgetView.extend({
        
        render: function(){
            // Initialization: creation of the HTML elements
            // for our widget.
            
            // Add a <div> in the widget area.
            this.$table = $('<div />')
                .attr('id', 'table_' + (table_id++))
                .appendTo(this.$el);
            // Create the Handsontable table.
            this.$table.handsontable({
            });
            
        },
        
        update: function() {
            // Python --> Javascript update.
            
            // Get the model's JSON string, and parse it.
            var data = $.parseJSON(this.model.get('value'));
            // Give it to the Handsontable widget.
            this.$table.handsontable({data: data});
            
            // Don't touch this...
            return HandsonTableView.__super__.update.apply(this);
        },
        
        // Tell Backbone to listen to the change event 
        // of input controls.
        events: {"change": "handle_table_change"},
        
        handle_table_change: function(event) {
            // Javascript --> Python update.
            
            // Get the table instance.
            var ht = this.$table.handsontable('getInstance');
            // Get the data, and serialize it in JSON.
            var json = JSON.stringify(ht.getData());
            // Update the model with the JSON string.
            this.model.set('value', json);
            
            // Don't touch this...
            this.touch();
        },
    });
    
    // Register the HandsonTableView with the widget manager.
    WidgetManager.register_widget_view(
        'HandsonTableView', HandsonTableView);
});
```


4. Now, we have a synchronized table widget that we can already use. But we'd like to integrate it with Pandas. To do this, we create a light wrapper around a `DataFrame` instance. We create two callback functions for synchronizing the Pandas object with the IPython widget. Changes in the GUI will automatically trigger a change in the `DataFrame`, but the converse is not true. We'll need to re-display the widget if we change the `DataFrame` in Python.


``` python
from io import StringIO  # Python 2: from StringIO import StringIO
import numpy as np
import pandas as pd
```



``` python
class HandsonDataFrame(object):
    def __init__(self, df):
        self._df = df
        self._widget = HandsonTableWidget()
        self._widget.on_trait_change(self._on_data_changed, 
                                     'value')
        self._widget.on_displayed(self._on_displayed)
        
    def _on_displayed(self, e):
        # DataFrame ==> Widget (upon initialization only)
        json = self._df.to_json(orient='values')
        self._widget.value = json
        
    def _on_data_changed(self, e, val):
        # Widget ==> DataFrame (called every time the user
        # changes a value in the graphical widget)
        buf = StringIO(val)
        self._df = pd.read_json(buf, orient='values')
        
    def to_dataframe(self):
        return self._df
        
    def show(self):
        display(self._widget)
```


5. Now, let's test all that! We first create a random `DataFrame`.


``` python
data = np.random.randint(size=(3, 5), low=100, high=900)
df = pd.DataFrame(data)
df
```


6. We wrap it in a `HandsonDataFrame` and show it.


``` python
ht = HandsonDataFrame(df)
ht.show()
```


7. We can now *change* the values interactively, and they will be changed in Python accordingly.


``` python
ht.to_dataframe()
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 3.7. Processing webcam images in real-time from the notebook

In this recipe, we show how to communicate data in both directions from the notebook to the Python kernel, and conversely. Specifically, we will retrieve the webcam feed from the browser using HTML5's `<video>` element, and pass it to Python in real time using the interactive capabilities of the IPython notebook 2.0+. This way, we can process the image in Python with an edge detector (implemented in scikit-image), and display it in the notebook in real time.

Most of the code for this recipe comes from [Jason Grout's example](https://github.com/jasongrout/ipywidgets).

1. We need to import quite a few modules.


``` python
from IPython.html.widgets import DOMWidget
from IPython.utils.traitlets import Unicode, Bytes, Instance
from IPython.display import display

from skimage import io, filter, color
import urllib
import base64
from PIL import Image
import StringIO
import numpy as np
from numpy import array, ndarray
import matplotlib.pyplot as plt
```


2. We define two functions to convert images from and to base64 strings. This conversion is a common way to pass binary data between processes (here, the browser and Python).


``` python
def to_b64(img):
    imgdata = StringIO.StringIO()
    pil = Image.fromarray(img)
    pil.save(imgdata, format='PNG')
    imgdata.seek(0)
    return base64.b64encode(imgdata.getvalue())
```



``` python
def from_b64(b64):
    im = Image.open(StringIO.StringIO(base64.b64decode(b64)))
    return array(im)
```


3. We define a Python function that will process the webcam image in real time. It accepts and returns a NumPy array. This function applies an edge detector with the `roberts()` function in scikit-image.


``` python
def process_image(image):
    img = filter.roberts(image[:,:,0]/255.)
    return (255-img*255).astype(np.uint8)
```


4. Now, we create a custom widget to handle the bidirectional communication of the video flow from the browser to Python and reciprocally.


``` python
class Camera(DOMWidget):
    _view_name = Unicode('CameraView', sync=True)
    
    # This string contains the base64-encoded raw
    # webcam image (browser -> Python).
    imageurl = Unicode('', sync=True)
    
    # This string contains the base64-encoded processed 
    # webcam image(Python -> browser).
    imageurl2 = Unicode('', sync=True)

    # This function is called whenever the raw webcam
    # image is changed.
    def _imageurl_changed(self, name, new):
        head, data = new.split(',', 1)
        if not data:
            return
        
        # We convert the base64-encoded string
        # to a NumPy array.
        image = from_b64(data)
        
        # We process the image.
        image = process_image(image)
        
        # We convert the processed image
        # to a base64-encoded string.
        b64 = to_b64(image)
        
        self.imageurl2 = 'data:image/png;base64,' + b64
```


5. The next step is to write the Javascript code for the widget.


``` python
%%javascript

var video        = $('<video>')[0];
var canvas       = $('<canvas>')[0];
var canvas2       = $('<img>')[0];
var width = 320;
var height = 0;

require(["widgets/js/widget"], function(WidgetManager){
    var CameraView = IPython.DOMWidgetView.extend({
        render: function(){
            var that = this;

            // We append the HTML elements.
            setTimeout(function() {
                that.$el.append(video).
                         append(canvas).
                         append(canvas2);}, 200);
            
            // We initialize the webcam.
            var streaming = false;
            navigator.getMedia = ( navigator.getUserMedia ||
                                 navigator.webkitGetUserMedia ||
                                 navigator.mozGetUserMedia ||
                                 navigator.msGetUserMedia);

            navigator.getMedia({video: true, audio: false},
                function(stream) {
                  if (navigator.mozGetUserMedia) {
                    video.mozSrcObject = stream;
                  } else {
                    var vendorURL = (window.URL || 
                                     window.webkitURL);
                    video.src = vendorURL.createObjectURL(
                        stream);
                  }
                    video.controls = true;
                  video.play();
                },
                function(err) {
                  console.log("An error occured! " + err);
                }
            );
            
            // We initialize the size of the canvas.
            video.addEventListener('canplay', function(ev){
                if (!streaming) {
                  height = video.videoHeight / (
                      video.videoWidth/width);
                  video.setAttribute('width', width);
                  video.setAttribute('height', height);
                  canvas.setAttribute('width', width);
                  canvas.setAttribute('height', height);
                  canvas2.setAttribute('width', width);
                  canvas2.setAttribute('height', height);
                    
                  streaming = true;
                }
            }, false);
            
            // Play/Pause functionality.
            var interval;
            video.addEventListener('play', function(ev){
                // We get the picture every 100ms.    
                interval = setInterval(takepicture, 100);
            })
            video.addEventListener('pause', function(ev){
                clearInterval(interval);
            })
            
            // This function is called at each time step.
            // It takes a picture and sends it to the model.
            function takepicture() {
                canvas.width = width; canvas.height = height;
                canvas2.width = width; canvas2.height = height;
                
                video.style.display = 'none';
                canvas.style.display = 'none';
                
                // We take a screenshot from the webcam feed and 
                // we put the image in the first canvas.
                canvas.getContext('2d').drawImage(video, 
                    0, 0, width, height);
                
                // We export the canvas image to the model.
                that.model.set('imageurl',
                               canvas.toDataURL('image/png'));
                that.touch();
            }
        },
        
        update: function(){
            // This function is called whenever Python modifies
            // the second (processed) image. We retrieve it and
            // we display it in the second canvas.
            var img = this.model.get('imageurl2');
            canvas2.src = img;
            return CameraView.__super__.update.apply(this);
        }
    });
    
    // Register the view with the widget manager.
    WidgetManager.register_widget_view('CameraView', 
                                       CameraView);
});
```


6. Finally, we create and display the widget.


``` python
c = Camera()
display(c)
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 3.7. Processing webcam images in real-time from the notebook

In this recipe, we show how to communicate data in both directions from the notebook to the Python kernel, and conversely. Specifically, we will retrieve the webcam feed from the browser using HTML5's `<video>` element, and pass it to Python in real time using the interactive capabilities of the IPython notebook 2.0+. This way, we can process the image in Python with an edge detector (implemented in scikit-image), and display it in the notebook in real time.

Most of the code for this recipe comes from [Jason Grout's example](https://github.com/jasongrout/ipywidgets).

1. We need to import quite a few modules.


``` python
from IPython.html.widgets import DOMWidget
from IPython.utils.traitlets import Unicode, Bytes, Instance
from IPython.display import display

from skimage import io, filter, color
import urllib
import base64
from PIL import Image
from io import BytesIO
import numpy as np
from numpy import array, ndarray
import matplotlib.pyplot as plt
```


2. We define two functions to convert images from and to base64 strings. This conversion is a common way to pass binary data between processes (here, the browser and Python).


``` python
def to_b64(img):
    imgdata = BytesIO()
    pil = Image.fromarray(img)
    pil.save(imgdata, format='PNG')
    imgdata.seek(0)
    return urllib.parse.quote(base64.b64encode(imgdata.getvalue()))
```



``` python
def from_b64(b64):
    im = Image.open(BytesIO(base64.b64decode(b64)))
    return array(im)
```


3. We define a Python function that will process the webcam image in real time. It accepts and returns a NumPy array. This function applies an edge detector with the `roberts()` function in scikit-image.


``` python
def process_image(image):
    img = filter.roberts(image[:,:,0]/255.)
    return (255-img*255).astype(np.uint8)
```


4. Now, we create a custom widget to handle the bidirectional communication of the video flow from the browser to Python and reciprocally.


``` python
class Camera(DOMWidget):
    _view_name = Unicode('CameraView', sync=True)
    
    # This string contains the base64-encoded raw
    # webcam image (browser -> Python).
    imageurl = Unicode('', sync=True)
    
    # This string contains the base64-encoded processed 
    # webcam image(Python -> browser).
    imageurl2 = Unicode('', sync=True)

    # This function is called whenever the raw webcam
    # image is changed.
    def _imageurl_changed(self, name, new):
        head, data = new.split(',', 1)
        if not data:
            return
        
        # We convert the base64-encoded string
        # to a NumPy array.
        image = from_b64(data)
        
        # We process the image.
        image = process_image(image)
        
        # We convert the processed image
        # to a base64-encoded string.
        b64 = to_b64(image)
        
        self.imageurl2 = 'data:image/png;base64,' + b64
```


5. The next step is to write the Javascript code for the widget.


``` python
%%javascript

var video        = $('<video>')[0];
var canvas       = $('<canvas>')[0];
var canvas2       = $('<img>')[0];
var width = 320;
var height = 0;

require(["widgets/js/widget"], function(WidgetManager){
    var CameraView = IPython.DOMWidgetView.extend({
        render: function(){
            var that = this;

            // We append the HTML elements.
            setTimeout(function() {
                that.$el.append(video).
                         append(canvas).
                         append(canvas2);}, 200);
            
            // We initialize the webcam.
            var streaming = false;
            navigator.getMedia = ( navigator.getUserMedia ||
                                 navigator.webkitGetUserMedia ||
                                 navigator.mozGetUserMedia ||
                                 navigator.msGetUserMedia);

            navigator.getMedia({video: true, audio: false},
                function(stream) {
                  if (navigator.mozGetUserMedia) {
                    video.mozSrcObject = stream;
                  } else {
                    var vendorURL = (window.URL || 
                                     window.webkitURL);
                    video.src = vendorURL.createObjectURL(
                        stream);
                  }
                    video.controls = true;
                  video.play();
                },
                function(err) {
                  console.log("An error occured! " + err);
                }
            );
            
            // We initialize the size of the canvas.
            video.addEventListener('canplay', function(ev){
                if (!streaming) {
                  height = video.videoHeight / (
                      video.videoWidth/width);
                  video.setAttribute('width', width);
                  video.setAttribute('height', height);
                  canvas.setAttribute('width', width);
                  canvas.setAttribute('height', height);
                  canvas2.setAttribute('width', width);
                  canvas2.setAttribute('height', height);
                    
                  streaming = true;
                }
            }, false);
            
            // Play/Pause functionality.
            var interval;
            video.addEventListener('play', function(ev){
                // We get the picture every 100ms.    
                interval = setInterval(takepicture, 100);
            })
            video.addEventListener('pause', function(ev){
                clearInterval(interval);
            })
            
            // This function is called at each time step.
            // It takes a picture and sends it to the model.
            function takepicture() {
                canvas.width = width; canvas.height = height;
                canvas2.width = width; canvas2.height = height;
                
                video.style.display = 'none';
                canvas.style.display = 'none';
                
                // We take a screenshot from the webcam feed and 
                // we put the image in the first canvas.
                canvas.getContext('2d').drawImage(video, 
                    0, 0, width, height);
                
                // We export the canvas image to the model.
                that.model.set('imageurl',
                               canvas.toDataURL('image/png'));
                that.touch();
            }
        },
        
        update: function(){
            // This function is called whenever Python modifies
            // the second (processed) image. We retrieve it and
            // we display it in the second canvas.
            var img = this.model.get('imageurl2');
            canvas2.src = img;
            return CameraView.__super__.update.apply(this);
        }
    });
    
    // Register the view with the widget manager.
    WidgetManager.register_widget_view('CameraView', 
                                       CameraView);
});
```


6. Finally, we create and display the widget.


``` python
c = Camera()
display(c)
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).

