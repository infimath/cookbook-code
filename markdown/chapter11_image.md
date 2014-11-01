# image


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 11.1. Manipulating the exposure of an image

You need scikit-image for this recipe. You will find the installation instructions [here](http://scikit-image.org/download.html).

You also need to download the *Beach* dataset. (http://ipython-books.github.io)

1. Let's import the packages.


``` python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import skimage.exposure as skie
%matplotlib inline

```


2. We open an image with matplotlib. We only take a single RGB component to have a grayscale image.


``` python
img = plt.imread('data/pic1.jpg')[...,0]
```


3. We create a function that displays the image along with its **histogram**.


``` python
def show(img):
    # Display the image.
    plt.figure(figsize=(8,2));
    plt.subplot(121);
    plt.imshow(img, cmap=plt.cm.gray);
    plt.axis('off');
    # Display the histogram.
    plt.subplot(122);
    plt.hist(img.ravel(), lw=0, bins=256);
    plt.xlim(0, img.max());
    plt.yticks([]);
    plt.show()
```


4. Let's display the image along with its histogram.


``` python
show(img)
```


The histogram is unbalanced and the image appears slightly over-exposed.

5. Now, we rescale the intensity of the image using scikit-image's `rescale_intensity` function. The `in_range` and `out_range` define a linear mapping from the original image to the modified image. The pixels that are outside `in_range` are clipped to the extremal values of `out_range`. Here, the darkest pixels (intensity less than 100) become completely black (0), whereas the brightest pixels (>240) become completely white (255).


``` python
show(skie.rescale_intensity(img,
     in_range=(100, 240), out_range=(0, 255)))
```


Many intensity values seem to be missing in the histogram, which reflects the poor quality of this exposure correction technique.

6. We now use a more advanced exposure correction technique called **Contrast Limited Adaptive Histogram Equalization**.


``` python
show(skie.equalize_adapthist(img))
```


The histogram seems more balanced, and the image now appears more contrasted.

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 11.2. Applying filters on an image

1. Let's import the packages.


``` python
import numpy as np
import matplotlib.pyplot as plt
import skimage
import skimage.filter as skif
import skimage.data as skid
import matplotlib as mpl
%matplotlib inline

```


2. We create a function that displays a grayscale image.


``` python
def show(img):
    plt.figure(figsize=(4,2));
    plt.imshow(img, cmap=plt.cm.gray);
    plt.axis('off')
    plt.show();
```


3. Now, we load the Lena image (bundled in scikit-image). We select a single RGB component to get a grayscale image.


``` python
img = skimage.img_as_float(skid.lena())[...,0]
```



``` python
show(img)
```


4. Let's apply a blurring **Gaussian filter** to the image.


``` python
show(skif.gaussian_filter(img, 5.))
```


5. We now apply a **Sobel filter** that enhances the edges in the image.


``` python
sobimg = skif.sobel(img)
show(sobimg)
```


6. We can threshold the filtered image to get a *sketch effect*. We obtain a binary image that only contains the edges. We use a notebook widget to find an adequate thresholding value.


``` python
from IPython.html import widgets
@widgets.interact(x=(0.01, .4, .005))
def edge(x):
    show(sobimg<x)
```


7. Finally, we add some noise to the image to illustrate the effect of a denoising filter.


``` python
img = skimage.img_as_float(skid.lena())
# We take a portion of the image to show the details.
img = img[200:-100, 200:-150]
# We add Gaussian noise.
img = np.clip(img + .3 * np.random.rand(*img.shape), 0, 1)
```



``` python
show(img)
```


8. The `denoise_tv_bregman` function implements total-variation denoising using split-Bregman optimization.


``` python
show(skimage.restoration.denoise_tv_bregman(img, 5.))
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 11.3. Segmenting an image

1. Let's import the packages.


``` python
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import coins
from skimage.filter import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import closing, square
from skimage.measure import regionprops, label
from skimage.color import lab2rgb
import matplotlib as mpl
%matplotlib inline
```


2. We create a function to display a grayscale image.


``` python
def show(img, cmap=None):
    cmap = cmap or plt.cm.gray
    plt.figure(figsize=(4,2));
    plt.imshow(img, cmap=cmap);
    plt.axis('off');
    plt.show();
```


3. We retrieve a test image bundled in scikit-image, showing various coins on a plain background.


``` python
img = coins()
```



``` python
show(img)
```


4. The first step to segment the image consists in finding an intensity threshold separating the (bright) coins from the (dark) background. **Otsu's method** defines a simple algorithm to find such a threshold automatically.


``` python
threshold_otsu(img)
```



``` python
show(img>107)
```


5. There appears to be a problem in the top left corner of the image, with part of the background being too bright. Let's use the notebook widgets to find a better threshold.


``` python
from IPython.html import widgets
@widgets.interact(t=(10, 240))
def threshold(t):
    show(img>t)
```


6. The threshold 120 looks better. The next step consists in cleaning the binary image by smoothing the coins and removing the border. Scikit-image contains a few functions for these purposes.


``` python
img_bin = clear_border(closing(img>120, square(5)))
show(img_bin)
```


7. Next, we perform the segmentation task itself with the `label` function. This function detects the connected components in the image, and attributes a unique label to every component. Here, we color-code the labels in the binary image.


``` python
labels = label(img_bin)
show(labels, cmap=plt.cm.rainbow)
```


8. Small artifacts in the image result in spurious labels that do not correspond to coins. Therefore we only keep components with more than 100 pixels. The `regionprops` function allows us to retrieve specific properties of the components (here, the area and the bounding box).


``` python
regions = regionprops(labels, 
                      ['Area', 'BoundingBox'])
boxes = np.array([label['BoundingBox'] for label in regions 
                                       if label['Area'] > 100])
print("There are {0:d} coins.".format(len(boxes)))
```


9. Finally, we show the label number on top of each component in the original image.


``` python
plt.figure(figsize=(6,4));
plt.imshow(img, cmap=plt.cm.gray);
plt.axis('off');
xs = boxes[:,[1,3]].mean(axis=1)
ys = boxes[:,[0,2]].mean(axis=1)
for i, box in enumerate(boxes):
    plt.text(xs[i]-5, ys[i]+5, str(i))
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 11.4. Finding points of interest in an image

You need to download the *Child* dataset on the book's website. (http://ipython-books.github.io)

1. Let's import the packages.


``` python
import numpy as np
import matplotlib.pyplot as plt
import skimage
import skimage.feature as sf
import matplotlib as mpl
%matplotlib inline

```


2. We create a function to display a colored or grayscale image.


``` python
def show(img, cmap=None):
    cmap = cmap or plt.cm.gray
    plt.figure(figsize=(4,2));
    plt.imshow(img, cmap=cmap);
    plt.axis('off');
```


3. We load an image.


``` python
img = plt.imread('data/pic2.jpg')
```



``` python
show(img)
```


4. We find salient points in the image with the Harris corner method. The first step consists in computing the **Harris corner measure response image** with the `corner_harris` function (we will explain this measure in *How it works...*).


``` python
corners = sf.corner_harris(img[:,:,0])
```



``` python
show(corners)
```


We see that the patterns in the child's coat are particularly well detected by this algorithm.

5. The next step consists in detecting corners from this measure image, using the `corner_peaks` function.


``` python
peaks = sf.corner_peaks(corners)
```



``` python
show(img)
plt.plot(peaks[:,1], peaks[:,0], 'or', ms=4);
```


6. Finally, we create a box around the corner points to define our region of interest.


``` python
ymin, xmin = peaks.min(axis=0)
ymax, xmax = peaks.max(axis=0)
w, h = xmax-xmin, ymax-ymin
```



``` python
k = .25
xmin -= k*w
xmax += k*w
ymin -= k*h
ymax += k*h
```



``` python
show(img[ymin:ymax,xmin:xmax])
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 11.5. Detecting faces in an image with OpenCV

You need OpenCV and the Python wrapper. You can find installation instructions on [OpenCV's website](http://docs.opencv.org/trunk/doc/py_tutorials/py_tutorials.html).

On Windows, you can install [Chris Gohlke's package](http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv).

You also need to download the *Family* dataset on the book's website. (http://ipython-books.github.io).

1. First, we import the packages.


``` python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
%matplotlib inline

```


2. We open the JPG image with OpenCV.


``` python
img = cv2.imread('data/pic3.jpg')
```


3. Then, we convert it to a grayscale image using OpenCV's `cvtColor` function.


``` python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```


4. To detect faces, we will use the **Viola–Jones object detection framework**. A cascade of Haar-like classifiers has been trained to detect faces. The result of the training is stored in a XML file (part of the *Family* dataset available on the book's website). We load this cascade from this XML file with OpenCV's `CascadeClassifier` class.


``` python
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
```


5. Finally, the `detectMultiScale` method of the classifier detects the objects on a grayscale image, and returns a list of rectangles around these objects.


``` python
for x, y, w, h in face_cascade.detectMultiScale(gray, 1.3):
    cv2.rectangle(gray, (x,y), (x+w,y+h), (255,0,0), 2)
plt.figure(figsize=(6,4));
plt.imshow(gray, cmap=plt.cm.gray);
plt.axis('off');
```


We see that, although all detected objects are indeed faces, one face out of four is not detected. This is probably due to the fact that this face is not perfectly facing the camera, whereas the faces in the training set were. This shows that the efficacy of this method is limited by the quality and generality of the training set.

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 11.6. Applying digital filters to speech sounds

> **PYTHON 3 VERSION**

You need the pydub package: you can install it with `pip install pydub`. (https://github.com/jiaaro/pydub/)

This package requires the open-source multimedia library FFmpeg for the decompression of MP3 files. (http://www.ffmpeg.org)

1. Let's import the packages.


``` python
import urllib
from io import BytesIO
import numpy as np
import scipy.signal as sg
import pydub
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import matplotlib as mpl
%matplotlib inline

```


2. We create a Python function to generate a sound from an English sentence. This function uses Google's Text-To-Speech (TTT) API. We retrieve the sound in mp3 format, and we convert it to the Wave format with pydub. Finally, we retrieve the raw sound data by removing the wave header with NumPy.


``` python
def speak(sentence):
    url = "http://translate.google.com/translate_tts?tl=en&q=" + \
          urllib.parse.quote_plus(sentence)
    req = urllib.request.Request(url, headers={'User-Agent': ''}) 
    mp3 = urllib.request.urlopen(req).read()
    # We convert the mp3 bytes to wav.
    audio = pydub.AudioSegment.from_mp3(BytesIO(mp3))
    wave = audio.export('_', format='wav')
    wave.seek(0)
    wave = wave.read()
    # We get the raw data by removing the 24 first bytes 
    # of the header.
    x = np.frombuffer(wave, np.int16)[24:] / 2.**15
    return x, audio.frame_rate
```


3. We create a function that plays a sound (represented by a NumPy vector) in the notebook, using IPython's `Audio` class.


``` python
def play(x, fr, autoplay=False):
    display(Audio(x, rate=fr, autoplay=autoplay))
```


4. Let's play the sound "Hello world". We also display the waveform with matplotlib.


``` python
x, fr = speak("Hello world")
play(x, fr)
plt.figure(figsize=(6,3));
t = np.linspace(0., len(x)/fr, len(x))
plt.plot(t, x, lw=1);
```


5. Now, we will hear the effect of a Butterworth low-pass filter applied to this sound (500 Hz cutoff frequency).


``` python
b, a = sg.butter(4, 500./(fr/2.), 'low')
x_fil = sg.filtfilt(b, a, x)
```



``` python
play(x_fil, fr)
plt.figure(figsize=(6,3));
plt.plot(t, x, lw=1);
plt.plot(t, x_fil, lw=1);
```


We hear a muffled voice.

6. And now with a high-pass filter (1000 Hz cutoff frequency).


``` python
b, a = sg.butter(4, 1000./(fr/2.), 'high')
x_fil = sg.filtfilt(b, a, x)
```



``` python
play(x_fil, fr)
plt.figure(figsize=(6,3));
plt.plot(t, x, lw=1);
plt.plot(t, x_fil, lw=1);
```


It sounds like a phone call.

7. Finally, we can create a simple widget to quickly test the effect of a high-pass filter with an arbitrary cutoff frequency.


``` python
from IPython.html import widgets
@widgets.interact(t=(100., 5000., 100.))
def highpass(t):
    b, a = sg.butter(4, t/(fr/2.), 'high')
    x_fil = sg.filtfilt(b, a, x)
    play(x_fil, fr, autoplay=True)
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 11.6. Applying digital filters to speech sounds

> **PYTHON 2 VERSION**

You need the pydub package: you can install it with `pip install pydub`. (https://github.com/jiaaro/pydub/)

This package requires the open-source multimedia library FFmpeg for the decompression of MP3 files. (http://www.ffmpeg.org)

1. Let's import the packages.


``` python
import urllib
import urllib2
import cStringIO
import numpy as np
import scipy.signal as sg
import pydub
import matplotlib.pyplot as plt
from IPython.display import Audio, display
%matplotlib inline
```


2. We create a Python function to generate a sound from an English sentence. This function uses Google's Text-To-Speech (TTT) API. We retrieve the sound in mp3 format, and we convert it to the Wave format with pydub. Finally, we retrieve the raw sound data by removing the wave header with NumPy.


``` python
def speak(sentence):
    url = "http://translate.google.com/translate_tts?tl=en&q=" + \
          urllib.quote_plus(sentence)
    req = urllib2.Request(url, headers={'User-Agent': ''}) 
    mp3 = urllib2.urlopen(req).read()
    # We convert the mp3 bytes to wav.
    audio = pydub.AudioSegment.from_mp3(cStringIO.StringIO(mp3))
    wave = audio.export(cStringIO.StringIO(), format='wav')
    wave.reset()
    wave = wave.read()
    # We get the raw data by removing the 24 first bytes 
    # of the header.
    x = np.frombuffer(wave, np.int16)[24:] / 2.**15
    return x, audio.frame_rate
```


3. We create a function that plays a sound (represented by a NumPy vector) in the notebook, using IPython's `Audio` class.


``` python
def play(x, fr, autoplay=False):
    display(Audio(x, rate=fr, autoplay=autoplay))
```


4. Let's play the sound "Hello world". We also display the waveform with matplotlib.


``` python
x, fr = speak("Hello world")
play(x, fr)
plt.figure(figsize=(6,3));
t = np.linspace(0., len(x)/fr, len(x))
plt.plot(t, x, lw=1);
```


5. Now, we will hear the effect of a Butterworth low-pass filter applied to this sound (500 Hz cutoff frequency).


``` python
b, a = sg.butter(4, 500./(fr/2.), 'low')
x_fil = sg.filtfilt(b, a, x)
```



``` python
play(x_fil, fr)
plt.figure(figsize=(6,3));
plt.plot(t, x, lw=1);
plt.plot(t, x_fil, lw=1);
```


We hear a muffled voice.

6. And now with a high-pass filter (1000 Hz cutoff frequency).


``` python
b, a = sg.butter(4, 1000./(fr/2.), 'high')
x_fil = sg.filtfilt(b, a, x)
```



``` python
play(x_fil, fr)
plt.figure(figsize=(6,3));
plt.plot(t, x, lw=1);
plt.plot(t, x_fil, lw=1);
```


It sounds like a phone call.

7. Finally, we can create a simple widget to quickly test the effect of a high-pass filter with an arbitrary cutoff frequency.


``` python
from IPython.html import widgets
@widgets.interact(t=(100., 5000., 100.))
def highpass(t):
    b, a = sg.butter(4, t/(fr/2.), 'high')
    x_fil = sg.filtfilt(b, a, x)
    play(x_fil, fr, autoplay=True)
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 11.7. Creating a sound synthesizer in the notebook

1. We import NumPy, matplotlib, and various IPython packages and objects.


``` python
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display, clear_output
from IPython.html import widgets
from functools import partial
import matplotlib as mpl
%matplotlib inline

```


2. We define the sampling rate and the duration of the notes.


``` python
rate = 16000.
duration = .5
t = np.linspace(0., duration, rate * duration)
```


3. We create a function that generates and plays the sound of a note (sine function) at a given frequency, using NumPy and IPython's `Audio` class.


``` python
def synth(f):
    x = np.sin(f * 2. * np.pi * t)
    display(Audio(x, rate=rate, autoplay=True))
```


4. Here is the fundamental 440 Hz note.


``` python
synth(440)
```


5. Now, we generate the note frequencies of our piano. The chromatic scale is obtained by a geometric progression with common ratio $2^{1/12}$.


``` python
notes = zip('C,C#,D,D#,E,F,F#,G,G#,A,A#,B,C'.split(','),
            440. * 2 ** (np.arange(3, 17) / 12.))
```


6. Finally, we create the piano with the notebook widgets. Each note is a button, and all buttons are contained in an horizontal box container. Clicking on one note plays a sound at the corresponding frequency.


``` python
container = widgets.ContainerWidget()
buttons = []
for note, f in notes:
    button = widgets.ButtonWidget(description=note)
    def on_button_clicked(f, b):
        clear_output()
        synth(f)
    button.on_click(partial(on_button_clicked, f))
    button.set_css({'width': '30px', 
                    'height': '60px',
                    'padding': '0',
                    'color': ('black', 'white')['#' in note],
                    'background': ('white', 'black')['#' in note],
                    'border': '1px solid black',
                    'float': 'left'})
    buttons.append(button)
container.children = buttons
display(container)
container.remove_class('vbox')
container.add_class('hbox')
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).

