# best practices


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 2.7. Writing unit tests with nose

**This is the Python 3 version of the recipe.**

Although Python has a native unit testing module (unittest), we will rather use Nose which is more convenient and more powerful. Having an extra dependency is not a problem as the Nose package is only required to launch the test suite, and not to use the software itself.

## Creation of the Python module


``` python
%%writefile datautils.py
# Version 1.
import os
from urllib.request import urlopen  # Python 2: use urllib2

def download(url):
    """Download a file and save it in the current folder.
    Return the name of the downloaded file."""
    # Get the filename.
    file = os.path.basename(url)
    # Download the file unless it already exists.
    if not os.path.exists(file):
        with open(file, 'w') as f:
            f.write(urlopen(url).read())
    return file
```


## Creation of the test module


``` python
%%writefile test_datautils.py
from urllib.request import (HTTPHandler, install_opener, 
                            build_opener, addinfourl)
import os
import shutil
import tempfile
from io import StringIO  # Python 2: use StringIO
from datautils import download

TEST_FOLDER = tempfile.mkdtemp()
ORIGINAL_FOLDER = os.getcwd()

class TestHTTPHandler(HTTPHandler):
    """Mock HTTP handler."""
    def http_open(self, req):
        resp = addinfourl(StringIO('test'), '', req.get_full_url(), 200)
        resp.msg = 'OK'
        return resp
    
def setup():
    """Install the mock HTTP handler for unit tests."""
    install_opener(build_opener(TestHTTPHandler))
    os.chdir(TEST_FOLDER)
    
def teardown():
    """Restore the normal HTTP handler."""
    install_opener(build_opener(HTTPHandler))
    # Go back to the original folder.
    os.chdir(ORIGINAL_FOLDER)
    # Delete the test folder.
    shutil.rmtree(TEST_FOLDER)

def test_download1():
    file = download("http://example.com/file.txt")
    # Check that the file has been downloaded.
    assert os.path.exists(file)
    # Check that the file contains the contents of the remote file.
    with open(file, 'r') as f:
        contents = f.read()
    print(contents)
    assert contents == 'test'
```


## Launching the tests


``` python
!nosetests
```


## Adding a failing test

Now, let's add a new test.


``` python
%%writefile test_datautils.py -a

def test_download2():
    file = download("http://example.com/")
    assert os.path.exists(file)
```



``` python
!nosetests
```


## Fixing the failing test

The new test fails because the filename cannot be inferred from the URL, so we need to handle this case.


``` python
%%writefile datautils.py
# Version 2.
import os
from urllib.request import urlopen  # Python 2: use urllib2

def download(url):
    """Download a file and save it in the current folder.
    Return the name of the downloaded file."""
    # Get the filename.
    file = os.path.basename(url)
    # Fix the bug, by specifying a fixed filename if the URL 
    # does not contain one.
    if not file:
        file = 'downloaded'
    # Download the file unless it already exists.
    if not os.path.exists(file):
        with open(file, 'w') as f:
            f.write(urlopen(url).read())
    return file
```



``` python
!nosetests
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 2.7. Writing unit tests with nose

**This is the Python 2 version of the recipe.**

Although Python has a native unit testing module (unittest), we will rather use Nose which is more convenient and more powerful. Having an extra dependency is not a problem as the Nose package is only required to launch the test suite, and not to use the software itself.

## Creation of the Python module


``` python
%%writefile datautils.py
# Version 1.
import os
from urllib2 import urlopen  # Python 3: use urllib.request

def download(url):
    """Download a file and save it in the current folder.
    Return the name of the downloaded file."""
    # Get the filename.
    file = os.path.basename(url)
    # Download the file unless it already exists.
    if not os.path.exists(file):
        with open(file, 'w') as f:
            f.write(urlopen(url).read())
    return file
```


## Creation of the test module


``` python
%%writefile test_datautils.py
from urllib2 import (HTTPHandler, install_opener, 
                            build_opener, addinfourl)
import os
import shutil
import tempfile
from StringIO import StringIO  # Python 3: use io
from datautils import download

TEST_FOLDER = tempfile.mkdtemp()
ORIGINAL_FOLDER = os.getcwd()

class TestHTTPHandler(HTTPHandler):
    """Mock HTTP handler."""
    def http_open(self, req):
        resp = addinfourl(StringIO('test'), '', req.get_full_url(), 200)
        resp.msg = 'OK'
        return resp
    
def setup():
    """Install the mock HTTP handler for unit tests."""
    install_opener(build_opener(TestHTTPHandler))
    os.chdir(TEST_FOLDER)
    
def teardown():
    """Restore the normal HTTP handler."""
    install_opener(build_opener(HTTPHandler))
    # Go back to the original folder.
    os.chdir(ORIGINAL_FOLDER)
    # Delete the test folder.
    shutil.rmtree(TEST_FOLDER)

def test_download1():
    file = download("http://example.com/file.txt")
    # Check that the file has been downloaded.
    assert os.path.exists(file)
    # Check that the file contains the contents of the remote file.
    with open(file, 'r') as f:
        contents = f.read()
    print(contents)
    assert contents == 'test'
```


## Launching the tests


``` python
!nosetests
```


## Adding a failing test

Now, let's add a new test.


``` python
%%writefile test_datautils.py -a

def test_download2():
    file = download("http://example.com/")
    assert os.path.exists(file)
```



``` python
!nosetests
```


## Fixing the failing test

The new test fails because the filename cannot be inferred from the URL, so we need to handle this case.


``` python
%%writefile datautils.py
# Version 2.
import os
from urllib2 import urlopen  # Python 3: use urllib.request

def download(url):
    """Download a file and save it in the current folder.
    Return the name of the downloaded file."""
    # Get the filename.
    file = os.path.basename(url)
    # Fix the bug, by specifying a fixed filename if the URL 
    # does not contain one.
    if not file:
        file = 'downloaded'
    # Download the file unless it already exists.
    if not os.path.exists(file):
        with open(file, 'w') as f:
            f.write(urlopen(url).read())
    return file
```



``` python
!nosetests
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).

