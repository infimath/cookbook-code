# signal


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 10.1. Analyzing the frequency components of a signal with a Fast Fourier Transform

Download the *Weather* dataset on the book's website. (http://ipython-books.github.io)

The data has been obtained here: http://www.ncdc.noaa.gov/cdo-web/datasets#GHCND.

1. Let's import the packages, including `scipy.fftpack` which includes many FFT-related routines.


``` python
import datetime
import numpy as np
import scipy as sp
import scipy.fftpack
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```


2. We import the data from the CSV file. The number `-9999` is used for N/A values. Pandas can easily handle this. In addition, we tell Pandas to parse dates contained in the `DATE` column.


``` python
df0 = pd.read_csv('data/weather.csv', 
                  na_values=(-9999), 
                  parse_dates=['DATE'])
```



``` python
df = df0[df0['DATE']>='19940101']
```



``` python
df.head()
```


3. Each row contains the precipitation and extremal temperatures recorded one day by one weather station in France. For every date, we want to get a single average temperature for the whole country. The `groupby` method provided by Pandas lets us do that easily. We also remove any NA value.


``` python
df_avg = df.dropna().groupby('DATE').mean()
```



``` python
df_avg.head()
```


4. Now, we get the list of dates and the list of corresponding temperature. The unit is in tenth of degree, and we get the average value between the minimal and maximal temperature, which explains why we divide by 20.


``` python
date = df_avg.index.to_datetime()
temp = (df_avg['TMAX'] + df_avg['TMIN']) / 20.
N = len(temp)
```


5. Let's take a look at the evolution of the temperature.


``` python
plt.figure(figsize=(6,3));
plt.plot_date(date, temp, '-', lw=.5);
plt.ylim(-10, 40);
plt.xlabel('Date');
plt.ylabel('Mean temperature');
```


6. We now compute the Fourier transform and the spectral density of the signal. The first step is to compute the FFT of the signal using the `fft` function.


``` python
temp_fft = sp.fftpack.fft(temp)
```


7. Once the FFT has been obtained, one needs to take the square of its absolute value to get the **power spectral density** (PSD).


``` python
temp_psd = np.abs(temp_fft) ** 2
```


8. The next step is to get the frequencies corresponding to the values of the PSD. The `fftfreq` utility function does just that. It takes as input the length of the PSD vector, as well as the frequency unit. Here, we choose an annual unit: a frequency of 1 corresponds to 1 year (365 days). We provide `1./365` because the original unit is in days.


``` python
fftfreq = sp.fftpack.fftfreq(len(temp_psd), 1./365)
```


9. The `fftfreq` function returns positive and negative frequencies. We are only interested in positive frequencies here since we have a real signal (this will be explained in *How it works...*).


``` python
i = fftfreq>0
```


10. We now plot the power spectral density of our signal, as a function of the frequency (in unit of `1/year`).


``` python
plt.figure(figsize=(8,4));
plt.plot(fftfreq[i], 10*np.log10(temp_psd[i]));
plt.xlim(0, 5);
plt.xlabel('Frequency (1/year)');
plt.ylabel('PSD (dB)');
```


We observe a peak for $f=1/year$: this is because the fundamental frequency of the signal is the yearly variation of the temperature.

11. Now, we cut out the frequencies higher than the fundamental frequency.


``` python
temp_fft_bis = temp_fft.copy()
temp_fft_bis[np.abs(fftfreq) > 1.1] = 0
```


12. The next step is to perform an **inverse FFT** to convert the modified Fourier transform back to the temporal domain. This way, we recover a signal that mainly contains the fundamental frequency, as shown in the figure below.


``` python
temp_slow = np.real(sp.fftpack.ifft(temp_fft_bis))
```



``` python
plt.figure(figsize=(6,3));
plt.plot_date(date, temp, '-', lw=.5);
plt.plot_date(date, temp_slow, '-');
plt.xlim(datetime.date(1994, 1, 1), datetime.date(2000, 1, 1));
plt.ylim(-10, 40);
plt.xlabel('Date');
plt.ylabel('Mean temperature');
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 10.2. Applying a linear filter to a digital signal

Download the *Nasdaq* dataset on the book's website. (http://ipython-books.github.io)

The data has been obtained here: http://finance.yahoo.com/q/hp?s=^IXIC&a=00&b=1&c=1990&d=00&e=1&f=2014&g=d

1. Let's import the packages.


``` python
import numpy as np
import scipy as sp
import scipy.signal as sg
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```


2 We load the Nasdaq data with Pandas.


``` python
nasdaq_df = pd.read_csv('data/nasdaq.csv')
```



``` python
nasdaq_df.head()
```


3. Let's extract two columns: the date, and the daily closing value.


``` python
date = pd.to_datetime(nasdaq_df['Date'])
nasdaq = nasdaq_df['Close']
```


4. Let's take a look at the raw signal.


``` python
plt.figure(figsize=(6,4));
plt.plot_date(date, nasdaq, '-');
```


5. Now, we will follow a first approach to get the slow component of the signal's variations. We will convolve the signal with a triangular window: this corresponds to a **FIR filter**. We will explain the idea behind this method in *How it works...*. Let's just say for now that we replace each value with a weighted mean of the signal around that value.


``` python
# We get a triangular window with 60 samples.
h = sg.get_window('triang', 60)
# We convolve the signal with this window.
fil = sg.convolve(nasdaq, h/h.sum())
```



``` python
plt.figure(figsize=(6,4));
# We plot the original signal...
plt.plot_date(date, nasdaq, '-', lw=1);
# ... and the filtered signal.
plt.plot_date(date, fil[:len(nasdaq)], '-');
```


6. Now, let's use another method. We create an IIR Butterworth low-pass filter to extract the slow variations of the signal. The `filtfilt` method allows us to apply a filter forward and backward in order to avoid phase delays.


``` python
plt.figure(figsize=(6,4));
plt.plot_date(date, nasdaq, '-', lw=1);
# We create a 4-th order Butterworth low-pass filter.
b, a = sg.butter(4, 2./365)
# We apply this filter to the signal.
plt.plot_date(date, sg.filtfilt(b, a, nasdaq), '-');
```


7. Finally, we now use the same method to create a high-pass filter and extract the *fast* variations of the signal.


``` python
plt.figure(figsize=(6,4));
plt.plot_date(date, nasdaq, '-', lw=1);
b, a = sg.butter(4, 2*5./365, btype='high')
plt.plot_date(date, sg.filtfilt(b, a, nasdaq), '-', lw=.5);
```


The fast variations around 2000 correspond to the **dot-com bubble burst**, reflecting the high market volatility and the fast fluctuations of the stock market indices at that time. (http://en.wikipedia.org/wiki/Dot-com_bubble)

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 10.3. Computing the autocorrelation of a time series

Download the *Babies* dataset on the book's website. (http://ipython-books.github.io)

1. We import the packages.


``` python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```


2. We read the data with Pandas. The dataset contains one CSV file per year. Each file contains all baby names given that year with the respective frequencies. We load the data in a dictionary, containing one `DataFrame` per year.


``` python
files = [file for file in os.listdir('data/') 
         if file.startswith('yob')]
```



``` python
years = np.array(sorted([int(file[3:7]) 
                         for file in files]))
```



``` python
data = {year: 
        pd.read_csv('data/yob{y:d}.txt'.format(y=year), 
                    index_col=0, header=None, 
                    names=['First name', 'Gender', 'Number']) 
        for year in years}
```



``` python
data[2012].head()
```


3. We write functions to retrieve the frequencies of baby names as a function of the name, gender, and birth year.


``` python
def get_value(name, gender, year):
    """Return the number of babies born a given year, with a 
    given gender and a given name."""
    try:
        return data[year][data[year]['Gender'] == gender] \
               ['Number'][name]
    except KeyError:
        return 0
```



``` python
def get_evolution(name, gender):
    """Return the evolution of a baby name over the years."""
    return np.array([get_value(name, gender, year) 
                     for year in years])
```


4. Let's define a function that computes the autocorrelation of a signal. This function is essentially based on NumPy's `correlate` function.


``` python
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]
```


5. Now, we create a function that displays the evolution of a baby name, as well as its autocorrelation.


``` python
def autocorr_name(name, gender, color):
    x = get_evolution(name, gender)
    z = autocorr(x)
    # Evolution of the name.
    plt.subplot(121);
    plt.plot(years, x, '-o'+color, label=name);
    plt.title("Baby names");
    # Autocorrelation.
    plt.subplot(122);
    plt.plot(z / float(z.max()), '-'+color, label=name);
    plt.legend();
    plt.title("Autocorrelation");
```


6. Let's take a look at two female names.


``` python
plt.figure(figsize=(12,4));
autocorr_name('Olivia', 'F', 'k');
autocorr_name('Maria', 'F', 'y');
```


The autocorrelation of Olivia is decaying much faster than Maria's. This is mainly because of the steep increase of the name Olivia at the end of the twentieth century. By contrast, the name Maria is evolving more slowly globally, and its autocorrelation is decaying somewhat slower.

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).

