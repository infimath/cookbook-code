# stats


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 7.1. Explore a dataset with Pandas and matplotlib

1. We import NumPy, Pandas and matplotlib.


``` python
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```


2. The dataset is a CSV file, i.e. a text file with comma-separated values. Pandas lets us load this file with a single function.


``` python
player = 'Roger Federer'
filename = "data/{name}.csv".format(
              name=player.replace(' ', '-'))
df = pd.read_csv(filename)
```


The loaded data is a `DataFrame`, a 2D tabular data where each row is an observation, and each column is a variable. We can have a first look at this dataset by just displaying it in the IPython notebook.


``` python
df
```


3. There are many columns. Each row corresponds to a match played by Roger Federer. Let's add a boolean variable indicating whether he has won the match or not. The `tail` method displays the last rows of the column.


``` python
df['win'] = df['winner'] == player
df['win'].tail()
```


4. `df['win']` is a `Series` object: it is very similar to a NumPy array, except that each value has an index (here, the match index). This object has a few standard statistical functions. For example, let's look at the proportion of matches won.


``` python
print("{player} has won {vic:.0f}% of his ATP matches.".format(
      player=player, vic=100*df['win'].mean()))
```


5. Now, we are going to look at the evolution of some variables across time. The `start date` field contains the start date of the tournament as a string. We can convert the type to a date type using the `pd.to_datetime` function.


``` python
date = pd.to_datetime(df['start date'])
```


6. We are now looking at the proportion of double faults in each match (taking into account that there are logically more double faults in longer matches!). This number is an indicator of the player's state of mind, his level of self-confidence, his willingness to take risks while serving, and other parameters.


``` python
df['dblfaults'] = (df['player1 double faults'] / 
                   df['player1 total points total'])
```


7. We can use the `head` and `tail` methods to take a look at the beginning and the end of the column, and `describe` to get summary statistics. In particular, let's note that some rows have `NaN` values (i.e. the number of double faults is not available for all matches).


``` python
df['dblfaults'].tail()
```



``` python
df['dblfaults'].describe()
```


8. A very powerful feature in Pandas is `groupby`. This function allows us to group together rows that have the same value in a particular column. Then, we can aggregate this group-by object to compute statistics in each group. For instance, here is how we can get the proportion of wins as a function of the tournament's surface.


``` python
df.groupby('surface')['win'].mean()
```


9. Now, we are going to display the proportion of double faults as a function of the tournament date, as well as the yearly average. To do this, we also use `groupby`.


``` python
gb = df.groupby('year')
```


10. `gb` is a `GroupBy` instance. It is similar to a `DataFrame`, but there are multiple rows per group (all matches played in each year). We can aggregate those rows using the `mean` operation. We use matplotlib's `plot_date` function because the x-axis contains dates.


``` python
plt.figure(figsize=(8, 4))
plt.plot_date(date.astype(datetime), df['dblfaults'], alpha=.25, lw=0);
plt.plot_date(gb['start date'].max(), 
              gb['dblfaults'].mean(), '-', lw=3);
plt.xlabel('Year');
plt.ylabel('Proportion of double faults per match.');
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 7.2. Getting started with statistical hypothesis testing: a simple z-test

Many frequentist methods for hypothesis testing roughly involve the following steps:

1. Writing down the hypotheses, notably the **null hypothesis** which is the *opposite* of the hypothesis you want to prove (with a certain degree of confidence).
2. Computing a **test statistics**, a mathematical formula depending on the test type, the model, the hypotheses, and the data.
3. Using the computed value to accept the hypothesis, reject it, or fail to conclude.
  
Here, we flip a coin $n$ times and we observe $h$ heads. We want to know whether the coin is fair (null hypothesis). This example is extremely simple yet quite good for pedagogical purposes. Besides, it is the basis of many more complex methods.

We denote by $\mathcal B(q)$ the Bernoulli distribution with unknown parameter $q$ (http://en.wikipedia.org/wiki/Bernoulli_distribution). A Bernoulli variable:

* is 0 (tail) with probability $1-q$,
* is 1 (head) with probability $q$.

1. Let's suppose that, after $n=100$ flips, we get $h=61$ heads. We choose a significance level of 0.05: is the coin fair or not? Our null hypothesis is: *the coin is fair* ($q = 1/2$).


``` python
import numpy as np
import scipy.stats as st
import scipy.special as sp
```



``` python
n = 100  # number of coin flips
h = 61  # number of heads
q = .5  # null-hypothesis of fair coin
```


2. Let's compute the **z-score**, which is defined by the following formula (`xbar` is the estimated average of the distribution). We will explain this formula in the next section *How it works...*


``` python
xbar = float(h)/n
z = (xbar - q) * np.sqrt(n / (q*(1-q))); z
```


3. Now, from the z-score, we can compute the p-value as follows:


``` python
pval = 2 * (1 - st.norm.cdf(z)); pval
```


4. This p-value is less than 0.05, so we reject the null hypothesis and conclude that *the coin is probably not fair*.

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 7.3. Getting started with Bayesian methods

Let $q$ be the probability of obtaining a head. Whereas $q$ was just a fixed number in the previous recipe, we consider here that it is a *random variable*. Initially, this variable follows a distribution called the **prior distribution**. It represents our knowledge about $q$ *before* we start flipping the coin. We will update this distribution after each trial (**posterior distribution**).

1. First, we assume that $q$ is a *uniform* random variable on the interval $[0, 1]$. That's our prior distribution: for all $q$, $P(q)=1$.
2. Then, we flip our coin $n$ times. We note $x_i$ the outcome of the $i$-th flip ($0$ for tail, $1$ for head).
3. What is the probability distribution of $q$ knowing the observations $x_i$? **Bayes' formula** allows us to compute the *posterior distribution* analytically (see the next section for the mathematical details):

$$P(q | \{x_i\}) = \frac{P(\{x_i\} | q) P(q)}{\displaystyle\int_0^1 P(\{x_i\} | q) P(q) dq} = (n+1)\binom n h  q^h (1-q)^{n-h}$$

We define the posterior distribution according to the mathematical formula above. We remark this this expression is $(n+1)$ times the *probability mass function* (PMF) of the binomial distribution, which is directly available in `scipy.stats`. (http://en.wikipedia.org/wiki/Binomial_distribution)


``` python
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
%matplotlib inline
```



``` python
posterior = lambda n, h, q: (n+1) * st.binom(n, q).pmf(h)
```


Let's plot this distribution for an observation of $h=61$ heads and $n=100$ total flips.


``` python
n = 100
h = 61
q = np.linspace(0., 1., 1000)
d = posterior(n, h, q)
```



``` python
plt.figure(figsize=(5,3));
plt.plot(q, d, '-k');
plt.xlabel('q parameter');
plt.ylabel('Posterior distribution');
plt.ylim(0, d.max()+1);
```


4. This distribution indicates the plausible values for $q$ given the observations. We could use it to derive a **credible interval**, likely to contain the actual value. (http://en.wikipedia.org/wiki/Credible_interval)

We can also derive a point estimate. For example, the **maximum a posteriori (MAP) estimation** consists in considering the *maximum* of this distribution as an estimate for $q$. We can find this maximum analytically or numerically. Here, we find analytically $\hat q = h/n$, which looks quite sensible. (http://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation)

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 7.4. Estimating the correlation between two variables with a contingency table and a chi-square test

You need to download the *Tennis* dataset on the book's website, and extract it in the current directory. (http://ipython-books.github.io)

1. Let's import NumPy, Pandas, SciPy.stats and matplotlib.


``` python
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
%matplotlib inline
```


2. We load the dataset corresponding to Roger Federer.


``` python
player = 'Roger Federer'
filename = "data/{name}.csv".format(
              name=player.replace(' ', '-'))
df = pd.read_csv(filename)
```


3. This is a particularly rich dataset. Each row corresponds to a match, and the 70 columns contain many player characteristics during that match.


``` python
print("Number of columns: " + str(len(df.columns)))
df[df.columns[:4]].tail()
```


4. Here, we only look at the proportion of points won, and the (relative) number of aces.


``` python
npoints = df['player1 total points total']
points = df['player1 total points won'] / npoints
aces = df['player1 aces'] / npoints
```



``` python
plt.plot(points, aces, '.');
plt.xlabel('% of points won');
plt.ylabel('% of aces');
plt.xlim(0., 1.);
plt.ylim(0.);
```


If the two variables were independent, we would not see any trend in the cloud of points. On this plot, it is a bit hard to tell. Let's use Pandas to compute a coefficient correlation.

5. We create a new `DataFrame` with only those fields (note that this step is not compulsory). We also remove the rows where one field is missing.


``` python
df_bis = pd.DataFrame({'points': points,
                       'aces': aces}).dropna()
df_bis.tail()
```


6. Let's compute the Pearson's correlation coefficient between the relative number of aces in the match, and the number of points won.


``` python
df_bis.corr()
```


A correlation of ~0.26 seems to indicate a positive correlation between our two variables. In other words, the more aces in a match, the more points the player wins (which is not very surprising!).

7. Now, to determine if there is a *statistically significant* correlation between the variables, we use a **chi-square test of independence of variables in a contingency table**.
8. First, we need to get binary variables (here, whether the number of points won or the number of aces is greater than their medians). For example, the value corresponding to the number of aces is True if the player is doing more aces than usual in a match, and False otherwise.


``` python
df_bis['result'] = df_bis['points'] > df_bis['points'].median()
df_bis['manyaces'] = df_bis['aces'] > df_bis['aces'].median()
```


9. Then, we create a **contingency table**, with the frequencies of all four possibilities (True & True, True & False, etc.).


``` python
pd.crosstab(df_bis['result'], df_bis['manyaces'])
```


10. Finally, we compute the chi-square test statistic and the associated p-value. The null hypothesis is the independence between the variables. SciPy implements this test in `scipy.stats.chi2_contingency`, which returns several objects. We're interested in the second result, which is the p-value.


``` python
st.chi2_contingency(_)
```


The p-value is much lower than 0.05, so we reject the null hypothesis and conclude that there is a statistically significant correlation between the proportion of aces and the proportion of points won in a match (for Roger Federer!).

As always, correlation does not imply causation... Here, it is likely that external factors influence both variables. (http://en.wikipedia.org/wiki/Correlation_does_not_imply_causation)

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 7.5. Fitting a probability distribution to data with the maximum likelihood method

You need the statsmodels package to retrieve the test dataset. (http://statsmodels.sourceforge.net)

1. Statsmodels is a Python package for conducting statistical data analyses. It also contains real-world datasets that one can use when experimenting new methods. Here, we load the *heart* dataset.


``` python
import numpy as np
import scipy.stats as st
import statsmodels.datasets
import matplotlib.pyplot as plt
%matplotlib inline
```



``` python
data = statsmodels.datasets.heart.load_pandas().data
```


2. Let's take a look at this DataFrame.


``` python
data.tail()
```


This dataset contains censored and uncensored data: a censor of 0 means that the patient was alive at the end of the study, so that we don't know the exact survival. We only know that the patient survived *at least* the indicated number of days. For simplicity here, we only keep uncensored data (we thereby create a bias toward patients that did not survive very long after their transplant...).


``` python
data = data[data.censors==1]
survival = data.survival
```


3. Let's take a look at the data graphically, by plotting the raw survival data and the histogram.


``` python
plt.figure(figsize=(10,4));
plt.subplot(121);
plt.plot(sorted(survival)[::-1], 'o');
plt.xlabel('Patient');
plt.ylabel('Survival time (days)');
plt.subplot(122);
plt.hist(survival, bins=15);
plt.xlabel('Survival time (days)');
plt.ylabel('Number of patients');
```


4. We observe that the histogram is decreasing very rapidly. Fortunately, the survival rates are today much higher (~70% after 5 years). Let's try to fit an [exponential distribution](http://en.wikipedia.org/wiki/Exponential_distribution) to the data. According to this model, $S$ (number of days of survival) is an exponential random variable with parameter $\lambda$, and the observations $s_i$ are sampled from this distribution. Let:

$$\overline s = \frac 1 n \sum s_i$$ 

be the sample mean. The likelihood function of an exponential distribution is, by definition (see proof in the next section):

$$\mathcal L(\lambda, \{s_i\}) = P(\{s_i\} | \lambda) = \lambda^n \exp\left(-\lambda n \overline s\right)$$

The **maximum likelihood estimate** for the rate parameter is, by definition, the $\lambda$ that maximizes the likelihood function. In other words, it is the parameter that maximizes the probability of observing the data, assuming that the observations are sampled from an exponential distribution.

Here, it can be shown that the likelihood function has a maximum when $\lambda = 1/\overline s$, which is the *maximum likelihood estimate for the rate parameter*. Let's compute this parameter numerically.


``` python
smean = survival.mean()
rate = 1./smean
```


5. To compare the fitted exponential distribution to the data, we first need to generate linearly spaced values for the x axis (days).


``` python
smax = survival.max()
days = np.linspace(0., smax, 1000)
dt = smax / 999.  # bin size: interval between two
                  # consecutive values in `days`
```


We can obtain the probability density function of the exponential distribution with SciPy. The parameter is the scale, the inverse of the estimated rate.


``` python
dist_exp = st.expon.pdf(days, scale=1./rate)
```


6. Now, let's plot the histogram and the obtained distribution. We need to rescale the theoretical distribution to the histogram (depending on the bin size and the total number of data points).


``` python
nbins = 30
plt.figure(figsize=(5,3));
plt.hist(survival, nbins);
plt.plot(days, dist_exp*len(survival)*smax/nbins,
         '-r', lw=3);
```


The fit is far from perfect... We were able to find an analytical formula for the maximum likelihood estimate here. In more complex situations, this is not always possible, so that one needs to resort to numerical methods. SciPy actually integrates numerical maximum likelihood routines for a large number of distributions. Here, we use this other method to estimate the parameter of the exponential distribution.


``` python
dist = st.expon
args = dist.fit(survival); args
```


7. We can use these parameters to perform a **Kolmogorov-Smirnov test**, which assesses the goodness of fit of the distribution with respect to the data. This test is based on a distance between the **empirical distribution function** of the data and the **cumulative distribution function** (CDF) of the reference distribution.


``` python
st.kstest(survival, dist.cdf, args)
```


The second output value is the p-value. Here, it is very low: the null hypothesis (stating that the observed data stems from an exponential distribution with a maximum likelihood rate parameter) can be rejected with high confidence. Let's try another distribution, the *Birnbaum-Sanders distribution*, which is typically used to model failure times. (http://en.wikipedia.org/wiki/Birnbaum-Saunders_distribution)


``` python
dist = st.fatiguelife
args = dist.fit(survival)
st.kstest(survival, dist.cdf, args)
```


This time, the p-value is 0.07, so that we would not reject the null hypothesis with a 5% confidence level. When plotting the resulting distribution, we observe a better fit than with the exponential distribution.


``` python
dist_fl = dist.pdf(days, *args)
nbins = 30
plt.figure(figsize=(5,3));
plt.hist(survival, nbins);
plt.plot(days, dist_exp*len(survival)*smax/nbins,
         '-r', lw=3, label='exp');
plt.plot(days, dist_fl*len(survival)*smax/nbins,
         '--g', lw=3, label='BS');
plt.xlabel("Survival time (days)");
plt.ylabel("Number of patients");
plt.legend();
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 7.6. Estimating a probability distribution nonparametrically with a Kernel Density Estimation

You need to download the *Storms* dataset on the book's website, and extract it in the current directory. (http://ipython-books.github.io)

You also need matplotlib's toolkit *basemap*. (http://matplotlib.org/basemap/)

1. Let's import the usual packages. The kernel density estimation with a Gaussian kernel is implemented in *SciPy.stats*.


``` python
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
%matplotlib inline
```


2. Let's open the data with Pandas.


``` python
# http://www.ncdc.noaa.gov/ibtracs/index.php?name=wmo-data
df = pd.read_csv("data/Allstorms.ibtracs_wmo.v03r05.csv")
```


3. The dataset contains information about most storms since 1848. A single storm may appear multiple times across several consecutive days.


``` python
df[df.columns[[0,1,3,8,9]]].head()
```


4. We use Pandas' `groupby` function to obtain the average location of every storm.


``` python
dfs = df.groupby('Serial_Num')
pos = dfs[['Latitude', 'Longitude']].mean()
y, x = pos.values.T
pos.head()
```


5. We display the storms on a map with basemap. This toolkit allows us to easily project the geographical coordinates on the map.


``` python
m = Basemap(projection='mill', llcrnrlat=-65 ,urcrnrlat=85,
            llcrnrlon=-180, urcrnrlon=180)
x0, y0 = m(-180, -65)
x1, y1 = m(180, 85)
plt.figure(figsize=(10,6))
m.drawcoastlines()
m.fillcontinents(color='#dbc8b2')
xm, ym = m(x, y)
m.plot(xm, ym, '.r', alpha=.1);
```


6. To perform the Kernel Density Estimation, we need to stack the x and y coordinates of the storms into a 2xN array.


``` python
h = np.vstack((xm, ym))
```



``` python
kde = st.gaussian_kde(h)
```


7. The `gaussian_kde` routine returned a Python function. To see the results on a map, we need to evaluate this function on a 2D grid spanning the entire map. We create this grid with `meshgrid`, and we pass the x, y values to the `kde` function. We need to arrange the shape of the array since `kde` accepts a 2xN array as input.


``` python
k = 50
tx, ty = np.meshgrid(np.linspace(x0, x1, 2*k),
                     np.linspace(y0, y1, k))
v = kde(np.vstack((tx.ravel(), ty.ravel()))).reshape((k, 2*k))
```


8. Finally, we display the estimated density with `imshow`.


``` python
plt.figure(figsize=(10,6))
m.drawcoastlines()
m.fillcontinents(color='#dbc8b2')
xm, ym = m(x, y)
m.imshow(v, origin='lower', extent=[x0,x1,y0,y1],
         cmap=plt.get_cmap('Reds'));
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 7.7. Fitting a Bayesian model by sampling from a posterior distribution with a Markov Chain Monte Carlo method

You can find the instructions to install PyMC on the package's website. (http://pymc-devs.github.io/pymc/)

You also need the *Storms* dataset from the book's website. (http://ipython-books.github.io)

1. Let's import the standard packages and PyMC.


``` python
import numpy as np
import pandas as pd
import pymc
import matplotlib.pyplot as plt
%matplotlib inline
```


2. Let's import the data with Pandas.


``` python
# http://www.ncdc.noaa.gov/ibtracs/index.php?name=wmo-data
df = pd.read_csv("data/Allstorms.ibtracs_wmo.v03r05.csv",
                 delim_whitespace=False)
```


3. With Pandas, it only takes a single line of code to get the annual number of storms in the North Atlantic Ocean. We first select the storms in that basin (`NA`), then we group the rows by year (`Season`), and we take the number of unique storm (`Serial_Num`) as each storm can span several days (`nunique` method).


``` python
cnt = df[df['Basin'] == ' NA'].groupby('Season') \
      ['Serial_Num'].nunique()
years = cnt.index
y0, y1 = years[0], years[-1]
arr = cnt.values
plt.figure(figsize=(8,4));
plt.plot(years, arr, '-ok');
plt.xlim(y0, y1);
plt.xlabel("Year");
plt.ylabel("Number of storms");
```


4. Now, we define our probabilistic model. We assume that storms arise following a time-dependent Poisson process with a deterministic rate. We assume this rate is a piecewise-constant function that takes a first value `early_mean` before a certain switch point, and a second value `late_mean` after that point. These three unknown parameters are treated as random variables (we will describe them more in the *How it works...* section).

A [Poisson process](http://en.wikipedia.org/wiki/Poisson_process) is a particular **point process**, that is, a stochastic process describing the random occurence of instantaneous events. The Poisson process is fully random: the events occur independently at a given rate.


``` python
switchpoint = pymc.DiscreteUniform('switchpoint',
                                   lower=0, upper=len(arr))
early_mean = pymc.Exponential('early_mean', beta=1)
late_mean = pymc.Exponential('late_mean', beta=1)
```


5. We define the piecewise-constant rate as a Python function.


``` python
@pymc.deterministic(plot=False)
def rate(s=switchpoint, e=early_mean, l=late_mean):
    out = np.empty(len(arr))
    out[:s] = e
    out[s:] = l
    return out
```


6. Finally, the observed variable is the annual number of storms. It follows a Poisson variable with a random mean (the rate of the underlying Poisson process). This fact is a known mathematical property of Poisson processes.


``` python
storms = pymc.Poisson('storms', mu=rate, value=arr, observed=True)
```


7. Now, we use the MCMC method to sample from the posterior distribution, given the observed data. The `sample` method launches the fitting iterative procedure.


``` python
model = pymc.Model([switchpoint, early_mean, late_mean, rate, storms])
```



``` python
mcmc = pymc.MCMC(model)
mcmc.sample(iter=10000, burn=1000, thin=10)
```


8. Let's plot the sampled Markov chains. Their stationary distribution corresponds to the posterior distribution we want to characterize.


``` python
plt.figure(figsize=(8,8))
plt.subplot(311);
plt.plot(mcmc.trace('switchpoint')[:]);
plt.ylabel("Switch point"); 
plt.subplot(312);
plt.plot(mcmc.trace('early_mean')[:]);
plt.ylabel("Early mean");
plt.subplot(313);
plt.plot(mcmc.trace('late_mean')[:]);
plt.xlabel("Iteration");
plt.ylabel("Late mean");
```


9. We also plot the distribution of the samples: they correspond to the posterior distributions of our parameters, after the data points have been taken into account.


``` python
plt.figure(figsize=(14,3))
plt.subplot(131);
plt.hist(mcmc.trace('switchpoint')[:] + y0, 15);
plt.xlabel("Switch point")
plt.ylabel("Distribution")
plt.subplot(132);
plt.hist(mcmc.trace('early_mean')[:], 15);
plt.xlabel("Early mean");
plt.subplot(133);
plt.hist(mcmc.trace('late_mean')[:], 15);
plt.xlabel("Late mean");
```


10. Taking the sample mean of these distributions, we get posterior estimates for the three unknown parameters, including the year where the frequency of storms suddenly increased.


``` python
yp = y0 + mcmc.trace('switchpoint')[:].mean()
em = mcmc.trace('early_mean')[:].mean()
lm = mcmc.trace('late_mean')[:].mean()
print((yp, em, lm))
```


11. Now we can plot the estimated rate on top of the observations.


``` python
plt.figure(figsize=(8,4));
plt.plot(years, arr, '-ok');
plt.axvline(yp, color='k', ls='--');
plt.plot([y0, yp], [em, em], '-b', lw=3);
plt.plot([yp, y1], [lm, lm], '-r', lw=3);
plt.xlim(y0, y1);
plt.xlabel("Year");
plt.ylabel("Number of storms");
```


For a possible scientific interpretation of the data considered here, see http://www.gfdl.noaa.gov/global-warming-and-hurricanes.

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 7.8. Analyzing data with R in the IPython notebook

**UPDATE (2014-09-29)**: in newer versions of rpy2, the IPython extension with the R magic is `rpy2.ipython` and not `rmagic` as stated in the book.

There are three steps to use R from IPython. First, install R and rpy2 (R to Python interface). Of course, you only need to do this step once. Then, to use R in an IPython session, you need to load the IPython R extension.

1. Download and install R for your operating system. (http://cran.r-project.org/mirrors.html)
2. Download and install [rpy2](http://rpy.sourceforge.net/rpy2.html). Windows users can try to download an *experimental* installer on Chris Gohlke's webpage. (http://www.lfd.uci.edu/~gohlke/pythonlibs/#rpy2)
3. Then, to be able to execute R code in an IPython notebook, execute `%load_ext rpy2.ipython` first.

rpy2 does not appear to work well on Windows. We recommend using Linux or Mac OS X.

To install R and rpy2 on Ubuntu, run the following commands:

    sudo apt-get install r-base-dev
    sudo apt-get install python-rpy2

Here, we will use the following workflow. First, we load data from Python. Then, we use R to design and fit a model, and to make some plots in the IPython notebook. We could also load data from R, or design and fit a statistical model with Python's statsmodels package, etc. In particular, the analysis we do here could be done entirely in Python, without resorting to the R language. This recipe just shows the basics of R and illustrates how R and Python can play together within an IPython session.

1. Let's load the *longley* dataset with the statsmodels package. This dataset contains a few economic indicators in the US from 1947 to 1962. We also load the IPython R extension.


``` python
import statsmodels.datasets as sd
```



``` python
data = sd.longley.load_pandas()
```



``` python
%load_ext rpy2.ipython
```


2. We define `x` and `y` as the exogeneous (independent) and endogenous (dependent) variables, respectively. The endogenous variable quantifies the total employment in the country.


``` python
data.endog_name, data.exog_name
```



``` python
y, x = data.endog, data.exog
```


3. For convenience, we add the endogenous variable to the `x` DataFrame.


``` python
x['TOTEMP'] = y
```



``` python
x
```


4. We will make a simple plot in R. First, we need to pass Python variables to R. We can use the `%R -i var1,var2` magic. Then, we can call R's `plot` command.


``` python
gnp = x['GNP']
totemp = x['TOTEMP']
```



``` python
%R
```



``` python
%R -i totemp,gnp plot(gnp, totemp)
```


5. Now that the data has been passed to R, we can fit a linear model to the data. The `lm` function lets us perform a linear regression. Here, we want to express `totemp` (total employement) as a function of the country's GNP.


``` python
%%R
fit <- lm(totemp ~ gnp);  # Least-squares regression
print(fit$coefficients)  # Display the coefficients of the fit.
plot(gnp, totemp)  # Plot the data points.
abline(fit)  # And plot the linear regression.
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).

