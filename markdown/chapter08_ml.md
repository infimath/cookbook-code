# ml


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 8.1. Getting started with scikit-learn

We will generate one-dimensional data with a simple model (including some noise), and we will try to fit a function to this data. With this function, we can predict values on new data points. This is a curve-fitting regression problem.

1. First, let's make all the necessary imports.


``` python
import numpy as np
import scipy.stats as st
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
%matplotlib inline

```


2. We now define the deterministic function underlying our generative model.


``` python
f = lambda x: np.exp(3 * x)
```


3. We generate the values along the curve on $[0, 2]$.


``` python
x_tr = np.linspace(0., 2, 200)
y_tr = f(x_tr)
```


4. Now, let's generate our data points within $[0, 1]$. We use the function $f$ and we add some Gaussian noise.


``` python
x = np.array([0, .1, .2, .5, .8, .9, 1])
y = f(x) + np.random.randn(len(x))
```


5. Let's plot our data points on $[0, 1]$.


``` python
plt.figure(figsize=(6,3));
plt.plot(x_tr[:100], y_tr[:100], '--k');
plt.plot(x, y, 'ok', ms=10);
```


6. Now, we use scikit-learn to fit a linear model to the data. There are three steps. First, we create the model (an instance of the `LinearRegression` class). Then we fit the model to our data. Finally, we predict values from our trained model.


``` python
# We create the model.
lr = lm.LinearRegression()
# We train the model on our training dataset.
lr.fit(x[:, np.newaxis], y);
# Now, we predict points with our trained model.
y_lr = lr.predict(x_tr[:, np.newaxis])
```


We need to convert `x` and `x_tr` to column vectors, as it is a general convention in scikit-learn that observations are rows, while features are columns. Here, we have 7 observations with 1 feature.

7. We now plot the result of the trained linear model. We obtain a regression line, in green here.


``` python
plt.figure(figsize=(6,3));
plt.plot(x_tr, y_tr, '--k');
plt.plot(x_tr, y_lr, 'g');
plt.plot(x, y, 'ok', ms=10);
plt.xlim(0, 1);
plt.ylim(y.min()-1, y.max()+1);
plt.title("Linear regression");
```


8. The linear fit is not well adapted here, since the data points are generated according to a non-linear model (an exponential curve). Therefore, we are now going to fit a non-linear model. More precisely, we will fit a polynomial function to our data points. We can still use linear regression for that, by pre-computing the exponents of our data points. This is done by generating a Vandermonde matrix, using the `np.vander` function. We will explain this trick in more detail in *How it works...*.


``` python
lrp = lm.LinearRegression()
plt.figure(figsize=(6,3));
plt.plot(x_tr, y_tr, '--k');

for deg, s in zip([2, 5], ['-', '.']):
    lrp.fit(np.vander(x, deg + 1), y);
    y_lrp = lrp.predict(np.vander(x_tr, deg + 1))
    plt.plot(x_tr, y_lrp, s, label='degree ' + str(deg));
    plt.legend(loc=2);
    plt.xlim(0, 1.4);
    plt.ylim(-10, 40);
    # Print the model's coefficients.
    print(' '.join(['%.2f' % c for c in lrp.coef_]))
plt.plot(x, y, 'ok', ms=10);
plt.title("Linear regression");
```


We have fitted two polynomial models of degree 2 and 5. The degree 2 polynomial fits the data points less precisely than the degree 5 polynomial. However, it seems more robust: the degree 5 polynomial seems really bad at predicting values outside the data points (look for example at the portion $x \geq 1$). This is what we call **overfitting**: by using a model too complex, we obtain a better fit on the trained dataset, but a less robust model outside this set.

9. We will now use a different learning model, called **ridge regression**. It works like linear regression, except that it prevents the polynomial's coefficients to explode (which was what happened in the overfitting example above). By adding a **regularization term** in the **loss function**, ridge regression imposes some structure on the underlying model. We will see more details in the next section.

The ridge regression model has a meta-parameter which represents the weight of the regularization term. We could try different values with trials and errors, using the `Ridge` class. However, scikit-learn includes another model called `RidgeCV` which includes a parameter search with cross-validation. In practice, it means that you don't have to tweak this parameter by hand: scikit-learn does it for you. Since the models of scikit-learn always follow the `fit`-`predict` API, all we have to do is replace `lm.LinearRegression` by `lm.RidgeCV` in the code above. We will give more details in the next section.


``` python
ridge = lm.RidgeCV()
plt.figure(figsize=(6,3));
plt.plot(x_tr, y_tr, '--k');

for deg, s in zip([2, 5], ['-', '.']):
    ridge.fit(np.vander(x, deg + 1), y);
    y_ridge = ridge.predict(np.vander(x_tr, deg + 1))
    plt.plot(x_tr, y_ridge, s, label='degree ' + str(deg));
    plt.legend(loc=2);
    plt.xlim(0, 1.5);
    plt.ylim(-5, 80);
    # Print the model's coefficients.
    print(' '.join(['%.2f' % c for c in ridge.coef_]))

plt.plot(x, y, 'ok', ms=10);
plt.title("Ridge regression");
```


This time, the degree 5 polynomial seems better than the simpler degree 2 polynomial (which now causes **underfitting**). The ridge regression reduces the overfitting issue here.

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 8.2. Predicting who will survive on the Titanic with logistic regression

This recipe is based on a [Kaggle competition](http://www.kaggle.com/c/titanic-gettingStarted) where the goal is to predict survival on the Titanic, based on real data. [Kaggle](http://www.kaggle.com/competitions) hosts machine learning competitions where anyone can download a dataset, train a model, and test the predictions on the website. The author of the best model wins a price. It is a fun way to get started with machine learning.

Here, we use this example to introduce logistic regression, a basic classifier. We also show how to perform a grid search with cross-validation.

You need to download the Titanic dataset on the book's website (https://ipython-books.github.io).

1. We import the standard libraries.


``` python
import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model as lm
import sklearn.cross_validation as cv
import sklearn.grid_search as gs
import matplotlib.pyplot as plt
%matplotlib inline

```


2. We load the train and test datasets with Pandas.


``` python
train = pd.read_csv('data/titanic_train.csv')
test = pd.read_csv('data/titanic_test.csv')
```



``` python
train[train.columns[[2,4,5,1]]].head()
```


3. Let's keep only a few fields for this example. We also convert the `sex` field to a binary variable, so that it can be handled correctly by NumPy and scikit-learn. Finally, we remove the rows containing `NaN` values.


``` python
data = train[['Sex', 'Age', 'Pclass', 'Survived']].copy()
data['Sex'] = data['Sex'] == 'female'
data = data.dropna()
```


4. Now, we convert this `DataFrame` to a NumPy array, so that we can pass it to scikit-learn.


``` python
data_np = data.astype(np.int32).values
X = data_np[:,:-1]
y = data_np[:,-1]
```


5. Let's have a look at the survival of male and female passengers, as a function of their age.


``` python
# We define a few boolean vectors.
female = X[:,0] == 1
survived = y == 1
# This vector contains the age of the passengers.
age = X[:,1]
# We compute a few histograms.
bins_ = np.arange(0, 81, 5)
S = {'male': np.histogram(age[survived & ~female], 
                          bins=bins_)[0],
     'female': np.histogram(age[survived & female], 
                            bins=bins_)[0]}
D = {'male': np.histogram(age[~survived & ~female], 
                          bins=bins_)[0],
     'female': np.histogram(age[~survived & female], 
                            bins=bins_)[0]}
```



``` python
# We now plot the data.
bins = bins_[:-1]
plt.figure(figsize=(10,3));
for i, sex, color in zip((0, 1),
                         ('male', 'female'),
                         ('#3345d0', '#cc3dc0')):
    plt.subplot(121 + i);
    plt.bar(bins, S[sex], bottom=D[sex], color=color,
            width=5, label='survived');
    plt.bar(bins, D[sex], color='k', width=5, label='died');
    plt.xlim(0, 80);
    plt.grid(None);
    plt.title(sex + " survival");
    plt.xlabel("Age (years)");
    plt.legend();
```


6. Let's try to train a `LogisticRegression` classifier. We first need to create a train and a test dataset.


``` python
# We split X and y into train and test datasets.
(X_train, X_test, 
 y_train, y_test) = cv.train_test_split(X, y, test_size=.05)
```



``` python
# We instanciate the classifier.
logreg = lm.LogisticRegression();
```


7. Let's train the model and get the predicted values on the test set.


``` python
logreg.fit(X_train, y_train)
y_predicted = logreg.predict(X_test)
```


The following figure shows the actual and predicted results.


``` python
plt.figure(figsize=(8, 3));
plt.imshow(np.vstack((y_test, y_predicted)),
           interpolation='none', cmap='bone');
plt.xticks([]); plt.yticks([]);
plt.title(("Actual and predicted survival outcomes"
          " on the test set"));
```


8. To get an estimation of the performance of the model, we can use the `cross_val_score` that computes the cross-validation score. This function uses by default a 3-fold stratified cross-validation procedure, but this can be changed with the `cv` keyword argument.


``` python
cv.cross_val_score(logreg, X, y)
```


This function returns, for each pair of train and test set, a prediction score.

9. The `LogisticRegression` class accepts a `C` hyperparameter as argument. This parameter quantifies the regularization strength. To find a good value, we can perform a grid search with the `GridSearchCV` class. It takes as input an estimator, and a dictionary of parameter values. This new estimator uses cross-validation to select the best parameter.


``` python
grid = gs.GridSearchCV(logreg, {'C': np.logspace(-5, 5, 200)}, n_jobs=4)
grid.fit(X_train, y_train);
grid.best_params_
```


Here is the performance of the best estimator.


``` python
cv.cross_val_score(grid.best_estimator_, X, y)
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 8.3. Learning to recognize handwritten digits with a K-nearest neighbors classifier

1. Let's do the traditional imports.


``` python
import numpy as np
import sklearn
import sklearn.datasets as ds
import sklearn.cross_validation as cv
import sklearn.neighbors as nb
import matplotlib.pyplot as plt
%matplotlib inline

```


2. Let's load the digits dataset, part of the `datasets` module of scikit-learn. This dataset contains hand-written digits that have been manually labeled.


``` python
digits = ds.load_digits()
X = digits.data
y = digits.target
print((X.min(), X.max()))
print(X.shape)
```


In the matrix `X`, each row contains the $8 \times 8=64$ pixels (in grayscale, values between 0 and 16). The pixels are ordered according to the row-major order.

3. Let's display some of the images.


``` python
nrows, ncols = 2, 5
plt.figure(figsize=(6,3));
plt.gray()
for i in range(ncols * nrows):
    ax = plt.subplot(nrows, ncols, i + 1)
    ax.matshow(digits.images[i,...])
    plt.xticks([]); plt.yticks([]);
    plt.title(digits.target[i]);
```


4. Now, let's fit a K-nearest neighbors classifier on the data.


``` python
(X_train, X_test, 
 y_train, y_test) = cv.train_test_split(X, y, test_size=.25)
```



``` python
knc = nb.KNeighborsClassifier()
```



``` python
knc.fit(X_train, y_train);
```


5. Let's evaluate the score of the trained classifier on the test dataset.


``` python
knc.score(X_test, y_test)
```


6. Now, let's see if our classifier can recognize a "hand-written" digit!


``` python
# Let's draw a 1.
one = np.zeros((8, 8))
one[1:-1, 4] = 16  # The image values are in [0, 16].
one[2, 3] = 16
```



``` python
plt.figure(figsize=(2,2));
plt.imshow(one, interpolation='none');
plt.grid(False);
plt.xticks(); plt.yticks();
plt.title("One");
```



``` python
knc.predict(one.ravel())
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 8.4. Learning from text: Naive Bayes for Natural Language Processing

In this recipe, we show how to handle text data with scikit-learn. Working with text requires careful preprocessing and feature extraction. It is also quite common to deal with highly sparse matrices.

We will learn to recognize whether a comment posted during a public discussion is considered insulting to one of the participants. We will use a labeled dataset from [Impermium](https://impermium.com), released during a [Kaggle competition](https://www.kaggle.com/c/detecting-insults-in-social-commentary).

You need to download the *troll* dataset on the book's website. (https://ipython-books.github.io)

1. Let's import our libraries.


``` python
import numpy as np
import pandas as pd
import sklearn
import sklearn.cross_validation as cv
import sklearn.grid_search as gs
import sklearn.feature_extraction.text as text
import sklearn.naive_bayes as nb
import matplotlib.pyplot as plt
%matplotlib inline

```


2. Let's open the csv file with Pandas.


``` python
df = pd.read_csv("data/troll.csv")
```


3. Each row is a comment. There are three columns: whether the comment is insulting (1) or not (0), the data, and the unicode-encoded contents of the comment.


``` python
df[['Insult', 'Comment']].tail()
```


4. Now, we are going to define the feature matrix $\mathbf{X}$ and the labels $\mathbf{y}$.


``` python
y = df['Insult']
```


Obtaining the feature matrix from the text is not trivial. Scikit-learn can only work with numerical matrices. How to convert text into a matrix of numbers? A classical solution is to first extract a **vocabulary**: a list of words used throughout the corpus. Then, we can count, for each sample, the frequency of each word. We end up with a **sparse matrix**: a huge matrix containiny mostly zeros. Here, we do this in two lines. We will give more explanations in *How it works...*.


``` python
tf = text.TfidfVectorizer()
X = tf.fit_transform(df['Comment'])
print(X.shape)
```


5. There are 3947 comments and 16469 different words. Let's estimate the sparsity of this feature matrix.


``` python
print("Each sample has ~{0:.2f}% non-zero features.".format(
          100 * X.nnz / float(X.shape[0] * X.shape[1])))
```


6. Now, we are going to train a classifier as usual. We first split the data into a train and test set.


``` python
(X_train, X_test,
 y_train, y_test) = cv.train_test_split(X, y,
                                        test_size=.2)
```


7. We use a **Bernoulli Naive Bayes classifier** with a grid search on the parameter $\alpha$.


``` python
bnb = gs.GridSearchCV(nb.BernoulliNB(), param_grid={'alpha':np.logspace(-2., 2., 50)})
bnb.fit(X_train, y_train);
```


8. What is the performance of this classifier on the test dataset?


``` python
bnb.score(X_test, y_test)
```


9. Let's take a look at the words corresponding to the largest coefficients (the words we find frequently in insulting comments).


``` python
# We first get the words corresponding to each feature.
names = np.asarray(tf.get_feature_names())
# Next, we display the 50 words with the largest
# coefficients.
print(','.join(names[np.argsort(
    bnb.best_estimator_.coef_[0,:])[::-1][:50]]))
```


10. Finally, let's test our estimator on a few test sentences.


``` python
print(bnb.predict(tf.transform([
    "I totally agree with you.",
    "You are so stupid.",
    "I love you."
    ])))
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 8.5. Using Support Vector Machines for classification tasks

1. Let's do the traditional imports.


``` python
import numpy as np
import pandas as pd
import sklearn
import sklearn.datasets as ds
import sklearn.cross_validation as cv
import sklearn.grid_search as gs
import sklearn.svm as svm
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline

```


2. We generate 2D points and assign a binary label according to a linear operation on the coordinates.


``` python
X = np.random.randn(200, 2)
y = X[:, 0] + X[:, 1] > 1
```


3. We now fit a linear **Support Vector Classifier** (SVC). This classifier tries to separate the two groups of points with a linear boundary (a line here, more generally a hyperplane).


``` python
# We train the classifier.
est = svm.LinearSVC()
est.fit(X, y);
```


4. We define a function that displays the boundaries and decision function of a trained classifier.


``` python
# We generate a grid in the square [-3,3 ]^2.
xx, yy = np.meshgrid(np.linspace(-3, 3, 500),
                     np.linspace(-3, 3, 500))
# This function takes a SVM estimator as input.
def plot_decision_function(est):
    # We evaluate the decision function on the grid.
    Z = est.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cmap = plt.cm.Blues
    # We display the decision function on the grid.
    plt.figure(figsize=(5,5));
    plt.imshow(Z,
                extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                aspect='auto', origin='lower', cmap=cmap);
    # We display the boundaries.
    plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                colors='k');
    # We display the points with their true labels.
    plt.scatter(X[:, 0], X[:, 1], s=30, c=.5+.5*y, lw=1, 
                cmap=cmap, vmin=0, vmax=1);
    plt.axhline(0, color='k', ls='--');
    plt.axvline(0, color='k', ls='--');
    plt.xticks(());
    plt.yticks(());
    plt.axis([-3, 3, -3, 3]);
```


5. Let's take a look at the classification results with the linear SVC.


``` python
plot_decision_function(est);
plt.title("Linearly separable, linear SVC");
```


The linear SVC tried to separate the points with a line and it did a pretty good job.

6. We now modify the labels with a *XOR* function. A point's label is 1 if the coordinates have different signs. This classification is not linearly separable. Therefore, a linear SVC fails completely.


``` python
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
# We train the classifier.
est = gs.GridSearchCV(svm.LinearSVC(), 
                      {'C': np.logspace(-3., 3., 10)});
est.fit(X, y);
print("Score: {0:.1f}".format(
      cv.cross_val_score(est, X, y).mean()))
# Plot the decision function.
plot_decision_function(est);
plt.title("XOR, linear SVC");
```


7. Fortunately, it is possible to use non-linear SVCs by using non-linear **kernels**. Kernels specify a non-linear transformation of the points into a higher-dimensional space. Transformed points in this space are assumed to be more linearly separable, although they are not necessarily in the original space. By default, the `SVC` classifier in scikit-learn uses the **Radial Basis Function** (RBF) kernel.


``` python
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
est = gs.GridSearchCV(svm.SVC(), 
                      {'C': np.logspace(-3., 3., 10),
                    'gamma': np.logspace(-3., 3., 10)});
est.fit(X, y);
print("Score: {0:.3f}".format(
      cv.cross_val_score(est, X, y).mean()))
plot_decision_function(est.best_estimator_);
plt.title("XOR, non-linear SVC");
```


This time, the non-linear SVC does a pretty good job at classifying these non-linearly separable points.

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.

# 8.6. Using a random forest to select important features for regression

**Decisions trees** are frequently used to represent workflows or algorithms. They also form a method for non-parametric supervised learning. A tree mapping observations to target values is learnt on a training set and gives the outcomes of new observations.

**Random forests** are ensembles of decision trees. Multiple decision trees are trained and aggregated to form a model that is more performant than any of the individual trees. This general idea is the purpose of **ensemble learning**.

There are many types of ensemble methods. Random forests are an instance of **bootstrap aggregating**, also called **bagging**, where models are trained on randomly drawn subsets of the training set.

Random forests yield information about the importance of each feature for the classification or regression task. In this recipe, we use this method to find the features the most influent on the price of Boston houses. We will use a classic dataset containing a range of diverse indicators about the houses' neighborhoud.

## How to do it...

1. We import the packages.


``` python
import numpy as np
import sklearn as sk
import sklearn.datasets as skd
import sklearn.ensemble as ske
import matplotlib.pyplot as plt
import matplotlib as mpl
%matplotlib inline
mpl.rcParams['figure.dpi'] = mpl.rcParams['savefig.dpi'] = 300
```


2. We load the Boston dataset.


``` python
data = skd.load_boston()
```


The details of this dataset can be found in `data['DESCR']`. Here is the description of all features:

- *CRIM*,     per capita crime rate by town
- *ZN*,       proportion of residential land zoned for lots over 25,000 sq.ft.
- *INDUS*,    proportion of non-retail business acres per town
- *CHAS*,     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- *NOX*,      nitric oxides concentration (parts per 10 million)
- *RM*,       average number of rooms per dwelling
- *AGE*,      proportion of owner-occupied units built prior to 1940
- *DIS*,      weighted distances to five Boston employment centres
- *RAD*,      index of accessibility to radial highways
- *TAX*,      full-value property-tax rate per USD 10,000
- *PTRATIO*,  pupil-teacher ratio by town
- *B*,        $1000(Bk - 0.63)^2$ where Bk is the proportion of blacks by town
- *LSTAT*,    % lower status of the population
- *MEDV*,     Median value of owner-occupied homes in $1000's

The target value is `MEDV`.

3. We create a `RandomForestRegressor` model.


``` python
reg = ske.RandomForestRegressor()
```


4. We get the samples and the target values from this dataset.


``` python
X = data['data']
y = data['target']
```


5. Let's fit the model.


``` python
reg.fit(X, y);
```


6. The importance of our features can be found in `reg.feature_importances_`. We sort them by decreasing order of importance.


``` python
fet_ind = np.argsort(reg.feature_importances_)[::-1]
fet_imp = reg.feature_importances_[fet_ind]
```


7. Finally, we plot a histogram of the features importance.


``` python
fig = plt.figure(figsize=(8,4));
ax = plt.subplot(111);
plt.bar(np.arange(len(fet_imp)), fet_imp, width=1, lw=2);
plt.grid(False);
ax.set_xticks(np.arange(len(fet_imp))+.5);
ax.set_xticklabels(data['feature_names'][fet_ind]);
plt.xlim(0, len(fet_imp));
```


We find that *LSTAT* (proportion of lower status of the population) and *RM* (number of rooms per dwelling) are the most important features determining the price of a house. As an illustration, here is a scatter plot of the price as a function of *LSTAT*:


``` python
plt.scatter(X[:,-1], y);
plt.xlabel('LSTAT indicator');
plt.ylabel('Value of houses (k$)');
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.

# 8.7. Reducing the dimensionality of a data with a Principal Component Analysis

1. We import NumPy, matplotlib, and scikit-learn.


``` python
import numpy as np
import sklearn
import sklearn.decomposition as dec
import sklearn.datasets as ds
import matplotlib.pyplot as plt
%matplotlib inline

```


2. The Iris flower dataset is available in the *datasets* module of scikit-learn.


``` python
iris = ds.load_iris()
X = iris.data
y = iris.target
print(X.shape)
```


3. Each row contains four parameters related to the morphology of the flower. Let's display the first two components in two dimensions. The color reflects the iris variety of the flower (the label, between 0 and 2).


``` python
plt.figure(figsize=(6,3));
plt.scatter(X[:,0], X[:,1], c=y,
            s=30, cmap=plt.cm.rainbow);
```


4. We now apply PCA on the dataset to get the transformed matrix. This operation can be done in a single line with scikit-learn: we instantiate a `PCA` model, and call the `fit_transform` method. This function computes the principal components first, and projects the data then.


``` python
X_bis = dec.PCA().fit_transform(X)
```


5. We now display the same dataset, but in a new coordinate system (or equivalently, a linearly transformed version of the initial dataset).


``` python
plt.figure(figsize=(6,3));
plt.scatter(X_bis[:,0], X_bis[:,1], c=y,
            s=30, cmap=plt.cm.rainbow);
```


Points belonging to the same classes are now grouped together, even though the `PCA` estimator dit *not* use the labels. The PCA was able to find a projection maximizing the variance, which corresponds here to a projection where the classes are well separated.

6. The `scikit.decomposition` module contains several variants of the classic `PCA` estimator: `ProbabilisticPCA`, `SparsePCA`, `RandomizedPCA`, `KernelPCA`... As an example, let's take a look at `KernelPCA`, a non-linear version of PCA.


``` python
X_ter = dec.KernelPCA(kernel='rbf').fit_transform(X)
plt.figure(figsize=(6,3));
plt.scatter(X_ter[:,0], X_ter[:,1], c=y, s=30, cmap=plt.cm.rainbow);
```


> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).


> This is one of the 100 recipes of the [IPython Cookbook](http://ipython-books.github.io/), the definitive guide to high-performance scientific computing and data science in Python.


# 8.8. Detecting hidden structures in a dataset with clustering

1. Let's import the libraries.


``` python
from itertools import permutations
import numpy as np
import sklearn
import sklearn.decomposition as dec
import sklearn.cluster as clu
import sklearn.datasets as ds
import sklearn.grid_search as gs
import matplotlib.pyplot as plt
%matplotlib inline

```


2. Let's generate a random dataset with three clusters.


``` python
X, y = ds.make_blobs(n_samples=200, n_features=2, centers=3)
```


3. We will need a couple of functions to relabel and display the results of the clustering algorithms.


``` python
def relabel(cl):
    """Relabel a clustering with three clusters
    to match the original classes."""
    if np.max(cl) != 2:
        return cl
    perms = np.array(list(permutations((0, 1, 2))))
    i = np.argmin([np.sum(np.abs(perm[cl] - y))
                   for perm in perms])
    p = perms[i]
    return p[cl]
```



``` python
def display_clustering(labels, title):
    """Plot the data points with the cluster colors."""
    # We relabel the classes when there are 3 clusters.
    labels = relabel(labels)
    plt.figure(figsize=(8,3));
    # Display the points with the true labels on the left, 
    # and with the clustering labels on the right.
    for i, (c, title) in enumerate(zip(
            [y, labels], ["True labels", title])):
        plt.subplot(121 + i);
        plt.scatter(X[:,0], X[:,1], c=c, s=30, 
                    linewidths=0, cmap=plt.cm.rainbow);
        plt.xticks([]); plt.yticks([]);
        plt.title(title);
```


4. Now, we cluster the dataset with the **K-means** algorithm, a classic and simple clustering algorithm.


``` python
km = clu.KMeans()
km.fit(X);
display_clustering(km.labels_, "KMeans")
```


5. This algorithm requires the number of clusters at initialization time. In general, however, we do not necessarily now the number of clusters in the dataset. Here, let's try with `n_clusters=3` (that's cheating, because we happen to know that there are 3 clusters!).


``` python
km = clu.KMeans(n_clusters=3)
km.fit(X);
display_clustering(km.labels_, "KMeans(3)")
```


6. Let's try a few other clustering algorithms implemented in scikit-learn. The simplicity of the API makes it really easy to try different methods: it is just a matter of changing the name of the class.


``` python
plt.figure(figsize=(8,5));
plt.subplot(231);
plt.scatter(X[:,0], X[:,1], c=y, s=30,
            linewidths=0, cmap=plt.cm.rainbow);
plt.xticks([]); plt.yticks([]);
plt.title("True labels");
for i, est in enumerate([
        clu.SpectralClustering(3),
        clu.AgglomerativeClustering(3),
        clu.MeanShift(),
        clu.AffinityPropagation(),
        clu.DBSCAN(),
    ]):
    est.fit(X);
    c = relabel(est.labels_)
    plt.subplot(232 + i);
    plt.scatter(X[:,0], X[:,1], c=c, s=30,
                linewidths=0, cmap=plt.cm.rainbow);
    plt.xticks([]); plt.yticks([]);
    plt.title(est.__class__.__name__);
```


The first two algorithms required the number of clusters as input. The next two did not, but they were able to find the right number 3. The last two failed at finding the correct number of clusters (*overclustering*).

> You'll find all the explanations, figures, references, and much more in the book (to be released later this summer).

> [IPython Cookbook](http://ipython-books.github.io/), by [Cyrille Rossant](http://cyrille.rossant.net), Packt Publishing, 2014 (500 pages).

