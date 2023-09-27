[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Yu-Group/iterative-Random-Forest/master)

# iterative Random Forest
The algorithm details are available at: 

Sumanta Basu, Karl Kumbier, James B. Brown, Bin Yu,  Iterative Random Forests to detect predictive and stable high-order interactions, PNAS
<https://www.pnas.org/content/115/8/1943>

The implementation is a joint effort of several people in UC Berkeley. See the [Authors.md](Authors.md) for the complete list.
The weighted random forest implementation is based on the random forest source code and API design from [scikit-learn](http://scikit-learn.org/stable/index.html), details can be found in *API design for machine learning software: experiences from the scikit-learn project, Buitinck et al., 2013.*. The setup file is based on the setup file from [skgarden](https://github.com/scikit-garden/scikit-garden/tree/master/skgarden). 

## Installation
To install, simply run `pip install irf`. If you run into any issues, see [installation help](installation.md).

## A simple demo
In order to use irf, you need to import it in python.

```python
import numpy as np
from irf import irf_utils
from irf.ensemble import RandomForestClassifierWithWeights
```
Generate a simple data set with 2 features: 1st feature is a noise feature that has no power in predicting the labels, the 2nd feature determines the label perfectly:
```python
n_samples = 1000
n_features = 10
X_train = np.random.uniform(low=0, high=1, size=(n_samples, n_features))
y_train = np.random.choice([0, 1], size=(n_samples,), p=[.5, .5])
X_test = np.random.uniform(low=0, high=1, size=(n_samples, n_features))
y_test = np.random.choice([0, 1], size=(n_samples,), p=[.5, .5])
# The second feature (which is indexed by 1) is very important
X_train[:, 1] = X_train[:, 1] + y_train
X_test[:, 1] = X_test[:, 1] + y_test
```
Then run irf
```
all_rf_weights, all_K_iter_rf_data, \
    all_rf_bootstrap_output, all_rit_bootstrap_output, \
    stability_score = irf_utils.run_iRF(X_train=X_train,
                                        X_test=X_test,
                                        y_train=y_train,
                                        y_test=y_test,
                                        K=5,                          # number of iteration
                                        rf = RandomForestClassifierWithWeights(n_estimators=20),
                                        B=30,
                                        random_state_classifier=2018, # random seed
                                        propn_n_samples=.2,
                                        bin_class_type=1,
                                        M=20,
                                        max_depth=5,
                                        noisy_split=False,
                                        num_splits=2,
                                        n_estimators_bootstrap=5)
```
all_rf_weights stores all the weights for each iteration:
```python
print(all_rf_weights['rf_weight5'])
```
The proposed feature combination and their scores:
```python
print(stability_score)
```


