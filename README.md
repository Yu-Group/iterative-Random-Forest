# iterative Random Forest
The algorithm details are available at: 

Sumanta Basu, Karl Kumbier, James B. Brown, Bin Yu,  Iterative Random Forests to detect predictive and stable high-order interactions, 
<https://arxiv.org/abs/1706.08457>

## Basic setup and installation
Before the installation, please make sure you installed the following packages correctly via pip:
```bash
pip install cython numpy scikit-learn 
```
Installing irf package is simple. Just clone this repo and use pip install.
```bash
git clone https://github.com/shifwang/irf
```

Then go to the `irf` folder and use pip install:
```bash
pip install -e .
```
If irf is installed successfully, you should be able to see it using pip list:
```bash
pip list | grep irf
```
and you should be able to run all the tests (assume the working directory is in the package iterative-Random-Forest):
```bash
python irf/tests/test_irf_utils.py
python irf/tests/test_irf_weighted.py
```
## A simple demo
In order to use irf, you need to import it in python.

```python
import numpy as np
from irf import irf_utils
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
                                        n_estimators=20,              # number of trees in the forest
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
Contributors:

Shamindra Shrotriya <https://github.com/shamindras>

Runjing(Bryan) Liu <runjing_liu@berkeley.edu>

Stéfan van der Walt <stefan@mentat.za.net>

Chris Holdgraf <choldgraf@berkeley.edu>

Karl Kumbier <kkumbier@berkeley.edu>

Yu(Hue) Wang <wang.yu@berkeley.edu>
