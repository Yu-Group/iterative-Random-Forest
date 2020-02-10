#!/usr/bin/python
from irf import irf_utils
import numpy as np
from irf.ensemble import RandomForestClassifierWithWeights


def test_iRF_weight1():
    # Check when label is random, whether the feature importance of every
    # feature is the same.
    n_samples = 1000
    n_features = 10
    random_state_classifier = 2018
    np.random.seed(random_state_classifier)
    X_train = np.random.uniform(low=0, high=1, size=(n_samples, n_features))
    y_train = np.random.choice([0, 1], size=(n_samples,), p=[.5, .5])
    X_test = np.random.uniform(low=0, high=1, size=(n_samples, n_features))
    y_test = np.random.choice([0, 1], size=(n_samples,), p=[.5, .5])
    all_rf_weights, all_K_iter_rf_data, \
        all_rf_bootstrap_output, all_rit_bootstrap_output, \
        stability_score = irf_utils.run_iRF(X_train=X_train,
                                            X_test=X_test,
                                            y_train=y_train,
                                            y_test=y_test,
                                            K=5,
                                            rf = RandomForestClassifierWithWeights(n_estimators=20),
                                            B=30,
                                            random_state_classifier=2018,
                                            propn_n_samples=.2,
                                            bin_class_type=1,
                                            M=20,
                                            max_depth=5,
                                            noisy_split=False,
                                            num_splits=2,
                                            n_estimators_bootstrap=5)
    assert np.max(all_rf_weights['rf_weight5']) < .135


def test_iRF_weight2():
    # Check when feature 1 fully predict the label, its importance should be 1.
    n_samples = 1000
    n_features = 10
    random_state_classifier = 2018
    np.random.seed(random_state_classifier)
    X_train = np.random.uniform(low=0, high=1, size=(n_samples, n_features))
    y_train = np.random.choice([0, 1], size=(n_samples,), p=[.5, .5])
    X_test = np.random.uniform(low=0, high=1, size=(n_samples, n_features))
    y_test = np.random.choice([0, 1], size=(n_samples,), p=[.5, .5])
    # first feature is very important
    X_train[:, 1] = X_train[:, 1] + y_train
    X_test[:, 1] = X_test[:, 1] + y_test
    all_rf_weights, all_K_iter_rf_data, \
        all_rf_bootstrap_output, all_rit_bootstrap_output, \
        stability_score = irf_utils.run_iRF(X_train=X_train,
                                            X_test=X_test,
                                            y_train=y_train,
                                            y_test=y_test,
                                            K=5,
                                            rf = RandomForestClassifierWithWeights(n_estimators=20),
                                            B=30,
                                            random_state_classifier=2018,
                                            propn_n_samples=.2,
                                            bin_class_type=1,
                                            M=20,
                                            max_depth=5,
                                            noisy_split=False,
                                            num_splits=2,
                                            n_estimators_bootstrap=5)
    print(all_rf_weights['rf_weight5'])
    assert all_rf_weights['rf_weight5'][1] == 1
test_iRF_weight1()
test_iRF_weight2()
