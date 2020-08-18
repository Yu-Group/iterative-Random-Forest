from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from abc import ABCMeta, abstractmethod
from ..tree.tree import (WeightedDecisionTreeClassifier, 
                         WeightedDecisionTreeRegressor)
from ..utils import get_rf_tree_data
import numpy as np

class RandomForestClassifierWithWeights(RandomForestClassifier):
    @property
    def n_paths(self):
        if not hasattr(self, "estimators_"):
            return 0
        out = 0
        for tree in self.estimators_:
            out += np.sum(tree.tree_.feature == -2)
        return out
    def fit(self, X, y, sample_weight=None, feature_weight=None):
        self.base_estimator = WeightedDecisionTreeClassifier()
        self.base_estimator_ = WeightedDecisionTreeClassifier()
        if feature_weight is not None:
            self.base_estimator.feature_weight = feature_weight
            self.base_estimator_.feature_weight = feature_weight
            
        return super(RandomForestClassifierWithWeights, self).fit(X, y, sample_weight)

class RandomForestRegressorWithWeights(RandomForestRegressor):
    @property
    def n_paths(self):
        if not hasattr(self, "estimators_"):
            return 0
        out = 0
        for tree in self.estimators_:
            out += np.sum(tree.tree_.feature == -2)
        return out
    def fit(self, X, y, sample_weight=None, feature_weight=None):
        self.base_estimator = WeightedDecisionTreeRegressor()
        self.base_estimator_ = WeightedDecisionTreeRegressor()
        if feature_weight is not None:
            self.base_estimator_.feature_weight = np.array(feature_weight).copy()
            self.base_estimator.feature_weight = np.array(feature_weight).copy()

        return super(RandomForestRegressorWithWeights, self).fit(X, y, sample_weight)
        
class wrf(RandomForestClassifierWithWeights):
    def fit(self, X, y, sample_weight=None, feature_weight=None, K = 5, 
            keep_record = True, X_test = None, y_test = None):
        self.all_rf_weights = dict()
        if keep_record:
            self.all_K_iter_rf_data = dict()
            assert X_test is not None, 'X_test should not be None when keep_record'
            assert y_test is not None, 'y_test should not be None when keep_record'
        for k in range(K):
            if k == 0:
                # Initially feature weights are None
                feature_importances = feature_weight
                self.all_rf_weights['rf_weights{}'.format(k)] = feature_importances
                
            # fit weighted RF
            # fit the classifier
            super(wrf, self).fit(
                    X=X,
                    y=y,
                    feature_weight=feature_importances)
            
            # Update feature weights using the
            # new feature importance score
            feature_importances = self.feature_importances_
            self.all_rf_weights["rf_weight{}".format(k + 1)] = feature_importances
            if keep_record:
                self.all_K_iter_rf_data["rf_iter{}".format(k+1)] = get_rf_tree_data(
                        rf=self,
                        X_train=X,
                        X_test=X_test,
                        y_test=y_test)
        return self

#Eric: Doesn't lfook like much is changed here, as it looks like no significant differences between regressor and classifier
#       come back to later to make sure there really is no difference                
class wrf_reg(RandomForestRegressorWithWeights): # Hue: change the name so that it does not clash with the first one.
    def fit(self, X, y, sample_weight=None, feature_weight=None, K = 5, 
            keep_record = True, X_test = None, y_test = None):
        self.all_rf_weights = dict()
        if keep_record:
            self.all_K_iter_rf_data = dict()
            assert X_test is not None, 'X_test should not be None when keep_record'
            assert y_test is not None, 'y_test should not be None when keep_record'
        for k in range(K):
            if k == 0:
                # Initially feature weights are None
                feature_importances = feature_weight
                self.all_rf_weights['rf_weights{}'.format(k)] = feature_importances
                
            # fit weighted RF
            # fit the regressor
            super(wrf_reg, self).fit(
                    X=X,
                    y=y,
                    feature_weight=feature_importances)
            
            # Update feature weights using the
            # new feature importance score
            feature_importances = self.feature_importances_
            self.all_rf_weights["rf_weight{}".format(k + 1)] = feature_importances
            if keep_record:
                self.all_K_iter_rf_data["rf_iter{}".format(k+1)] = get_rf_tree_data(
                        rf=self,
                        X_train=X,
                        X_test=X,
                        y_test=y)
        return self
