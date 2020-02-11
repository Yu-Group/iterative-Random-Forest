from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from abc import ABCMeta, abstractmethod
from ..tree.tree import (WeightedDecisionTreeClassifier, 
                         WeightedDecisionTreeRegressor)

class RandomForestClassifierWithWeights(RandomForestClassifier):
    def fit(self, X, y, sample_weight=None, feature_weight=None):
        if feature_weight is not None:
            self.base_estimator = WeightedDecisionTreeClassifier()
            self.base_estimator.feature_weight = feature_weight
            
        return super(RandomForestClassifierWithWeights, self).fit(X, y, sample_weight)

class RandomForestRegressorWithWeights(RandomForestRegressor):
    def fit(self, X, y, sample_weight=None, feature_weight=None):
        if feature_weight is not None:
            self.base_estimator = WeightedDecisionTreeRegressor()
            self.base_estimator.feature_weight = feature_weight

        return super(RandomForestRegressorWithWeights, self).fit(X, y, sample_weight)
        
class wrf(RandomForestClassifierWithWeights):
    def fit(self, X, y, sample_weight=None, feature_weight=None, K = 5, keep_rf = True):
        self.all_rf_weights = dict()
        self.all_rfs = dict()
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
            if keep_rf:
                self.all_rfs['rf_{}'.format(k+1)] = clone(self)
                self.all_rfs['rf_{}'.format(k+1)].estimators_ = clone(self.estimators_)
        return self

#Eric: Doesn't lfook like much is changed here, as it looks like no significant differences between regressor and classifier
#       come back to later to make sure there really is no difference                
class wrf_reg(RandomForestRegressorWithWeights): # Hue: change the name so that it does not clash with the first one.
    def fit(self, X, y, sample_weight=None, feature_weight=None, K = 5, keep_rf = True):
        self.all_rf_weights = dict()
        self.all_rfs = dict()
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
            if keep_rf:
                self.all_rfs['rf_{}'.format(k+1)] = clone(self)
                self.all_rfs['rf_{}'.format(k+1)].estimators_ = clone(self.estimators_)
        return self
