from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
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
    def fit(self, X, y, sample_weight=None, feature_weight=None, K = 5):
        for k in range(K):
            if k == 0:
                # Initially feature weights are None
                feature_importances = feature_weight
                
                # fit the classifier
                super(wrf, self).fit(X=X,
                         y=y,
                         feature_weight=feature_importances)
                
                # Update feature weights using the
                # new feature importance score
                feature_importances = self.feature_importances_

            else:
                # fit weighted RF
                # fit the classifier
                super(wrf, self).fit(
                        X=X,
                        y=y,
                        feature_weight=feature_importances)
                
                # Update feature weights using the
                # new feature importance score
                feature_importances = self.feature_importances_
        return self

#Eric: Doesn't lfook like much is changed here, as it looks like no significant differences between regressor and classifier
#       come back to later to make sure there really is no difference                
class wrf_reg(RandomForestRegressorWithWeights): # Hue: change the name so that it does not clash with the first one.
    def fit(self, X, y, sample_weight=None, feature_weight=None, K = 5):
        for k in range(K):
            if k == 0:
                # Initially feature weights are None
                feature_importances = feature_weight
                
                # fit the classifier
                super(wrf_reg, self).fit(X=X,
                         y=y,
                         feature_weight=feature_importances)
                
                # Update feature weights using the
                # new feature importance score
                feature_importances = self.feature_importances_

            else:
                # fit weighted RF
                # fit the regressor
                super(wrf_reg, self).fit(
                        X=X,
                        y=y,
                        feature_weight=feature_importances)
                
                # Update feature weights using the
                # new feature importance score
                feature_importances = self.feature_importances_
        return self
