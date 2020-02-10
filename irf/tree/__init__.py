"""
The :mod:`sklearn.tree` module includes decision tree-based models for
classification and regression.
"""

from .tree import WeightedDecisionTreeClassifier
from .tree import WeightedDecisionTreeRegressor
#from .tree import ExtraTreeClassifier
#from .tree import ExtraTreeRegressor
from sklearn.tree import export_graphviz

__all__ = ["DecisionTreeClassifier", "DecisionTreeRegressor",
           "ExtraTreeClassifier", "ExtraTreeRegressor", "export_graphviz"]
