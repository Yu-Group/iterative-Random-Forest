#!/usr/bin/python

import numpy as np
from sklearn import metrics
from . import tree
from .tree import _tree
from functools import partial
from functools import reduce
from scipy import stats
import matplotlib.pyplot as plt

# Needed for the scikit-learn wrapper function
from sklearn.utils import resample
from .ensemble import RandomForestClassifierWithWeights
from math import ceil


def all_tree_paths(dtree, root_node_id=0):
    """
    Get all the individual tree paths from root node to the leaves
    for a decision tree classifier object [1]_.

    Parameters
    ----------
    dtree : DecisionTreeClassifier object
        An individual decision tree classifier object generated from a
        fitted RandomForestClassifierWithWeights object in scikit learn.

    root_node_id : int, optional (default=0)
        The index of the root node of the tree. Should be set as default to
        0 and not changed by the user

    Returns
    -------
    paths : list
        Return a list containing 1d numpy arrays of the node paths
        taken from the root node to the leaf in the decsion tree
        classifier. There is an individual array for each
        leaf node in the decision tree.

    Notes
    -----
        To obtain a deterministic behaviour during fitting,
        ``random_state`` has to be fixed.

    References
    ----------
        .. [1] https://en.wikipedia.org/wiki/Decision_tree_learning

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>> from irf.ensemble import RandomForestClassifierWithWeights
    >>> raw_data = load_breast_cancer()
    >>> X_train, X_test, y_train, y_test = train_test_split(
        raw_data.data, raw_data.target, train_size=0.9,
        random_state=2017)
    >>> rf = RandomForestClassifierWithWeights(
        n_estimators=3, random_state=random_state_classifier)
    >>> rf.fit(X=X_train, y=y_train)
    >>> estimator0 = rf.estimators_[0]
    >>> tree_dat0 = getTreeData(X_train = X_train,
                                dtree = estimator0,
                                root_node_id = 0)
    >>> tree_dat0['all_leaf_node_classes']
    ...                             # doctest: +SKIP
    ...
    [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
    """

    # Use these lists to parse the tree structure
    children_left = dtree.tree_.children_left
    children_right = dtree.tree_.children_right

    if root_node_id is None:
        paths = []

    if root_node_id == _tree.TREE_LEAF:
        raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

    # if left/right is None we'll get empty list anyway
    if children_left[root_node_id] != _tree.TREE_LEAF:
        paths = [np.append(root_node_id, l)
                 for l in all_tree_paths(dtree, children_left[root_node_id]) +
                 all_tree_paths(dtree, children_right[root_node_id])]

    else:
        paths = [root_node_id]
    return paths


def get_validation_metrics(inp_class_reg_obj, y_true, X_test):
    """
    Get the various Random Forest/ Decision Tree metrics
    This is currently setup only for classification forests and trees
        TODO/ CHECK: We need to update this for regression purposes later
        TODO/ CHECK: For classification we need to validate that
               the maximum number of
               labels is 2 for the training/ testing data

    Get all the individual tree paths from root node to the leaves
    for a decision tree classifier object [1]_.

    Parameters
    ----------
    inp_class_reg_obj : DecisionTreeClassifier or RandomForestClassifierWithWeights
        object [1]_
        An individual decision tree or random forest classifier
        object generated from a fitted Classifier object in scikit learn.

    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    Returns
    -------
    classification_metrics : dict
        Return a dictionary containing various validation metrics on
        the input fitted Classifier object

    References
    ----------
        .. [1] https://en.wikipedia.org/wiki/Decision_tree_learning

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>> from irf.ensemble import RandomForestClassifierWithWeights
    >>> raw_data = load_breast_cancer()
    >>> X_train, X_test, y_train, y_test = train_test_split(
        raw_data.data, raw_data.target, train_size=0.9,
        random_state=2017)
    >>> rf = RandomForestClassifierWithWeights(
        n_estimators=3, random_state=random_state_classifier)
    >>> rf.fit(X=X_train, y=y_train)
    >>> rf_metrics = get_validation_metrics(inp_class_reg_obj = rf,
                                          y_true = y_test,
                                          X_test = X_test)
    >>> rf_metrics['confusion_matrix']

    ...                             # doctest: +SKIP
    ...
    array([[12,  2],
          [ 1, 42]])
    """

    # If the object is not a scikit learn classifier then let user know
    if type(inp_class_reg_obj).__name__ not in \
       ["DecisionTreeClassifier", "WeightedDecisionTreeClassifier", "RandomForestClassifierWithWeights"]:
        raise TypeError("input needs to be a DecisionTreeClassifier object, \
        you have input a {} object".format(type(inp_class_reg_obj)))

    # if the number of classes is not binary let the user know accordingly
    if inp_class_reg_obj.n_classes_ != 2:
        raise ValueError("The number of classes for classification must \
        be binary, you currently have fit to {} \
        classes".format(inp_class_reg_obj.n_classes_))

    # Get the predicted values on the validation data
    y_pred = inp_class_reg_obj.predict(X=X_test)

    # CLASSIFICATION metrics calculations

    # Cohen's kappa: a statistic that measures inter-annotator agreement.
    # cohen_kappa_score = metrics.cohen_kappa_score(y1, y2[, labels, ...])

    # Compute Area Under the Curve (AUC) using the trapezoidal rule
    # fpr, tpr, thresholds = metrics.roc_curve(y_true = y_true,
    #                                          y_pred = y_pred)
    # auc = metrics.auc(fpr, tpr)

    # Compute average precision (AP) from prediction scores
    # average_precision_score = metrics.average_precision_score(y_true =
    # y_true, y_score)

    # Compute the Brier score.
    # metrics.brier_score_loss(y_true = y_true, y_prob[, ...])

    # Compute the F-beta score
    # metrics.fbeta_score(y_true = y_true, y_pred = y_pred, beta[, ...])

    # Average hinge loss (non-regularized)
    # metrics.hinge_loss(y_true = y_true, pred_decision[, ...])

    # Compute the Matthews correlation coefficient (MCC) for binary classes
    # metrics.matthews_corrcoef(y_true = y_true, y_pred[, ...])

    # Compute precision-recall pairs for different probability thresholds
    # metrics.precision_recall_curve(y_true = y_true, ...)

    # Compute precision, recall, F-measure and support for each class
    # metrics.precision_recall_fscore_support(...)

    # Compute Area Under the Curve (AUC) from prediction scores
    # metrics.roc_auc_score(y_true = y_true, y_score[, ...])

    # Compute Receiver operating characteristic (ROC)
    # metrics.roc_curve(y_true = y_true, y_score[, ...])

    # Jaccard similarity coefficient score
    # jaccard_similarity_score =
    # metrics.jaccard_similarity_score(y_true = y_true, y_pred = y_pred)

    # Compute the F1 score, also known as balanced F-score or F-measure
    f1_score = metrics.f1_score(y_true=y_true, y_pred=y_pred)

    # Compute the average Hamming loss.
    hamming_loss = metrics.hamming_loss(y_true=y_true, y_pred=y_pred)

    # Log loss, aka logistic loss or cross-entropy loss.
    log_loss = metrics.log_loss(y_true=y_true, y_pred=y_pred)

    # Compute the precision
    precision_score = metrics.precision_score(y_true=y_true, y_pred=y_pred)

    # Compute the recall
    recall_score = metrics.recall_score(y_true=y_true, y_pred=y_pred)

    # Accuracy classification score
    accuracy_score = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)

    # Build a text report showing the main classification metrics
    # classification_report = metrics.classification_report(
    # y_true=y_true, y_pred=y_pred)

    # Compute confusion matrix to evaluate the accuracy of a classification
    confusion_matrix = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)

    # Zero-one classification loss.
    zero_one_loss = metrics.zero_one_loss(y_true=y_true, y_pred=y_pred)

    # Load all metrics into a single dictionary
    classification_metrics = {"hamming_loss": hamming_loss,
                              "log_loss": log_loss,
                              "recall_score": recall_score,
                              "precision_score": precision_score,
                              "accuracy_score": accuracy_score,
                              "f1_score": f1_score,
                              # "classification_report": classification_report,
                              "confusion_matrix": confusion_matrix,
                              "zero_one_loss": zero_one_loss}

    return classification_metrics


def get_tree_data(X_train, X_test, y_test, dtree, root_node_id=0):
    """
    This returns all of the required summary results from an
    individual decision tree

    Parameters
    ----------
    dtree : DecisionTreeClassifier object
        An individual decision tree classifier object generated from a
        fitted RandomForestClassifierWithWeights object in scikit learn.

    X_train : array-like or sparse matrix, shape = [n_samples, n_features]
        Training vector, where n_samples in the number of samples and
        n_features is the number of features.

    X_test : array-like or sparse matrix, shape = [n_samples, n_features]
        Test vector, where n_samples in the number of samples and
        n_features is the number of features.

    y_test : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    root_node_id : int, optional (default=0)
        The index of the root node of the tree. Should be set as default to
        0 and not changed by the user

    Returns
    -------
    tree_data : dict
        Return a dictionary containing various tree metrics
    from the input fitted Classifier object

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.ensemble import RandomForestClassifierWithWeights
    >>> raw_data = load_breast_cancer()
    >>> X_train, X_test, y_train, y_test = train_test_split(
        raw_data.data, raw_data.target, train_size=0.9,
        random_state=2017)
    >>> rf = RandomForestClassifierWithWeights(
        n_estimators=3, random_state=2018)
    >>> rf.fit(X=X_train, y=y_train)
    >>> estimator0 = rf.estimators_[0]
    >>> estimator0_out = get_tree_data(X_train=X_train,
                                     dtree=estimator0,
                                     root_node_id=0)
    >>> print(estimator0_out['all_leaf_nodes'])
    ...                             # doctest: +SKIP
    ...
    [6, 8, 9, 10, 12, 14, 15, 19, 22, 23, 24, \
     25, 26, 29, 30, 32, 34, 36, 37, 40, 41, 42]
    """

    max_node_depth = dtree.tree_.max_depth
    n_nodes = dtree.tree_.node_count
    value = dtree.tree_.value
    n_node_samples = dtree.tree_.n_node_samples
    root_n_node_samples = float(dtree.tree_.n_node_samples[0])
    X_train_n_samples = X_train.shape[0]

    # Get the total number of features in the training data
    tot_num_features = X_train.shape[1]

    # Get indices for all the features used - 0 indexed and ranging
    # to the total number of possible features in the training data
    all_features_idx = np.array(range(tot_num_features), dtype='int64')

    # Get the raw node feature indices from the decision tree classifier
    # attribute positive and negative - we want only non-negative indices
    # It is hard to tell which features this came from i.e. indices
    # are zero, corresponding feature columns for consistency
    # in reference
    node_features_raw_idx = dtree.tree_.feature

    # Get the refined non-negative feature indices for each node
    # Start with a range over the total number of features and
    # subset the relevant indices from the raw indices array
    # print(np.array(node_features_raw_idx))
    node_features_idx = all_features_idx[np.array(node_features_raw_idx)]

    # Count the unique number of features used
    num_features_used = (np.unique(node_features_idx)).shape[0]

    # Get all of the paths used in the tree
    all_leaf_node_paths = all_tree_paths(dtree=dtree,
                                         root_node_id=root_node_id)

    # Get list of leaf nodes
    # In all paths it is the final node value
    all_leaf_nodes = [path[-1] for path in all_leaf_node_paths]

    # Get the total number of training samples used in each leaf node
    all_leaf_node_samples = [n_node_samples[node_id].astype(int)
                             for node_id in all_leaf_nodes]

    # Get proportion of training samples used in each leaf node
    # compared to the training samples used in the root node
    all_leaf_node_samples_percent = [
        100. * n_leaf_node_samples / root_n_node_samples
        for n_leaf_node_samples in all_leaf_node_samples]

    # Final predicted values in each class at each leaf node
    all_leaf_node_values = [value[node_id].astype(
        int) for node_id in all_leaf_nodes]

    # Scaled values of the leaf nodes in each of the binary classes
    all_scaled_leaf_node_values = [value / X_train_n_samples
                                   for value in all_leaf_node_values]

    # Total number of training samples used in the prediction of
    # each class at each leaf node
    tot_leaf_node_values = [np.sum(leaf_node_values)
                            for leaf_node_values in all_leaf_node_values]

    # All leaf node depths
    # The depth is 0 indexed i.e. root node has depth 0
    leaf_nodes_depths = [np.size(path) - 1 for path in all_leaf_node_paths]

    # Predicted Classes
    # Check that we correctly account for ties in determining the class here
    all_leaf_node_classes = [all_features_idx[np.argmax(
        value)] for value in all_leaf_node_values]

    # Get all of the features used along the leaf node paths i.e.
    # features used to split a node
    # CHECK: Why does the leaf node have a feature associated with it?
    # Investigate further
    # Removed the final leaf node value so that this feature does not get
    # included currently
    all_leaf_paths_features = [node_features_idx[path[:-1]]
                               for path in all_leaf_node_paths]

    # Get the unique list of features along a path
    # NOTE: This removes the original ordering of the features along the path
    # The original ordering could be preserved using a special function but
    # will increase runtime
    all_uniq_leaf_paths_features = [
        np.unique(feature_path) for feature_path in all_leaf_paths_features]

    # get the validation classification metrics for the
    # decision tree against the test data
    validation_metrics = get_validation_metrics(inp_class_reg_obj=dtree,
                                                y_true=y_test,
                                                X_test=X_test)

    # Dictionary of all tree values
    tree_data = {"num_features_used": num_features_used,
                 "node_features_idx": node_features_idx,
                 "max_node_depth": max_node_depth,
                 "n_nodes": n_nodes,
                 "all_leaf_node_paths": all_leaf_node_paths,
                 "all_leaf_nodes": all_leaf_nodes,
                 "leaf_nodes_depths": leaf_nodes_depths,
                 "all_leaf_node_samples": all_leaf_node_samples,
                 "all_leaf_node_samples_percent":
                 all_leaf_node_samples_percent,
                 "all_leaf_node_values": all_leaf_node_values,
                 "all_scaled_leaf_node_values": all_scaled_leaf_node_values,
                 "tot_leaf_node_values": tot_leaf_node_values,
                 "all_leaf_node_classes": all_leaf_node_classes,
                 "all_leaf_paths_features": all_leaf_paths_features,
                 "all_uniq_leaf_paths_features": all_uniq_leaf_paths_features,
                 "validation_metrics": validation_metrics}
    return tree_data

# Get all RF and decision tree data


def get_rf_tree_data(rf, X_train, X_test, y_test):
    """
    Get the entire fitted random forest and its decision tree data
    as a convenient dictionary format
    """

    # random forest feature importances i.e. next iRF iteration weights
    feature_importances = rf.feature_importances_

    # standard deviation of the feature importances
    feature_importances_std = np.std(
        [dtree.feature_importances_ for dtree in rf.estimators_], axis=0)
    feature_importances_rank_idx = np.argsort(feature_importances)[::-1]

    # get all the validation rf_metrics
    rf_validation_metrics = get_validation_metrics(inp_class_reg_obj=rf,
                                                   y_true=y_test,
                                                   X_test=X_test)

    # Create a dictionary with all random forest metrics
    # This currently includes the entire random forest fitted object
    all_rf_tree_outputs = {"rf_obj": rf,
                           "get_params": rf.get_params,
                           "rf_validation_metrics": rf_validation_metrics,
                           "feature_importances": feature_importances,
                           "feature_importances_std": feature_importances_std,
                           "feature_importances_rank_idx":
                           feature_importances_rank_idx}

    # CHECK: Ask SVW if the following should be paralellized!
    for idx, dtree in enumerate(rf.estimators_):
        dtree_out = get_tree_data(X_train=X_train,
                                  X_test=X_test,
                                  y_test=y_test,
                                  dtree=dtree,
                                  root_node_id=0)

        # Append output to our combined random forest outputs dict
        all_rf_tree_outputs["dtree{}".format(idx)] = dtree_out

    return all_rf_tree_outputs


# Random Intersection Tree (RIT)

def get_rit_tree_data(all_rf_tree_data,
                      bin_class_type=1,
                      M=10,  # number of trees (RIT) to build
                      max_depth=3,
                      noisy_split=False,
                      num_splits=2):
    """
    A wrapper for the Random Intersection Trees (RIT) algorithm
    """
    #FIXME no prevalence cutoff for rit

    all_rit_tree_outputs = {}
    for idx, rit_tree in enumerate(range(M)):

        # Create the weighted randomly sampled paths as a generator
        gen_random_leaf_paths = generate_rit_samples(
            all_rf_tree_data=all_rf_tree_data,
            bin_class_type=bin_class_type)

        # Create the RIT object
        rit = build_tree(feature_paths=gen_random_leaf_paths,
                         max_depth=max_depth,
                         noisy_split=noisy_split,
                         num_splits=num_splits)

        # Get the intersected node values
        # CHECK remove this for the final value
        rit_intersected_values = [
            node[1]._val for node in rit.traverse_depth_first()]
        # Leaf node values i.e. final intersected features
        rit_leaf_node_values = [node[1]._val for node in rit.leaf_nodes()]
        rit_leaf_node_union_value = reduce(np.union1d, rit_leaf_node_values)
        rit_output = {"rit": rit,
                      "rit_intersected_values": rit_intersected_values,
                      "rit_leaf_node_values": rit_leaf_node_values,
                      "rit_leaf_node_union_value": rit_leaf_node_union_value}
        # Append output to our combined random forest outputs dict
        all_rit_tree_outputs["rit{}".format(idx)] = rit_output

    return all_rit_tree_outputs


# FILTERING leaf paths
# Filter Comprehension helper function


def _dtree_filter_comp(dtree_data,
                       filter_key,
                       bin_class_type):
    """
    List comprehension filter helper function to filter
    the data from the `get_tree_data` function output

    Parameters
    ----------
    dtree_data : dictionary
        Summary dictionary output after calling `get_tree_data` on a
        scikit learn decision tree object

    filter_key : str
        The specific variable from the summary dictionary
        i.e. `dtree_data` which we want to filter based on
        leaf class_names

    bin class type : int
        Takes a {0,1} class-value depending on the class
        to be filtered

    Returns
    -------
    tree_data : list
        Return a list containing specific tree metrics
        from the input fitted Classifier object

    """

    # Decision Tree values to filter
    dtree_values = dtree_data[filter_key]

    # Filter based on the specific value of the leaf node classes
    leaf_node_classes = dtree_data['all_leaf_node_classes']

    # perform the filtering and return list
    return [i for i, j in zip(dtree_values,
                              leaf_node_classes)
            if j == bin_class_type]


def filter_leaves_classifier(dtree_data,
                             bin_class_type):
    """
    Filters the leaf node data from a decision tree
    for either {0,1} classes for iRF purposes

    Parameters
    ----------
    dtree_data : dictionary
        Summary dictionary output after calling `get_tree_data` on a
        scikit learn decision tree object

    bin class type : int
        Takes a {0,1} class-value depending on the class
        to be filtered

    Returns
    -------
    all_filtered_outputs : dict
        Return a dictionary containing various lists of
        specific tree metrics for each leaf node from the
        input classifier object
    """

    filter_comp = partial(_dtree_filter_comp,
                          dtree_data=dtree_data,
                          bin_class_type=bin_class_type)

    # Get Filtered values by specified binary class

    # unique feature paths from root to leaf node
    uniq_feature_paths = filter_comp(filter_key='all_uniq_leaf_paths_features')

    # total number of training samples ending up at each node
    tot_leaf_node_values = filter_comp(filter_key='tot_leaf_node_values')

    # depths of each of the leaf nodes
    leaf_nodes_depths = filter_comp(filter_key='leaf_nodes_depths')

    # validation metrics for the tree
    validation_metrics = dtree_data['validation_metrics']

    # return all filtered outputs as a dictionary
    all_filtered_outputs = {"uniq_feature_paths": uniq_feature_paths,
                            "tot_leaf_node_values": tot_leaf_node_values,
                            "leaf_nodes_depths": leaf_nodes_depths,
                            "validation_metrics": validation_metrics}

    return all_filtered_outputs


def weighted_random_choice(values, weights):
    """
    Discrete distribution, drawing values with the frequency
    specified in weights.
    Weights do not need to be normalized.
    Parameters:
        values: list of values 
    Return:
        a generator that do weighted sampling
    """
    if not len(weights) == len(values):
        raise ValueError('Equal number of values and weights expected')

    weights = np.array(weights)
    # normalize the weights
    weights = weights / weights.sum()
    dist = stats.rv_discrete(values=(range(len(weights)), weights))
    #FIXME this part should be improved by assigning values directly
    #    to the stats.rv_discrete function.  -- Yu

    while True:
        yield values[dist.rvs()]


def generate_rit_samples(all_rf_tree_data, bin_class_type=1):
    """
    Draw weighted samples from all possible decision paths
    from the decision trees in the fitted random forest object
    """

    # Number of decision trees
    n_estimators = all_rf_tree_data['rf_obj'].n_estimators

    all_weights = []
    all_paths = []
    for dtree in range(n_estimators):
        filtered = filter_leaves_classifier(
            dtree_data=all_rf_tree_data['dtree{}'.format(dtree)],
            bin_class_type=bin_class_type)
        all_weights.extend(filtered['tot_leaf_node_values'])
        all_paths.extend(filtered['uniq_feature_paths'])

    # Return the generator of randomly sampled observations
    # by specified weights
    return weighted_random_choice(all_paths, all_weights)


def select_random_path():
    X = np.random.random(size=(80, 100)) > 0.3
    XX = [np.nonzero(row)[0] for row in X]
    # Create the random array generator
    while True:
        yield XX[np.random.randint(low=0, high=len(XX))]


class RITNode(object):
    """
    A helper class used to construct the RIT Node
    in the generation of the Random Intersection Tree (RIT)
    """

    def __init__(self, val):
        self._val = val
        self._children = []

    def is_leaf(self):
        return len(self._children) == 0

    @property
    def children(self):
        return self._children

    def add_child(self, val):
        val_intersect = np.intersect1d(self._val, val)
        self._children.append(RITNode(val_intersect))

    def is_empty(self):
        return len(self._val) == 0

    @property
    def nr_children(self):
        return len(self._children) + \
            sum(child.nr_children for child in self._children)

    def _traverse_depth_first(self, _idx):
        yield _idx[0], self
        for child in self.children:
            _idx[0] += 1
            yield from RITNode._traverse_depth_first(child, _idx=_idx)


class RITTree(RITNode):
    """
    Class for constructing the RIT
    """

    def __len__(self):
        return self.nr_children + 1

    def traverse_depth_first(self):
        yield from RITNode._traverse_depth_first(self, _idx=[0])

    def leaf_nodes(self):
        for node in self.traverse_depth_first():
            if node[1].is_leaf():
                yield node

                #


def build_tree(feature_paths, max_depth=3,
               num_splits=5, noisy_split=False,
               _parent=None,
               _depth=0):
    """

    Builds out the random intersection tree based
    on the specified parameters [1]_

    Parameters
    ----------
    feature_paths : generator of list of ints
    ...

    max_depth : int
        The built tree will never be deeper than `max_depth`.

    num_splits : int
            At each node, the maximum number of children to be added.

    noisy_split: bool
        At each node if True, then number of children to
        split will be (`num_splits`, `num_splits + 1`)
        based on the outcome of a bernoulli(0.5)
        random variable

    References
    ----------
        .. [1] Shah, Rajen Dinesh, and Nicolai Meinshausen.
                "Random intersection trees." Journal of
                Machine Learning Research 15.1 (2014): 629-654.
    """

    expand_tree = partial(build_tree, feature_paths,
                          max_depth=max_depth,
                          num_splits=num_splits,
                          noisy_split=noisy_split)

    if _parent is None:
        tree = RITTree(next(feature_paths))
        expand_tree(_parent=tree, _depth=0)
        return tree

    else:
        _depth += 1
        if _depth >= max_depth:
            return
        if noisy_split:
            num_splits += np.random.randint(low=0, high=2)
        for i in range(num_splits):
            _parent.add_child(next(feature_paths))
            added_node = _parent.children[-1]
            if not added_node.is_empty():
                expand_tree(_parent=added_node, _depth=_depth)


# extract interactions from RIT output
def rit_interactions(all_rit_tree_data):
    """
    Extracts all interactions produced by one run of RIT
    To get interactions across many runs of RIT (like when we do bootstrap \
        sampling for stability),
        first concantenate those dictionaries into one

    Parameters
    ------
    all_rit_tree_data : dict
        Output of RIT as defined by the function 'get_rit_tree_data'

    Returns
    ------
    interact_counts : dict
        A dictionary whose keys are the discovered interactions and
        whose values store their respective frequencies
    """

    interactions = []
    # loop through all trees
    for k in all_rit_tree_data:
        # loop through all found interactions
        for j in range(len(all_rit_tree_data[k]['rit_intersected_values'])):
            # if not null:
            if len(all_rit_tree_data[k]['rit_intersected_values'][j]) != 0:

                # stores interaction as string : eg. np.array([1,12,23])
                # becomes '1_12_23'
                a = '_'.join(
                    map(str,
                        all_rit_tree_data[k]['rit_intersected_values'][j]))
                interactions.append(a)

    interact_counts = {m: interactions.count(m) for m in interactions}
    return interact_counts


def _get_histogram(interact_counts, xlabel='interaction',
                   ylabel='stability',
                   sort=False):
    """
    Helper function to plot the histogram from a dictionary of
    count data

    Paremeters
    -------
    interact_counts : dict
        counts of interactions as outputed from the 'rit_interactions' function

    xlabel : str, optional (default = 'interaction')
        label on the x-axis

    ylabel : str, optional (default = 'counts')
        label on the y-axis

    sorted : boolean, optional (default = 'False')
        If True, sort the histogram from interactions with highest frequency
        to interactions with lowest frequency
    """

    if sort:
        data_y = sorted(interact_counts.values(), reverse=True)
        data_x = sorted(interact_counts, key=interact_counts.get,
                        reverse=True)
    else:
        data_x = interact_counts.keys()
        data_y = interact_counts.values()

    plt.figure(figsize=(15, 8))
    plt.clf()
    plt.bar(np.arange(len(data_x)), data_y, align='center', alpha=0.5)
    plt.xticks(np.arange(len(data_x)), data_x, rotation='vertical')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def _get_stability_score(all_rit_bootstrap_output):
    """
    Get the stabilty score from B bootstrap Random Forest
    Fits with RITs
    """

    # Initialize values
    bootstrap_interact = []
    B = len(all_rit_bootstrap_output)

    for b in range(B):
        rit_counts = rit_interactions(
            all_rit_bootstrap_output['rf_bootstrap{}'.format(b)])
        rit_counts = list(rit_counts.keys())
        bootstrap_interact.append(rit_counts)

    def flatten(l): return [item for sublist in l for item in sublist]
    all_rit_interactions = flatten(bootstrap_interact)
    stability = {m: all_rit_interactions.count(
        m) / B for m in all_rit_interactions}
    return stability

def run_iRF(X_train,
            X_test,
            y_train,
            y_test,
            rf,
            initial_weights = None,
            K=7,
            B=10,
            random_state_classifier=2018,
            propn_n_samples=0.2,
            bin_class_type=1,
            M=4,
            max_depth=2,
            noisy_split=False,
            num_splits=2,
            n_estimators_bootstrap=5):
    """
    Runs the iRF algorithm in full.


    Parameters
    --------
    X_train : array-like or sparse matrix, shape = [n_samples, n_features]
        Training vector, where n_samples in the number of samples and
        n_features is the number of features.

    X_test : array-like or sparse matrix, shape = [n_samples, n_features]
        Test vector, where n_samples in the number of samples and
        n_features is the number of features.

    y_train : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values for training.

    y_test : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values for testing.

    rf : RandomForestClassifierWithWeights to fit


    K : int, optional (default = 7)
        The number of iterations in iRF.

    n_estimators : int, optional (default = 20)
        The number of trees in the random forest when computing weights.

    B : int, optional (default = 10)
        The number of bootstrap samples

    random_state_classifier : int, optional (default = 2018)
        The random seed for reproducibility.

    propn_n_samples : float, optional (default = 0.2)
        The proportion of samples drawn for bootstrap.

    bin_class_type : int, optional (default = 1)
        ...

    max_depth : int, optional (default = 2)
        The built tree will never be deeper than `max_depth`.

    num_splits : int, optional (default = 2)
            At each node, the maximum number of children to be added.

    noisy_split: bool, optional (default = False)
        At each node if True, then number of children to
        split will be (`num_splits`, `num_splits + 1`)
        based on the outcome of a bernoulli(0.5)
        random variable

    n_estimators_bootstrap : int, optional (default = 5)
        The number of trees in the random forest when
        fitting to bootstrap samples

    Returns
    --------
    all_rf_weights: dict
        stores feature weights across all iterations

    all_rf_bootstrap_output: dict
        stores rf information across all bootstrap samples

    all_rit_bootstrap_output: dict
        stores rit information across all bootstrap samples

    stability_score: dict
        stores interactions in as its keys and stabilities scores as the values

    """

    # Set the random state for reproducibility
    np.random.seed(random_state_classifier)

    # Convert the bootstrap resampling proportion to the number
    # of rows to resample from the training data
    n_samples = ceil(propn_n_samples * X_train.shape[0])

    # All Random Forest data
    all_K_iter_rf_data = {}

    # Initialize dictionary of rf weights
    # CHECK: change this name to be `all_rf_weights_output`
    all_rf_weights = {}

    # Initialize dictionary of bootstrap rf output
    all_rf_bootstrap_output = {}

    # Initialize dictionary of bootstrap RIT output
    all_rit_bootstrap_output = {}

    for k in range(K):
        if k == 0:

            # Initially feature weights are None
            feature_importances = initial_weights

            # Update the dictionary of all our RF weights
            all_rf_weights["rf_weight{}".format(k)] = feature_importances

            # fit the classifier
            rf.fit(
                X=X_train,
                y=y_train,
                feature_weight=all_rf_weights["rf_weight{}".format(k)])

            # Update feature weights using the
            # new feature importance score
            feature_importances = rf.feature_importances_

            # Load the weights for the next iteration
            all_rf_weights["rf_weight{}".format(k + 1)] = feature_importances

        else:
            # fit weighted RF
            # Use the weights from the previous iteration
            rf.fit(
                X=X_train,
                y=y_train,
                feature_weight=all_rf_weights["rf_weight{}".format(k)])

            # Update feature weights using the
            # new feature importance score
            feature_importances = rf.feature_importances_

            # Load the weights for the next iteration
            all_rf_weights["rf_weight{}".format(k + 1)] = feature_importances

        all_K_iter_rf_data["rf_iter{}".format(k)] = get_rf_tree_data(
            rf=rf,
            X_train=X_train,
            X_test=X_test,
            y_test=y_test)

    # Run the RITs
    for b in range(B):

        # Take a bootstrap sample from the training data
        # based on the specified user proportion
        X_train_rsmpl, y_rsmpl = resample(
            X_train, y_train, n_samples=n_samples)
        # FIXME in iRF R package, when y is discrete, this should be a stratified bootstrap

        # Set up the weighted random forest
        # Using the weight from the (K-1)th iteration i.e. RF(w(K))
        rf_bootstrap = RandomForestClassifierWithWeights(
            # CHECK: different number of trees to fit for bootstrap samples
            n_estimators=n_estimators_bootstrap)

        # Fit RF(w(K)) on the bootstrapped dataset
        rf_bootstrap.fit(
            X=X_train_rsmpl,
            y=y_rsmpl,
            feature_weight=all_rf_weights["rf_weight{}".format(K)])

        # All RF tree data
        # CHECK: why do we need y_train here?
        all_rf_tree_data = get_rf_tree_data(
            rf=rf_bootstrap,
            X_train=X_train_rsmpl,
            X_test=X_test,
            y_test=y_test)

        # Update the rf bootstrap output dictionary
        all_rf_bootstrap_output['rf_bootstrap{}'.format(b)] = all_rf_tree_data

        # Run RIT on the interaction rule set
        # CHECK - each of these variables needs to be passed into
        # the main run_rit function
        all_rit_tree_data = get_rit_tree_data(
            all_rf_tree_data=all_rf_tree_data,
            bin_class_type=bin_class_type,
            M=M,
            max_depth=max_depth,
            noisy_split=noisy_split,
            num_splits=num_splits)

        # Update the rf bootstrap output dictionary
        # We will reference the RIT for a particular rf bootstrap
        # using the specific bootstrap id - consistent with the
        # rf bootstrap output data
        all_rit_bootstrap_output['rf_bootstrap{}'.format(
            b)] = all_rit_tree_data

    stability_score = _get_stability_score(
        all_rit_bootstrap_output=all_rit_bootstrap_output)

    return all_rf_weights,\
        all_K_iter_rf_data, all_rf_bootstrap_output,\
        all_rit_bootstrap_output, stability_score


def _hist_features(all_rf_tree_data, n_estimators,
                   xlabel='features',
                   ylabel='frequency',
                   title='Frequency of features along decision paths'):
    """
    Generate histogram of number of appearances a feature appeared
    along a decision path in the forest
    """

    all_features = []

    for i in range(n_estimators):
        tree_id = 'dtree' + str(i)

        a = np.concatenate(
            all_rf_tree_data[tree_id]['all_uniq_leaf_paths_features'])
        all_features.append(a)

    all_features = np.concatenate(all_features)

    counts = {m: np.sum(all_features == m) for m in all_features}
    data_y = sorted(counts.values(), reverse=True)
    data_x = sorted(counts, key=counts.get, reverse=True)
    plt.figure(figsize=(15, 8))
    plt.clf()
    plt.bar(np.arange(len(data_x)), data_y, align='center', alpha=0.5)
    plt.xticks(np.arange(len(data_x)), data_x, rotation='vertical')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
