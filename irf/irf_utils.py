#!/usr/bin/python

import numpy as np
from sklearn import metrics
from . import tree
from .tree import _tree
from functools import partial
from functools import reduce
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.base import (clone, 
                          ClassifierMixin,
                          RegressorMixin)
from .utils import get_rf_tree_data

# Needed for the scikit-learn wrapper function
from sklearn.utils import resample
from sklearn.ensemble import (RandomForestClassifier,
                              RandomForestRegressor)
from .ensemble import (wrf, wrf_reg)

from math import ceil

# Needed for FPGrowth
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession
from pyspark.sql.types import *



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
            if bin_class_type is None or j == bin_class_type]


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
    if len(weights) == 0:
        raise ValueError("weights has zero length.")

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
    n_estimators = all_rf_tree_data['get_params']['n_estimators']

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

def _FP_Growth_get_stability_score(all_FP_Growth_bootstrap_output, bootstrap_num):
    """
    Get the stabilty score from B bootstrap Random Forest
    Fits with FP-Growth
    """

    # Initialize values
    bootstrap_interact = []
    B = len(all_FP_Growth_bootstrap_output)

    for b in range(B):
        itemsets = all_FP_Growth_bootstrap_output['rf_bootstrap{}'.format(b)]
        top_itemsets = itemsets.head(bootstrap_num)
        top_itemsets = list(top_itemsets["items"].map(lambda s: "_".join([str(x) for x in sorted(s)])))
        bootstrap_interact.append(top_itemsets)

    def flatten(l): return [item for sublist in l for item in sublist]
    all_FP_Growth_interactions = flatten(bootstrap_interact)
    stability = {m: all_FP_Growth_interactions.count(
        m) / B for m in all_FP_Growth_interactions}
    return stability

def run_iRF(X_train,
            X_test,
            y_train,
            y_test,
            rf,
            rf_bootstrap=None,
            initial_weights = None,
            K=7,
            B=10,
            random_state_classifier=2018,
            signed=False,
            propn_n_samples=0.2,
            bin_class_type=1,
            M=4,
            max_depth=2,
            noisy_split=False,
            num_splits=2,
            n_estimators_bootstrap=5):
    """
    Runs the iRF algorithm.


    Parameters
    ----------
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

    rf : RandomForestClassifier/Regressor to fit, it will not be used directly
        Only the parameters of rf will be used.

    rf_bootstrap : RandomForest model to fit to the bootstrap samples, optional
        default None, which means the same as rf

    K : int, optional (default = 7)
        The number of iterations in iRF.

    n_estimators : int, optional (default = 20)
        The number of trees in the random forest when computing weights.

    B : int, optional (default = 10)
        The number of bootstrap samples

    signed : bool, optional (default = False)
        Whether use signed interaction or not

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
    
    
    if issubclass(type(rf), RandomForestClassifier):
        weightedRF = wrf(**rf.get_params())
    elif issubclass(type(rf), RandomForestRegressor):
        weightedRF = wrf_reg(**rf.get_params())
    else:
        raise ValueError('the type of rf cannot be {}'.format(type(rf)))
    
    weightedRF.fit(X=X_train, y=y_train, feature_weight = initial_weights, K=K,
                   X_test = X_test, y_test = y_test)
    all_rf_weights = weightedRF.all_rf_weights
    all_K_iter_rf_data = weightedRF.all_K_iter_rf_data
    

    # Run the RITs
    for b in range(B):

        # Take a bootstrap sample from the training data
        # based on the specified user proportion
        if isinstance(rf, ClassifierMixin):
            X_train_rsmpl, y_rsmpl = resample(
                X_train, y_train, n_samples=n_samples, stratify = y_train)
        else:
            X_train_rsmpl, y_rsmpl = resample(
                X_train, y_train, n_samples=n_samples)
            
        

        # Set up the weighted random forest
        # Using the weight from the (K-1)th iteration i.e. RF(w(K))
        if rf_bootstrap is None:
            rf_bootstrap = clone(rf)
        
        # CHECK: different number of trees to fit for bootstrap samples
        rf_bootstrap.n_estimators=n_estimators_bootstrap

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
            y_test=y_test,
            signed=signed)

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

def run_iRF_FPGrowth(X_train,
            X_test,
            y_train,
            y_test,
            rf,
            rf_bootstrap = None,
            initial_weights = None,
            K=7,
            B=10,
            random_state_classifier=2018,
            propn_n_samples=0.2,
            bin_class_type=1,
            min_confidence=0.8,
            min_support=0.1,
            signed=False,
            n_estimators_bootstrap=5,
            bootstrap_num=5):
    """
    Runs the iRF algorithm but instead of RIT for interactions, runs FP-Growth through Spark.


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

    rf : RandomForest model to fit
    
    rf_bootstrap : random forest model to fit in the RIT stage, default None, which means it is the same as rf.
        The number of trees in this model should be set smaller as this step is quite time consuming.

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

    min_confidence: float, optional (default = 0.8)
        FP-Growth has a parameter min_confidence which is the minimum frequency of an interaction set amongst all transactions
        in order for it to be returned
    
    bootstrap_num: float, optional (default = 5)
        Top number used in computing the stability score


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

    # Initialize dictionary of bootstrap FP-Growth output
    all_FP_Growth_bootstrap_output = {}
    
    if issubclass(type(rf), RandomForestClassifier):
        weightedRF = wrf(**rf.get_params())
    elif issubclass(type(rf) is RandomForestRegressor):
        weightedRF = wrf_reg(**rf.get_params())
    else:
        raise ValueError('the type of rf cannot be {}'.format(type(rf)))
    
    weightedRF.fit(X=X_train, y=y_train, feature_weight = initial_weights, K=K,
                   X_test = X_test, y_test = y_test)
    all_rf_weights = weightedRF.all_rf_weights
    all_K_iter_rf_data = weightedRF.all_K_iter_rf_data

    # Run the FP-Growths
    if rf_bootstrap is None:
            rf_bootstrap = rf
    for b in range(B):

        # Take a bootstrap sample from the training data
        # based on the specified user proportion
        if isinstance(rf, ClassifierMixin):
            X_train_rsmpl, y_rsmpl = resample(
                X_train, y_train, n_samples=n_samples, stratify = y_train)
        else:
            X_train_rsmpl, y_rsmpl = resample(
                X_train, y_train, n_samples=n_samples)

        # Set up the weighted random forest
        # Using the weight from the (K-1)th iteration i.e. RF(w(K))
        rf_bootstrap = clone(rf)
        
        # CHECK: different number of trees to fit for bootstrap samples
        rf_bootstrap.n_estimators=n_estimators_bootstrap

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
            y_test=y_test,
            signed=signed)

        # Update the rf bootstrap output dictionary
        all_rf_bootstrap_output['rf_bootstrap{}'.format(b)] = all_rf_tree_data

        # Run FP-Growth on interaction rule set
        all_FP_Growth_data = generate_all_samples(all_rf_tree_data, bin_class_type)

        spark = SparkSession \
                    .builder \
                    .appName("iterative Random Forests with FP-Growth") \
                    .getOrCreate()
    
        # Load all interactions into Spark dataframe
        input_list = [(i, all_FP_Growth_data[i].tolist()) for i in range(len(all_FP_Growth_data))]
        df = spark.createDataFrame(input_list, ["id", "items"])

        # Run FP-Growth on data
        fpGrowth = FPGrowth(itemsCol="items", minSupport=min_support, minConfidence=min_confidence)
        model = fpGrowth.fit(df)
        item_sets = model.freqItemsets.toPandas()

        # Update the rf_FP_Growth bootstrap output dictionary
        item_sets = item_sets.sort_values(by=["freq"], ascending=False)
        all_FP_Growth_bootstrap_output['rf_bootstrap{}'.format(
            b)] = item_sets

    stability_score = _FP_Growth_get_stability_score(
        all_FP_Growth_bootstrap_output=all_FP_Growth_bootstrap_output, bootstrap_num=bootstrap_num)

    return all_rf_weights,\
        all_K_iter_rf_data, all_rf_bootstrap_output,\
        all_FP_Growth_bootstrap_output, stability_score

def generate_all_samples(all_rf_tree_data, bin_class_type=1):
    n_estimators = all_rf_tree_data['rf_obj'].n_estimators

    all_paths = []
    for dtree in range(n_estimators):
        filtered = filter_leaves_classifier(
            dtree_data=all_rf_tree_data['dtree{}'.format(dtree)],
            bin_class_type=bin_class_type)
        all_paths.extend(filtered['uniq_feature_paths'])
    return all_paths

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
