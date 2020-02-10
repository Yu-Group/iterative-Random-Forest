#!/usr/bin/python

from __future__ import division

import matplotlib.pyplot as plt
import os
import yaml
import pydotplus
import pprint
import numpy as np
from . import tree
from sklearn.model_selection import train_test_split
from .ensemble import RandomForestClassifierWithWeights
from IPython.display import display, Image
from sklearn.datasets import load_breast_cancer


# CHECK: Ensure that the following list/emails is correct
# Authors:
#
#
#
# CHECK: License is correct
# License: BSD 3 clause

# =============================================================================
# Generate sample Random Forest Data
# =============================================================================


def generate_rf_example(sklearn_ds=load_breast_cancer(),
                        train_split_propn=0.9,
                        n_estimators=3,
                        feature_weight=None,
                        random_state_split=2017,
                        random_state_classifier=2018):
    """
    This fits a random forest classifier to the breast cancer/ iris datasets
    This can be called from the jupyter notebook so that analysis
    can take place quickly

    Parameters
    ----------
    sklearn_ds : sklearn dataset
        Choose from the `load_breast_cancer` or the `load_iris datasets`
        functions from the `sklearn.datasets` module

    train_split_propn : float
        Should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split.

    n_estimators : int, optional (default=10)
        The index of the root node of the tree. Should be set as default to
        3 and not changed by the user

    feature_weight : list, optional (default=None)
        The chance of splitting at each feature.

    random_state_split: int (default=2017)
        The seed used by the random number generator for the `train_test_split`
        function in creating our training and validation sets

    random_state_classifier: int (default=2018)
        The seed used by the random number generator for
        the `RandomForestClassifierWithWeights` function in fitting the random forest

    Returns
    -------
    X_train : array-like or sparse matrix, shape = [n_samples, n_features]
        Training features vector, where n_samples in the number of samples and
        n_features is the number of features.
    X_test : array-like or sparse matrix, shape = [n_samples, n_features]
        Test (validation) features vector, where n_samples in the
        number of samples and n_features is the number of features.
    y_train : array-like or sparse matrix, shape = [n_samples, n_classes]
        Training labels vector, where n_samples in the number of samples and
        n_classes is the number of classes.
    y_test : array-like or sparse matrix, shape = [n_samples, n_classes]
        Test (validation) labels vector, where n_samples in the
        number of samples and n_classes is the number of classes.
    rf : RandomForestClassifierWithWeights object
        The fitted random forest to the training data

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> X_train, X_test, y_train, y_test,
        rf = generate_rf_example(sklearn_ds =
                                load_breast_cancer())
    >>> print(X_train.shape)
    ...                             # doctest: +SKIP
    ...
    (512, 30)
    """

    # Load the relevant scikit learn data
    raw_data = sklearn_ds

    # Create the train-test datasets
    X_train, X_test, y_train, y_test = train_test_split(
        raw_data.data, raw_data.target, train_size=train_split_propn,
        random_state=random_state_split)

    # Just fit a simple random forest classifier with 2 decision trees
    rf = RandomForestClassifierWithWeights(
        n_estimators=n_estimators, random_state=random_state_classifier)

    # fit the classifier
    if feature_weight is None:
        rf.fit(X=X_train, y=y_train)
    else:
        rf.fit(X=X_train, y=y_train, feature_weight=feature_weight)

    return X_train, X_test, y_train, y_test, rf


# =============================================================================
# Draw a single random forest decision tree in jupyter
# =============================================================================


def draw_tree(decision_tree, out_file=None, filled=True, rounded=False,
              special_characters=True, node_ids=True, max_depth=None,
              feature_names=None, class_names=None, label='all',
              leaves_parallel=False, impurity=True, proportion=False,
              rotate=False):
    """
    A wrapper for the `export_graphviz` function in scikit learn

    Used to visually display the decision tree in the jupyter notebook
    This is useful for validation purposes of the key metrics collected
    from the decision tree object

    Parameters
    ----------

    decision_tree : decision tree classifier
        The decision tree to be exported to GraphViz.

    out_file : file object or string, optional (default='tree.dot')
        Handle or name of the output file. If ``None``, the result is
        returned as a string. This will the default from version 0.20.

    max_depth : int, optional (default=None)
        The maximum depth of the representation. If None, the tree is fully
        generated.

    feature_names : list of strings, optional (default=None)
        Names of each of the features.

    class_names : list of strings, bool or None, optional (default=None)
        Names of each of the target classes in ascending numerical order.
        Only relevant for classification and not supported for multi-output.
        If ``True``, shows a symbolic representation of the class name.

    label : {'all', 'root', 'none'}, optional (default='all')
        Whether to show informative labels for impurity, etc.
        Options include 'all' to show at every node, 'root' to show only at
        the top root node, or 'none' to not show at any node.

    filled : bool, optional (default=False)
        When set to ``True``, paint nodes to indicate majority class for
        classification, extremity of values for regression, or purity of node
        for multi-output.

    leaves_parallel : bool, optional (default=False)
        When set to ``True``, draw all leaf nodes at the bottom of the tree.

    impurity : bool, optional (default=True)
        When set to ``True``, show the impurity at each node.

    node_ids : bool, optional (default=False)
        When set to ``True``, show the ID number on each node.

    proportion : bool, optional (default=False)
        When set to ``True``, change the display of 'values' and/or 'samples'
        to be proportions and percentages respectively.

    rotate : bool, optional (default=False)
        When set to ``True``, orient tree left to right rather than top-down.

    rounded : bool, optional (default=False)
        When set to ``True``, draw node boxes with rounded corners and use
        Helvetica fonts instead of Times-Roman.

    special_characters : bool, optional (default=False)
        When set to ``False``, ignore special characters for PostScript
        compatibility.

    Returns
    -------
    A displayed png image of the decision tree based on
    the specified display options. This is intended to be run
    inside a jupyter notebook.

    """

    dot_data = tree.export_graphviz(decision_tree=decision_tree,
                                    out_file=out_file, filled=filled,
                                    rounded=rounded,
                                    special_characters=special_characters,
                                    node_ids=node_ids,
                                    max_depth=max_depth,
                                    feature_names=feature_names,
                                    class_names=class_names, label=label,
                                    leaves_parallel=leaves_parallel,
                                    impurity=impurity,
                                    proportion=proportion, rotate=rotate)
    graph = pydotplus.graph_from_dot_data(dot_data)
    img = Image(graph.create_png())
    display(img)


# =============================================================================
# Histogram and Plotting functions for Random Intersection Trees (RITs)
# and Decision Trees
# =============================================================================


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


# =============================================================================
# Read in yaml file as a Python dictionary
# =============================================================================


def yaml_to_dict(inp_yaml):
    """ Helper function to read in a yaml file into
        Python as a dictionary

    Parameters
    ----------
    inp_yaml : str
        A yaml text string containing to be parsed into a Python
        dictionary

    Returns
    -------
    out : dict
        The input yaml string parsed as a Python dictionary object
    """
    with open(inp_yaml, 'r') as stream:
        try:
            out = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return out

# =============================================================================
# Convert Python dictionary to yaml
# =============================================================================


def dict_to_yaml(inp_dict, out_yaml_dir, out_yaml_name):
    """ Helper function to convert Python dictionary
        into a yaml string file

    Parameters
    ----------
    inp_dict: dict
        The Python dictionary object to be output as a yaml file

    out_yaml_dir : str
        The output directory for yaml file created

    out_yaml_name : str
        The output filename for yaml file created
        e.g. for 'test.yaml' just set this value to 'test'
             the '.yaml' will be added by the function

    Returns
    -------
    out : str
        The yaml file with specified name and directory from
        the input Python dictionary
    """
    if not os.path.exists(out_yaml_dir):
        os.makedirs(out_yaml_dir)

    out_yaml_path = os.path.join(out_yaml_dir,
                                 out_yaml_name) + '.yaml'

    # Write out the yaml file to the specified path
    with open(out_yaml_path, 'w') as outfile:
        yaml.dump(inp_dict, outfile, default_flow_style=False)


# =============================================================================
# Pretty Print Dictionary in jupyter notebook
# =============================================================================

def pretty_print_dict(inp_dict, indent_val=4):
    """
     This is used to pretty print the dictionary
     this is particularly useful for printing the dictionary of outputs
     from each decision tree

    Parameters
        ----------
        inp_dict : dictionary
        Any python dictionary to be displayed in a pretty format

    indent_val : int (default=4)
        Indented value of the pretty printed dictionary. Set to 4 spaces
        by default

    Returns
        -------
        A pretty printed dictionary output. This is best run inside a
        jupyter notebook.

    """
    pp = pprint.PrettyPrinter(indent=indent_val)
    pp.pprint(inp_dict)
