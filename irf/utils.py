import numpy as np
from sklearn import metrics
from functools import partial
from functools import reduce
from sklearn.tree import _tree
from sklearn.base import ClassifierMixin
import matplotlib.pyplot as plt
import pyfpgrowth
from collections import OrderedDict
from itertools import groupby
from operator import itemgetter


# Get all RF and decision tree data


def get_rf_tree_data(rf, X_train, X_test, y_test, signed=False):
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
                           "get_params": rf.get_params(),
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
                                  root_node_id=0,
                                  signed=signed)

        # Append output to our combined random forest outputs dict
        all_rf_tree_outputs["dtree{}".format(idx)] = dtree_out

    return all_rf_tree_outputs

def get_tree_data(X_train, X_test, y_test, dtree, root_node_id=0, signed=False):
    """
    This returns all of the required summary results from an
    individual decision tree

    Parameters
    ----------
    dtree : DecisionTreeClassifier object
        An individual decision tree classifier object generated from a
        fitted RandomForestClassifier object in scikit learn.

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

    signed : bool, optional (default=False)
        Indicates whether to use signed interactions or not

    Returns
    -------
    tree_data : dict
        Return a dictionary containing various tree metrics
    from the input fitted Classifier object

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> raw_data = load_breast_cancer()
    >>> X_train, X_test, y_train, y_test = train_test_split(
        raw_data.data, raw_data.target, train_size=0.9,
        random_state=2017)
    >>> rf = RandomForestClassifier(
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
    if not signed:
        all_leaf_node_paths = all_tree_paths(dtree=dtree,
                                             root_node_id=root_node_id)
    else:
        all_leaf_node_paths = all_tree_signed_paths(dtree=dtree,
                                                    root_node_id=root_node_id)

    # Get list of leaf nodes
    # In all paths it is the final node value
    if not signed:
        all_leaf_nodes = [path[-1] for path in all_leaf_node_paths]
    else:
        all_leaf_nodes = [path[-1][0] for path in all_leaf_node_paths]

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
    if not signed:
        all_leaf_paths_features = [node_features_idx[path[:-1]]
                                    for path in all_leaf_node_paths]
    else:
        all_leaf_paths_features = []
        for path in all_leaf_node_paths:
            temp = []
            all_but_last = path[:-1]
            for elem in all_but_last:
                temp.append((node_features_idx[elem[0]], elem[1]))
            all_leaf_paths_features += [temp]

    # Get the unique list of features along a path
    # NOTE: This removes the original ordering of the features along the path
    # The original ordering could be preserved using a special function but
    # will increase runtime
    if signed:
        new_all_leaf_paths_features = []
        for path in all_leaf_paths_features:
            new_path = []
            for elem in path:
                new_path.append(str(elem[0]) + elem[1])
            new_all_leaf_paths_features += [new_path]
        all_leaf_paths_features = new_all_leaf_paths_features

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
    is_classification = isinstance(inp_class_reg_obj, ClassifierMixin)
    # if the number of classes is not binary let the user know accordingly
    if is_classification and inp_class_reg_obj.n_classes_ != 2:
        raise ValueError("The number of classes for classification must \
        be binary, you currently have fit to {} \
        classes".format(inp_class_reg_obj.n_classes_))

    # Get the predicted values on the validation data
    y_pred = inp_class_reg_obj.predict(X=X_test)

    if is_classification:
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
    else:
        return {
            "mse_loss":metrics.mean_squared_error(y_true=y_true, y_pred=y_pred),
            "mae_loss":metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred),
        }


def compute_impurity_decrease(dtree):
    '''
    Compute the impurity decrease at each node
    '''
    impurity = dtree.tree_.impurity
    weight = dtree.tree_.n_node_samples
    weight = [x / weight[0] for x in weight] # normalize the weight
    impurity_decrease = []
    n_nodes = len(weight)
    for i in range(n_nodes):
        left_child = dtree.tree_.children_left[i]
        right_child = dtree.tree_.children_right[i]
        if left_child < 0 or right_child < 0:
            impurity_decrease.append(-1)
        else:
            curr_impurity = weight[i] * impurity[i]
            left_impurity = weight[left_child] * impurity[left_child]
            right_impurity = weight[right_child] * impurity[right_child]
            impurity_decrease.append(curr_impurity - left_impurity - right_impurity)
    return impurity_decrease

def visualize_impurity_decrease(dtree_or_rf, yscale='log', xscale='log', **kwargs):
    ''' 
    Visualize the impurity decrease at each node.
    
    Parameters
    ----------

    dtree_or_rf : random forest or decision tree

    yscale : str, log or linear, optional with default log
        The yscale parameter in the histogram plot.

    xscale : str, log or linear, optional with default log
        The xscale, if log, the x axis will be the log10 of impurity decrease

    kwargs : other parameters that go into the hist plot

    Returns
    -------

    None
    '''
    out = []
    if hasattr(dtree_or_rf, 'tree_'):
        impurity_decrease = compute_impurity_decrease(dtree_or_rf)
        out = out + [x for x in impurity_decrease if x >= 0]
    elif hasattr(dtree_or_rf, 'estimators_'):
        for tree in dtree_or_rf.estimators_:
            impurity_decrease = compute_impurity_decrease(tree)
            out = out + [x for x in impurity_decrease if x >= 0]
    else:
        print("cannot recognize the input")
    if xscale == 'log':
        out = [np.log10(x) for x in out if x > 0]
    elif xscale != 'linear':
        print("cannot recognize xscale (%s)".format(xscale)
            + ", only take log or linear. using linear.")
    plt.hist(out, **kwargs)
    plt.yscale(yscale)
    plt.show()

def get_prevalent_interactions(
        rf,
        impurity_decrease_threshold,
        min_support=10,
        weight_scheme="depth",
        signed=False,
        mask=None,
        adjust_for_weights=False,
    ):
    '''
    Compute the prevalent interactions and their prevalence
        First, we use FP growth to find a series of candidate interactions.
        Second, we compute the weighted prevalence of each candidate.
    
    Parameters
    ----------

    rf : the random forest model

    impurity_decrease_threshold : float, if a split results in a decrease
        smaller than this parameter, then it will not appear in the path.
        If it is unclear how to select this for a rf, use visualize_impurity
        _decrease function to look at the histogram of impurity decrease for
        all the splits.

    min_support : int, optional with default 10,
        the minimum number of paths a interaction must appear to be considered

    weight_scheme : str, ["depth", "samplesize"],
        how to compute the weight

    mask : dict, default None
        this stores the name of each feature. Features with the same name are
        treated as the same.

    adjust_for_weights : bool, default False,
        whether adjust for weights for fpgrowth. Since fpgrowth does not allow
        weights for each path, that created some difficulty in selecting a
        threshold for fpgrowth when trees are very imbalanced. This parameter
        helps alleviate that issue by adjusting the input to fpgrowth using
        weights.
    Returns
    -------

    prevalence : dictionary, key correspond to patterns and values correspond
        to their weights.
    '''
    feature_paths, weight = get_filtered_feature_paths(
        rf,
        impurity_decrease_threshold,
        signed=signed,
        weight_scheme=weight_scheme,
    )
    feature_paths = [list(path) for path in feature_paths]
    if mask is not None:
        if isinstance(feature_paths[0][0], tuple):
            ff = lambda x: list(set([(mask[elem[0]], elem[1]) for elem in x]))
        else:
            ff = lambda x: list(set([mask[elem] for elem in x]))
        feature_paths = [ff(x) for x in feature_paths]
        #def my_reduce(obj1, obj2):
        #    return (obj1[0],obj1[1] + obj2[1])
        #feature_paths = [reduce(my_reduce, group)
        #       for _, group in groupby(sorted(feature_paths), key=itemgetter(0))]
    if adjust_for_weights:
        resampled_paths = []
        for path, w in zip(feature_paths, weight):
            resampled_paths += [path] * np.random.poisson(len(feature_paths) * w)
        patterns = pyfpgrowth.find_frequent_patterns(resampled_paths, min_support)
    else:
        patterns = pyfpgrowth.find_frequent_patterns(feature_paths, min_support)
    #print(feature_paths)
    prevalence = {p:0 for p in patterns}
    for key in patterns:
        p = set(list(key))
        for path, w in zip(feature_paths, weight):
            if p.issubset(path):
                prevalence[key] += w
    prevalence = OrderedDict(
        sorted(prevalence.items(), key=lambda t: -t[1] ** (1/len(t[0]))),
    )

    return prevalence
        
from matplotlib.ticker import MaxNLocator
def visualize_prevalent_interactions(prevalence, **kwargs):
    orders = [len(x) for x in prevalence]
    log2_prevalence = [np.log(x) / np.log(2) for x in prevalence.values()]
    plt.scatter(orders, log2_prevalence, alpha=0.7)
    plt.plot([0, max(orders)+0.5], [0, -max(orders)-0.5])
    plt.xlim(0, max(orders)+0.5)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    if 'ylim' in kwargs:
        plt.ylim(kwargs['ylim'][0], kwargs['ylim'][1])
    else:
        plt.ylim(min(log2_prevalence)-0.5, 0)
    plt.ylabel('log2(prevalence)')
    plt.xlabel('order of the interactions')
    plt.show()
    
def get_filtered_feature_paths(dtree_or_rf, threshold, signed=False, 
                               weight_scheme='depth'):
    '''
    Get the set of feature paths and their weights filtered by
        the impurity decrease
    Input:
        weight : ['depth', 'samplesize', 'label']
    '''
    if hasattr(dtree_or_rf, 'tree_'):
        impurity_decrease = compute_impurity_decrease(dtree_or_rf)
        features = dtree_or_rf.tree_.feature
        filtered = [x > threshold for x in impurity_decrease]
        if signed:
            tree_paths = all_tree_signed_paths(dtree_or_rf)
            #print(tree_paths)
            feature_paths = []
            for path in tree_paths:
                tmp = [(features[x[0]],x[1]) for x in path if filtered[x[0]]]
                # remove features that appear twice
                cleaned = []
                cache = set()
                for k in tmp:
                    if k[0] not in cache:
                        cleaned.append(k)
                        cache.add(k[0])
                feature_paths.append(cleaned)
            if weight_scheme == 'depth':
                weight = [2 ** (1-len(path)) for path in tree_paths]
            elif weight_scheme == 'samplesize':
                samplesize_per_node = dtree_or_rf.tree_.weighted_n_node_samples
                weight = [samplesize_per_node[path[-1]] for path in tree_paths]
            elif weight_scheme == 'label':
                raise NotImplementedError("this has not been implemented yet.")
            else:
                raise ValueError("weight scheme is not allowed.")
        else:
            tree_paths = all_tree_paths(dtree_or_rf)
            feature_paths = []
            for path in tree_paths:
                feature_paths.append(list(set([features[x] for x in path if filtered[x]])))
            if weight_scheme == 'depth':
                weight = [2 ** (1-len(path)) for path in tree_paths]
            elif weight_scheme == 'samplesize':
                samplesize_per_node = dtree_or_rf.tree_.weighted_n_node_samples
                weight = [samplesize_per_node[path[-1]] for path in tree_paths]
            else:
                raise ValueError("weight scheme is not allowed.")
        # make sure the weight sums up to 1.
        total = sum(weight)
        weight = [w / total for w in weight]
        return feature_paths, weight
    elif hasattr(dtree_or_rf, 'estimators_'):
        all_fs = []
        all_ws = []
        for tree in dtree_or_rf.estimators_:
            feature_paths, weight = get_filtered_feature_paths(tree, threshold, signed, weight_scheme)
            all_fs += feature_paths
            all_ws += [w / dtree_or_rf.n_estimators for w in weight]
        return all_fs, all_ws
        
def all_tree_signed_paths(dtree, root_node_id=0):
    """
    Get all the individual tree signed paths from root node to the leaves
    for a decision tree classifier object [1]_.

    Parameters
    ----------
    dtree : DecisionTreeClassifier object
        An individual decision tree classifier object generated from a
        fitted RandomForestClassifier object in scikit learn.

    root_node_id : int, optional (default=0)
        The index of the root node of the tree. Should be set as default to
        0 and not changed by the user

    Returns
    -------
    paths : list of lists
        Return a list of lists like this [(feature index, 'L'/'R'),...]
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
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> raw_data = load_breast_cancer()
    >>> X_train, X_test, y_train, y_test = train_test_split(
        raw_data.data, raw_data.target, train_size=0.9,
        random_state=2017)
    >>> rf = RandomForestClassifier(
        n_estimators=3, random_state=random_state_classifier)
    >>> rf.fit(X=X_train, y=y_train)
    >>> estimator0 = rf.estimators_[0]
    >>> tree_dat0 = all_tree_signed_paths(dtree = estimator0,
                                   root_node_id = 0)
    >>> tree_dat0
    ...                             # doctest: +SKIP
    ...
    """
    #TODO: use the decision path function in sklearn to optimize the code
    
    # Use these lists to parse the tree structure
    children_left = dtree.tree_.children_left
    children_right = dtree.tree_.children_right

    if root_node_id is None:
        paths = []

    if root_node_id == _tree.TREE_LEAF:
        raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

    # if left/right is None we'll get empty list anyway
    feature_id = dtree.tree_.feature[root_node_id] 
    if children_left[root_node_id] != _tree.TREE_LEAF:
        paths_left = [[(root_node_id, 'L')] + l
                 for l in all_tree_signed_paths(dtree, children_left[root_node_id])]
        paths_right = [[(root_node_id, 'R')] + l
                 for l in all_tree_signed_paths(dtree, children_right[root_node_id])]
        paths = paths_left + paths_right
    else:
        paths = [[(root_node_id, )]]
    return paths
    
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
        paths = [[root_node_id]]
    return paths

