#!/usr/bin/python
from irf import irf_jupyter_utils
from irf import irf_utils
from sklearn.datasets import load_breast_cancer
import numpy as np
from functools import reduce

# Load the breast cancer dataset
breast_cancer = load_breast_cancer()

# Generate the training and test datasets
X_train, X_test, y_train, \
    y_test, rf = irf_jupyter_utils.generate_rf_example(
        sklearn_ds=breast_cancer, n_estimators=10)

# Get all of the random forest and decision tree data
all_rf_tree_data = irf_utils.get_rf_tree_data(rf=rf,
                                              X_train=X_train,
                                              X_test=X_test,
                                              y_test=y_test)

# Get the RIT data and produce RITs
np.random.seed(12)
gen_random_leaf_paths = irf_utils.generate_rit_samples(
    all_rf_tree_data=all_rf_tree_data,
    bin_class_type=1)

# Build single Random Intersection Tree
# This is not using noisy splits i.e. 5 splits per node
rit0 = irf_utils.build_tree(
    feature_paths=gen_random_leaf_paths,
    max_depth=3,
    noisy_split=False,
    num_splits=5)

# Build single Random Intersection T
# This is using noisy splits i.e. {5, 6} splits per node
rit1 = irf_utils.build_tree(
    max_depth=3,
    noisy_split=True,
    feature_paths=gen_random_leaf_paths,
    num_splits=5)

# Build single Random Intersection Tree of depth 1
# This is not using noisy splits i.e. 5 splits per node
# This should only have a single (root) node
rit2 = irf_utils.build_tree(
    max_depth=1,
    noisy_split=True,
    feature_paths=gen_random_leaf_paths,
    num_splits=5)

# Get the entire RIT data
np.random.seed(12)
all_rit_tree_data = irf_utils.get_rit_tree_data(
    all_rf_tree_data=all_rf_tree_data,
    bin_class_type=1,
    M=10,
    max_depth=3,
    noisy_split=False,
    num_splits=2)

# Manually construct an RIT example
# Get the unique feature paths where the leaf
# node predicted class is just 1
# We are just going to get it from the first decision tree
# for this test case
uniq_feature_paths \
    = all_rf_tree_data['dtree0']['all_uniq_leaf_paths_features']
leaf_node_classes \
    = all_rf_tree_data['dtree0']['all_leaf_node_classes']
ones_only \
    = [i for i, j in zip(uniq_feature_paths, leaf_node_classes)
       if j == 1]

# Manually extract the last seven values for our example
# Just pick the last seven cases
# we are going to manually construct
# We are going to build a BINARY RIT of depth 3
# i.e. max `2**3 -1 = 7` intersecting nodes
ones_only_seven = ones_only[-7:]

# Manually build the RIT
# Construct a binary version of the RIT manually!
node0 = ones_only_seven[0]
node1 = np.intersect1d(node0, ones_only_seven[1])
node2 = np.intersect1d(node1, ones_only_seven[2])
node3 = np.intersect1d(node1, ones_only_seven[3])
node4 = np.intersect1d(node0, ones_only_seven[4])
node5 = np.intersect1d(node4, ones_only_seven[5])
node6 = np.intersect1d(node4, ones_only_seven[6])

intersected_nodes_seven \
    = [node0, node1, node2, node3, node4, node5, node6]

leaf_nodes_seven = [node2, node3, node5, node6]

rit_output \
    = reduce(np.union1d, (node2, node3, node5, node6))

# Now we can create the RIT using our built irf_utils
# build the generator of 7 values
ones_only_seven_gen = (n for n in ones_only_seven)

# Build the binary RIT using our irf_utils
rit_man0 = irf_utils.build_tree(
    feature_paths=ones_only_seven_gen,
    max_depth=3,
    noisy_split=False,
    num_splits=2)

# Calculate the union values

# First on the manually constructed RIT
rit_union_output_manual \
    = reduce(np.union1d, (node2, node3, node5, node6))

# Lastly on the RIT constructed using a function
rit_man0_union_output \
    = reduce(np.union1d, [node[1]._val
                          for node in rit_man0.leaf_nodes()])

# Test the manually constructed binary RIT


def test_manual_binary_RIT():
    # Check all node values
    assert [node[1]._val.tolist()
            for node in rit_man0.traverse_depth_first()] \
        == [node.tolist()
            for node in intersected_nodes_seven]

    # Check all leaf node intersected values
    assert [node[1]._val.tolist()
            for node in rit_man0.leaf_nodes()] ==\
        [node.tolist() for node in leaf_nodes_seven]

    # Check the union value calculation
    assert rit_union_output_manual.tolist()\
        == rit_man0_union_output.tolist()

# Test that the train test observations sum to the
# total data set observations
test_manual_binary_RIT()

def test_generate_rf_example1():

    # Check train test feature split from `generate_rf_example`
    # against the original breast cancer dataset
    assert X_train.shape[0] + X_test.shape[0] \
        == breast_cancer.data.shape[0]

    assert X_train.shape[1] == breast_cancer.data.shape[1]

    assert X_test.shape[1] == breast_cancer.data.shape[1]

    # Check feature and outcome sizes
    assert X_train.shape[0] + X_test.shape[0] \
        == y_train.shape[0] + y_test.shape[0]

    # Test build RIT
test_generate_rf_example1()

def test_build_tree():
    assert(len(rit0) <= 1 + 5 + 5**2)
    assert(len(rit1) <= 1 + 6 + 6**2)
    assert(len(rit2) == 1)
test_build_tree()

def test_rf_output():
    leaf_node_path = [[0, 1, 2, 3, 4, 5],
                      [0, 1, 2, 3, 4, 6, 7, 8],
                      [0, 1, 2, 3, 4, 6, 7, 9, 10, 11],
                      [0, 1, 2, 3, 4, 6, 7, 9, 10, 12],
                      [0, 1, 2, 3, 4, 6, 7, 9, 13],
                      [0, 1, 2, 3, 4, 6, 14, 15],
                      [0, 1, 2, 3, 4, 6, 14, 16],
                      [0, 1, 2, 3, 17, 18],
                      [0, 1, 2, 3, 17, 19],
                      [0, 1, 2, 20, 21, 22, 23],
                      [0, 1, 2, 20, 21, 22, 24, 25],
                      [0, 1, 2, 20, 21, 22, 24, 26],
                      [0, 1, 2, 20, 21, 27],
                      [0, 1, 2, 20, 28],
                      [0, 1, 29, 30],
                      [0, 1, 29, 31],
                      [0, 32, 33, 34, 35],
                      [0, 32, 33, 34, 36],
                      [0, 32, 33, 37, 38],
                      [0, 32, 33, 37, 39],
                      [0, 32, 40]]

    leaf_node_samples = [114, 1, 3, 1, 67, 1, 1,
                         1, 3, 2, 3, 1, 3, 7, 2, 7, 1, 11, 3, 1, 91]

    leaf_node_values = [[0, 189],
                        [3, 0],
                        [0, 5],
                        [1, 0],
                        [0, 101],
                        [1, 0],
                        [0, 1],
                        [2, 0],
                        [0, 3],
                        [0, 2],
                        [5, 0],
                        [0, 1], [0, 7], [10, 0], [0, 3],
                        [12, 0], [0, 2],
                        [19, 0], [0, 7], [1, 0], [137, 0]]

    leaf_paths_features = [[20, 24, 27, 10, 0],
                           [20, 24, 27, 10, 0, 6, 0],
                           [20, 24, 27, 10, 0, 6, 0, 14, 20],
                           [20, 24, 27, 10, 0, 6, 0, 14, 20],
                           [20, 24, 27, 10, 0, 6, 0, 14],
                           [20, 24, 27, 10, 0, 6, 18],
                           [20, 24, 27, 10, 0, 6, 18],
                           [20, 24, 27, 10, 28],
                           [20, 24, 27, 10, 28],
                           [20, 24, 27, 21, 6, 6],
                           [20, 24, 27, 21, 6, 6, 12],
                           [20, 24, 27, 21, 6, 6, 12],
                           [20, 24, 27, 21, 6],
                           [20, 24, 27, 21],
                           [20, 24, 22], [20, 24, 22],
                           [20, 7, 17, 29],
                           [20, 7, 17, 29],
                           [20, 7, 17, 28],
                           [20, 7, 17, 28],
                           [20, 7]]

    node_depths = [5, 7, 9, 9, 8, 7, 7, 5, 5,
                   6, 7, 7, 5, 4, 3, 3, 4, 4, 4, 4, 2]

    assert(np.all(
        np.concatenate(all_rf_tree_data['dtree1']['all_leaf_node_paths']) ==
        np.concatenate(leaf_node_path)))
    assert(np.all(all_rf_tree_data['dtree1']
                  ['all_leaf_node_samples'] == leaf_node_samples))

    assert(np.all(np.concatenate(
        all_rf_tree_data['dtree1']['all_leaf_node_values'], axis=0) ==
                  leaf_node_values))

    assert(np.all(np.concatenate(
        all_rf_tree_data['dtree1']['all_leaf_paths_features']) ==
        np.concatenate(leaf_paths_features)))

    assert(
        np.all(node_depths == all_rf_tree_data['dtree1']['leaf_nodes_depths']))
test_rf_output()


# test RIT_interactions
def test_rit_interactions():
    all_rit_tree_data_test = {'rit0':
                              {'rit_intersected_values':
                               [np.array([1, 2, 3]),
                                np.array([1, 2])]},
                              'rit1':
                              {'rit_intersected_values':
                               [np.array([1, 2, 3, 4]),
                                np.array([1, 2, 3])]},
                              'rit2':
                              {'rit_intersected_values':
                               [np.array([1, 2]),
                                np.array([5, 6]), np.array([])]},
                              'rit3':
                              {'rit_intersected_values':
                               [np.array([1, 2, 3]),
                                np.array([1, 2, 3, 4])]},
                              'rit4':
                              {'rit_intersected_values':
                               [np.array([1, 2, 3]),
                                np.array([1, 2, 3])]}}

    output = irf_utils.rit_interactions(all_rit_tree_data_test)

    L1 = output.keys()

    L3 = ['1_2_3', '1_2', '1_2_3_4', '5_6']
    L4 = [5, 2, 2, 1]
    output_test = dict(zip(L3, L4))

    # check keys
    assert(len(L1) == len(L3) and sorted(L1) == sorted(L3))

    # check values
    for key in output.keys():
        assert(output[key] == output_test[key])
test_rit_interactions()
