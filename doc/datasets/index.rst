.. _datasets:

=========================
Dataset loading utilities
=========================

.. currentmodule:: sklearn.datasets

The ``sklearn.datasets`` package embeds some small toy datasets
as introduced in the :ref:`Getting Started <loading_example_dataset>` section.

To evaluate the impact of the scale of the dataset (``n_samples`` and
``n_features``) while controlling the statistical properties of the data
(typically the correlation and informativeness of the features), it is
also possible to generate synthetic data.

This package also features helpers to fetch larger datasets commonly
used by the machine learning community to benchmark algorithm on data
that comes from the 'real world'.

General dataset API
===================

There are three distinct kinds of dataset interfaces for different types
of datasets.
The simplest one is the interface for sample images, which is described
below in the :ref:`sample_images` section.

The dataset generation functions and the svmlight loader share a simplistic
interface, returning a tuple ``(X, y)`` consisting of a ``n_samples`` *
``n_features`` numpy array ``X`` and an array of length ``n_samples``
containing the targets ``y``.

The toy datasets as well as the 'real world' datasets and the datasets
fetched from mldata.org have more sophisticated structure.
These functions return a dictionary-like object holding at least two items:
an array of shape ``n_samples`` * ``n_features`` with key ``data``
(except for 20newsgroups)
and a numpy array of length ``n_samples``, containing the target values,
with key ``target``.

The datasets also contain a description in ``DESCR`` and some contain
``feature_names`` and ``target_names``.
See the dataset descriptions below for details.


Toy datasets
============

scikit-learn comes with a few small standard datasets that do not
require to download any file from some external website.

.. autosummary::

   :toctree: ../modules/generated/
   :template: function.rst

   load_boston
   load_iris
   load_diabetes
   load_digits
   load_linnerud

These datasets are useful to quickly illustrate the behavior of the
various algorithms implemented in the scikit. They are however often too
small to be representative of real world machine learning tasks.

.. _sample_images:

Sample images
=============

The scikit also embed a couple of sample JPEG images published under Creative
Commons license by their authors. Those image can be useful to test algorithms
and pipeline on 2D data.

.. autosummary::

   :toctree: ../modules/generated/
   :template: function.rst

   load_sample_images
   load_sample_image

.. image:: ../auto_examples/cluster/images/plot_color_quantization_001.png
   :target: ../auto_examples/cluster/plot_color_quantization.html
   :scale: 30
   :align: right


.. warning::

  The default coding of images is based on the ``uint8`` dtype to
  spare memory.  Often machine learning algorithms work best if the
  input is converted to a floating point representation first.  Also,
  if you plan to use ``matplotlib.pyplpt.imshow`` don't forget to scale to the range
  0 - 1 as done in the following example.

.. topic:: Examples:

    * :ref:`example_cluster_plot_color_quantization.py`


.. _sample_generators:

Sample generators
=================

In addition, scikit-learn includes various random sample generators that
can be used to build artificial datasets of controlled size and complexity.

Generators for classification and clustering
--------------------------------------------

These generators produce a matrix of features and corresponding discrete
targets.

Single label
~~~~~~~~~~~~

Both :func:`make_blobs` and :func:`make_classification` create multiclass
datasets by allocating each class one or more normally-distributed clusters of
points.  :func:`make_blobs` provides greater control regarding the centers and
standard deviations of each cluster, and is used to demonstrate clustering.
:func:`make_classification` specialises in introducing noise by way of:
correlated, redundant and uninformative features; multiple Gaussian clusters
per class; and linear transformations of the feature space.

:func:`make_gaussian_quantiles` divides a single Gaussian cluster into
near-equal-size classes separated by concentric hyperspheres.
:func:`make_hastie_10_2` generates a similar binary, 10-dimensional problem.

.. image:: ../auto_examples/datasets/images/plot_random_dataset_001.png
   :target: ../auto_examples/datasets/plot_random_dataset.html
   :scale: 50
   :align: center

:func:`make_circles` and :func:`make_moons` generate 2d binary classification
datasets that are challenging to certain algorithms (e.g. centroid-based
clustering or linear classification), including optional Gaussian noise.
They are useful for visualisation. produces Gaussian
data with a spherical decision boundary for binary classification.

Multilabel
~~~~~~~~~~

:func:`make_multilabel_classification` generates random samples with multiple
labels, reflecting a bag of words drawn from a mixture of topics. The number of
topics for each document is drawn from a Poisson distribution, and the topics
themselves are drawn from a fixed random distribution. Similarly, the number of
words is drawn from Poisson, with words drawn from a multinomial, where each
topic defines a probability distribution over words. Simplifications with
respect to true bag-of-words mixtures include:

* Per-topic word distributions are independently drawn, where in reality all
  would be affected by a sparse base distribution, and would be correlated.
* For a document generated from multiple topics, all topics are weighted
  equally in generating its bag of words.
* Documents without labels words at random, rather than from a base
  distribution.

.. image:: ../auto_examples/datasets/images/plot_random_multilabel_dataset_001.png
   :target: ../auto_examples/datasets/plot_random_multilabel_dataset.html
   :scale: 50
   :align: center

Biclustering
~~~~~~~~~~~~

.. autosummary::

   :toctree: ../modules/generated/
   :template: function.rst

   make_biclusters
   make_checkerboard


Generators for regression
-------------------------

:func:`make_regression` produces regression targets as an optionally-sparse
random linear combination of random features, with noise. Its informative
features may be uncorrelated, or low rank (few features account for most of the
variance).

Other regression generators generate functions deterministically from
randomized features.  :func:`make_sparse_uncorrelated` produces a target as a
linear combination of four features with fixed coefficients.
Others encode explicitly non-linear relations:
:func:`make_friedman1` is related by polynomial and sine transforms;
:func:`make_friedman2` includes feature multiplication and reciprocation; and
:func:`make_friedman3` is similar with an arctan transformation on the target.

Generators for manifold learning
--------------------------------

.. autosummary::

   :toctree: ../modules/generated/
   :template: function.rst

   make_s_curve
   make_swiss_roll

Generators for decomposition
----------------------------

.. autosummary::

   :toctree: ../modules/generated/
   :template: function.rst

   make_low_rank_matrix
   make_sparse_coded_signal
   make_spd_matrix
   make_sparse_spd_matrix


.. _libsvm_loader:

Datasets in svmlight / libsvm format
====================================

scikit-learn includes utility functions for loading
datasets in the svmlight / libsvm format. In this format, each line
takes the form ``<label> <feature-id>:<feature-value>
<feature-id>:<feature-value> ...``. This format is especially suitable for sparse datasets.
In this module, scipy sparse CSR matrices are used for ``X`` and numpy arrays are used for ``y``.

You may load a dataset like as follows::

  >>> from sklearn.datasets import load_svmlight_file
  >>> X_train, y_train = load_svmlight_file("/path/to/train_dataset.txt")
  ...                                                         # doctest: +SKIP

You may also load two (or more) datasets at once::

  >>> X_train, y_train, X_test, y_test = load_svmlight_files(
  ...     ("/path/to/train_dataset.txt", "/path/to/test_dataset.txt"))
  ...                                                         # doctest: +SKIP

In this case, ``X_train`` and ``X_test`` are guaranteed to have the same number
of features. Another way to achieve the same result is to fix the number of
features::

  >>> X_test, y_test = load_svmlight_file(
  ...     "/path/to/test_dataset.txt", n_features=X_train.shape[1])
  ...                                                         # doctest: +SKIP

.. topic:: Related links:

 _`Public datasets in svmlight / libsvm format`: http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/

 _`Faster API-compatible implementation`: https://github.com/mblondel/svmlight-loader


.. make sure everything is in a toc tree

.. toctree::
    :maxdepth: 2
    :hidden:

    olivetti_faces
    twenty_newsgroups
    mldata
    labeled_faces
    covtype
    rcv1


.. include:: olivetti_faces.rst

.. include:: twenty_newsgroups.rst

.. include:: mldata.rst

.. include:: labeled_faces.rst

.. include:: covtype.rst

.. include:: rcv1.rst

.. _boston_house_prices:

.. include:: ../../sklearn/datasets/descr/boston_house_prices.rst

.. _breast_cancer:

.. include:: ../../sklearn/datasets/descr/breast_cancer.rst

.. _diabetes:

.. include:: ../../sklearn/datasets/descr/diabetes.rst

.. _digits:

.. include:: ../../sklearn/datasets/descr/digits.rst

.. _iris:

.. include:: ../../sklearn/datasets/descr/iris.rst

.. _linnerud:

.. include:: ../../sklearn/datasets/descr/linnerud.rst
