=============
API Reference
=============

This is the class and function reference of scikit-learn. Please refer to
the :ref:`full user guide <user_guide>` for further details, as the class and
function raw specifications may not be enough to give full guidelines on their
uses.


.. _base_ref:

:mod:`sklearn.base`: Base classes and utility functions
=======================================================

.. automodule:: sklearn.base
    :no-members:
    :no-inherited-members:

Base classes
------------
.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   base.BaseEstimator
   base.ClassifierMixin
   base.ClusterMixin
   base.RegressorMixin
   base.TransformerMixin

Functions
---------
.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   base.clone


.. _cluster_ref:

:mod:`sklearn.cluster`: Clustering
==================================

.. automodule:: sklearn.cluster
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`clustering` section for further details.

Classes
-------
.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   cluster.AffinityPropagation
   cluster.AgglomerativeClustering
   cluster.Birch
   cluster.DBSCAN
   cluster.FeatureAgglomeration
   cluster.KMeans
   cluster.MiniBatchKMeans
   cluster.MeanShift
   cluster.SpectralClustering

Functions
---------
.. autosummary::
   :toctree: generated/
   :template: function.rst

   cluster.estimate_bandwidth
   cluster.k_means
   cluster.ward_tree
   cluster.affinity_propagation
   cluster.dbscan
   cluster.mean_shift
   cluster.spectral_clustering

.. _bicluster_ref:

:mod:`sklearn.cluster.bicluster`: Biclustering
==============================================

.. automodule:: sklearn.cluster.bicluster
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`biclustering` section for further details.

Classes
-------
.. currentmodule:: sklearn.cluster.bicluster

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SpectralBiclustering
   SpectralCoclustering

.. _covariance_ref:

:mod:`sklearn.covariance`: Covariance Estimators
================================================

.. automodule:: sklearn.covariance
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`covariance` section for further details.

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   covariance.EmpiricalCovariance
   covariance.EllipticEnvelope
   covariance.GraphLasso
   covariance.GraphLassoCV
   covariance.LedoitWolf
   covariance.MinCovDet
   covariance.OAS
   covariance.ShrunkCovariance

.. autosummary::
   :toctree: generated/
   :template: function.rst

   covariance.empirical_covariance
   covariance.ledoit_wolf
   covariance.shrunk_covariance
   covariance.oas
   covariance.graph_lasso


:mod:`sklearn.model_selection`: Model Selection
===============================================

.. automodule:: sklearn.model_selection
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`cross_validation`, :ref:`grid_search` and
:ref:`learning_curve` sections for further details.

Splitter Classes
----------------

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   model_selection.KFold
   model_selection.LabelKFold
   model_selection.StratifiedKFold
   model_selection.LeaveOneLabelOut
   model_selection.LeavePLabelOut
   model_selection.LeaveOneOut
   model_selection.LeavePOut
   model_selection.ShuffleSplit
   model_selection.LabelShuffleSplit
   model_selection.StratifiedShuffleSplit
   model_selection.PredefinedSplit

Splitter Functions
------------------

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   model_selection.train_test_split

Hyper-parameter optimizers
--------------------------

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   model_selection.GridSearchCV
   model_selection.RandomizedSearchCV
   model_selection.ParameterGrid
   model_selection.ParameterSampler

Model validation
----------------

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   model_selection.cross_val_score
   model_selection.cross_val_predict
   model_selection.permutation_test_score
   model_selection.learning_curve
   model_selection.validation_curve

.. _datasets_ref:

:mod:`sklearn.datasets`: Datasets
=================================

.. automodule:: sklearn.datasets
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`datasets` section for further details.

Loaders
-------

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   datasets.clear_data_home
   datasets.get_data_home
   datasets.fetch_20newsgroups
   datasets.fetch_20newsgroups_vectorized
   datasets.load_boston
   datasets.load_diabetes
   datasets.load_digits
   datasets.load_files
   datasets.load_iris
   datasets.fetch_lfw_pairs
   datasets.fetch_lfw_people
   datasets.load_linnerud
   datasets.mldata_filename
   datasets.fetch_mldata
   datasets.fetch_olivetti_faces
   datasets.fetch_california_housing
   datasets.fetch_covtype
   datasets.fetch_kddcup99
   datasets.fetch_rcv1
   datasets.load_mlcomp
   datasets.load_sample_image
   datasets.load_sample_images
   datasets.load_svmlight_file
   datasets.load_svmlight_files
   datasets.dump_svmlight_file

Samples generator
-----------------

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   datasets.make_blobs
   datasets.make_classification
   datasets.make_circles
   datasets.make_friedman1
   datasets.make_friedman2
   datasets.make_friedman3
   datasets.make_gaussian_quantiles
   datasets.make_hastie_10_2
   datasets.make_low_rank_matrix
   datasets.make_moons
   datasets.make_multilabel_classification
   datasets.make_regression
   datasets.make_s_curve
   datasets.make_sparse_coded_signal
   datasets.make_sparse_spd_matrix
   datasets.make_sparse_uncorrelated
   datasets.make_spd_matrix
   datasets.make_swiss_roll
   datasets.make_biclusters
   datasets.make_checkerboard


.. _decomposition_ref:

:mod:`sklearn.decomposition`: Matrix Decomposition
==================================================

.. automodule:: sklearn.decomposition
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`decompositions` section for further details.

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   decomposition.PCA
   decomposition.IncrementalPCA
   decomposition.ProjectedGradientNMF
   decomposition.RandomizedPCA
   decomposition.KernelPCA
   decomposition.FactorAnalysis
   decomposition.FastICA
   decomposition.TruncatedSVD
   decomposition.NMF
   decomposition.SparsePCA
   decomposition.MiniBatchSparsePCA
   decomposition.SparseCoder
   decomposition.DictionaryLearning
   decomposition.MiniBatchDictionaryLearning
   decomposition.LatentDirichletAllocation

.. autosummary::
   :toctree: generated/
   :template: function.rst

   decomposition.fastica
   decomposition.dict_learning
   decomposition.dict_learning_online
   decomposition.sparse_encode

.. _dummy_ref:

:mod:`sklearn.dummy`: Dummy estimators
======================================

.. automodule:: sklearn.dummy
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`model_evaluation` section for further details.

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   dummy.DummyClassifier
   dummy.DummyRegressor

.. autosummary::
   :toctree: generated/
   :template: function.rst

.. _ensemble_ref:

:mod:`sklearn.ensemble`: Ensemble Methods
=========================================

.. automodule:: sklearn.ensemble
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`ensemble` section for further details.

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ensemble.AdaBoostClassifier
   ensemble.AdaBoostRegressor
   ensemble.BaggingClassifier
   ensemble.BaggingRegressor
   ensemble.ExtraTreesClassifier
   ensemble.ExtraTreesRegressor
   ensemble.GradientBoostingClassifier
   ensemble.GradientBoostingRegressor
   ensemble.IsolationForest
   ensemble.RandomForestClassifier
   ensemble.RandomTreesEmbedding
   ensemble.RandomForestRegressor
   ensemble.VotingClassifier

.. autosummary::
   :toctree: generated/
   :template: function.rst


partial dependence
------------------

.. automodule:: sklearn.ensemble.partial_dependence
   :no-members:
   :no-inherited-members:

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   ensemble.partial_dependence.partial_dependence
   ensemble.partial_dependence.plot_partial_dependence


.. _exceptions_ref:

:mod:`sklearn.exceptions`: Exceptions and warnings
==================================================

.. automodule:: sklearn.exceptions
   :no-members:
   :no-inherited-members:

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class_without_init.rst

   exceptions.NotFittedError
   exceptions.ChangedBehaviorWarning
   exceptions.ConvergenceWarning
   exceptions.DataConversionWarning
   exceptions.DataDimensionalityWarning
   exceptions.EfficiencyWarning
   exceptions.FitFailedWarning
   exceptions.NonBLASDotWarning
   exceptions.UndefinedMetricWarning

.. _feature_extraction_ref:

:mod:`sklearn.feature_extraction`: Feature Extraction
=====================================================

.. automodule:: sklearn.feature_extraction
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`feature_extraction` section for further details.

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   feature_extraction.DictVectorizer
   feature_extraction.FeatureHasher

From images
-----------

.. automodule:: sklearn.feature_extraction.image
   :no-members:
   :no-inherited-members:

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   feature_extraction.image.img_to_graph
   feature_extraction.image.grid_to_graph
   feature_extraction.image.extract_patches_2d
   feature_extraction.image.reconstruct_from_patches_2d

   :template: class.rst

   feature_extraction.image.PatchExtractor

.. _text_feature_extraction_ref:

From text
---------

.. automodule:: sklearn.feature_extraction.text
   :no-members:
   :no-inherited-members:

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   feature_extraction.text.CountVectorizer
   feature_extraction.text.HashingVectorizer
   feature_extraction.text.TfidfTransformer
   feature_extraction.text.TfidfVectorizer


.. _feature_selection_ref:

:mod:`sklearn.feature_selection`: Feature Selection
===================================================

.. automodule:: sklearn.feature_selection
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`feature_selection` section for further details.

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   feature_selection.GenericUnivariateSelect
   feature_selection.SelectPercentile
   feature_selection.SelectKBest
   feature_selection.SelectFpr
   feature_selection.SelectFdr
   feature_selection.SelectFromModel
   feature_selection.SelectFwe
   feature_selection.RFE
   feature_selection.RFECV
   feature_selection.VarianceThreshold

.. autosummary::
   :toctree: generated/
   :template: function.rst

   feature_selection.chi2
   feature_selection.f_classif
   feature_selection.f_regression
   feature_selection.mutual_info_classif
   feature_selection.mutual_info_regression


.. _gaussian_process_ref:

:mod:`sklearn.gaussian_process`: Gaussian Processes
===================================================

.. automodule:: sklearn.gaussian_process
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`gaussian_process` section for further details.

.. currentmodule:: sklearn

.. autosummary::
  :toctree: generated/
  :template: class.rst

  gaussian_process.GaussianProcessRegressor
  gaussian_process.GaussianProcessClassifier
  gaussian_process.GaussianProcess

Kernels:

.. autosummary::
  :toctree: generated/
  :template: class.rst

  gaussian_process.kernels.Kernel
  gaussian_process.kernels.Sum
  gaussian_process.kernels.Product
  gaussian_process.kernels.Exponentiation
  gaussian_process.kernels.ConstantKernel
  gaussian_process.kernels.WhiteKernel
  gaussian_process.kernels.RBF
  gaussian_process.kernels.Matern
  gaussian_process.kernels.RationalQuadratic
  gaussian_process.kernels.ExpSineSquared
  gaussian_process.kernels.DotProduct
  gaussian_process.kernels.PairwiseKernel
  gaussian_process.kernels.CompoundKernel
  gaussian_process.kernels.Hyperparameter

.. _isotonic_ref:

:mod:`sklearn.isotonic`: Isotonic regression
============================================

.. automodule:: sklearn.isotonic
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`isotonic` section for further details.

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   isotonic.IsotonicRegression

.. autosummary::
   :toctree: generated
   :template: function.rst

   isotonic.isotonic_regression
   isotonic.check_increasing

.. _kernel_approximation_ref:

:mod:`sklearn.kernel_approximation` Kernel Approximation
========================================================

.. automodule:: sklearn.kernel_approximation
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`kernel_approximation` section for further details.

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   kernel_approximation.AdditiveChi2Sampler
   kernel_approximation.Nystroem
   kernel_approximation.RBFSampler
   kernel_approximation.SkewedChi2Sampler

.. _kernel_ridge_ref:

:mod:`sklearn.kernel_ridge` Kernel Ridge Regression
========================================================

.. automodule:: sklearn.kernel_ridge
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`kernel_ridge` section for further details.

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   kernel_ridge.KernelRidge

.. _lda_ref:

:mod:`sklearn.discriminant_analysis`: Discriminant Analysis
===========================================================

.. automodule:: sklearn.discriminant_analysis
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`lda_qda` section for further details.

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated
   :template: class.rst

   discriminant_analysis.LinearDiscriminantAnalysis
   discriminant_analysis.QuadraticDiscriminantAnalysis


.. _linear_model_ref:

:mod:`sklearn.linear_model`: Generalized Linear Models
======================================================

.. automodule:: sklearn.linear_model
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`linear_model` section for further details.

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   linear_model.ARDRegression
   linear_model.BayesianRidge
   linear_model.ElasticNet
   linear_model.ElasticNetCV
   linear_model.HuberRegressor
   linear_model.Lars
   linear_model.LarsCV
   linear_model.Lasso
   linear_model.LassoCV
   linear_model.LassoLars
   linear_model.LassoLarsCV
   linear_model.LassoLarsIC
   linear_model.LinearRegression
   linear_model.LogisticRegression
   linear_model.LogisticRegressionCV
   linear_model.MultiTaskLasso
   linear_model.MultiTaskElasticNet
   linear_model.MultiTaskLassoCV
   linear_model.MultiTaskElasticNetCV
   linear_model.OrthogonalMatchingPursuit
   linear_model.OrthogonalMatchingPursuitCV
   linear_model.PassiveAggressiveClassifier
   linear_model.PassiveAggressiveRegressor
   linear_model.Perceptron
   linear_model.RandomizedLasso
   linear_model.RandomizedLogisticRegression
   linear_model.RANSACRegressor
   linear_model.Ridge
   linear_model.RidgeClassifier
   linear_model.RidgeClassifierCV
   linear_model.RidgeCV
   linear_model.SGDClassifier
   linear_model.SGDRegressor
   linear_model.TheilSenRegressor

.. autosummary::
   :toctree: generated/
   :template: function.rst

   linear_model.lars_path
   linear_model.lasso_path
   linear_model.lasso_stability_path
   linear_model.orthogonal_mp
   linear_model.orthogonal_mp_gram


.. _manifold_ref:

:mod:`sklearn.manifold`: Manifold Learning
==========================================

.. automodule:: sklearn.manifold
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`manifold` section for further details.

.. currentmodule:: sklearn

.. autosummary::
    :toctree: generated
    :template: class.rst

    manifold.LocallyLinearEmbedding
    manifold.Isomap
    manifold.MDS
    manifold.SpectralEmbedding
    manifold.TSNE

.. autosummary::
    :toctree: generated
    :template: function.rst

    manifold.locally_linear_embedding
    manifold.spectral_embedding


.. _metrics_ref:

:mod:`sklearn.metrics`: Metrics
===============================

See the :ref:`model_evaluation` section and the :ref:`metrics` section of the
user guide for further details.

.. automodule:: sklearn.metrics
   :no-members:
   :no-inherited-members:

.. currentmodule:: sklearn

Model Selection Interface
-------------------------
See the :ref:`scoring_parameter` section of the user guide for further
details.

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.make_scorer
   metrics.get_scorer

Classification metrics
----------------------

See the :ref:`classification_metrics` section of the user guide for further
details.

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.accuracy_score
   metrics.auc
   metrics.average_precision_score
   metrics.brier_score_loss
   metrics.classification_report
   metrics.cohen_kappa_score
   metrics.confusion_matrix
   metrics.f1_score
   metrics.fbeta_score
   metrics.hamming_loss
   metrics.hinge_loss
   metrics.jaccard_similarity_score
   metrics.log_loss
   metrics.matthews_corrcoef
   metrics.precision_recall_curve
   metrics.precision_recall_fscore_support
   metrics.precision_score
   metrics.recall_score
   metrics.roc_auc_score
   metrics.roc_curve
   metrics.zero_one_loss

Regression metrics
------------------

See the :ref:`regression_metrics` section of the user guide for further
details.

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.explained_variance_score
   metrics.mean_absolute_error
   metrics.mean_squared_error
   metrics.median_absolute_error
   metrics.r2_score

Multilabel ranking metrics
--------------------------
See the :ref:`multilabel_ranking_metrics` section of the user guide for further
details.

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.coverage_error
   metrics.label_ranking_average_precision_score
   metrics.label_ranking_loss


Clustering metrics
------------------

See the :ref:`clustering_evaluation` section of the user guide for further
details.

.. automodule:: sklearn.metrics.cluster
   :no-members:
   :no-inherited-members:

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.adjusted_mutual_info_score
   metrics.adjusted_rand_score
   metrics.calinski_harabaz_score
   metrics.completeness_score
   metrics.fowlkes_mallows_score
   metrics.homogeneity_completeness_v_measure
   metrics.homogeneity_score
   metrics.mutual_info_score
   metrics.normalized_mutual_info_score
   metrics.silhouette_score
   metrics.silhouette_samples
   metrics.v_measure_score

Biclustering metrics
--------------------

See the :ref:`biclustering_evaluation` section of the user guide for
further details.

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.consensus_score


Pairwise metrics
----------------

See the :ref:`metrics` section of the user guide for further details.

.. automodule:: sklearn.metrics.pairwise
   :no-members:
   :no-inherited-members:

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.pairwise.additive_chi2_kernel
   metrics.pairwise.chi2_kernel
   metrics.pairwise.distance_metrics
   metrics.pairwise.euclidean_distances
   metrics.pairwise.kernel_metrics
   metrics.pairwise.linear_kernel
   metrics.pairwise.manhattan_distances
   metrics.pairwise.pairwise_distances
   metrics.pairwise.pairwise_kernels
   metrics.pairwise.polynomial_kernel
   metrics.pairwise.rbf_kernel
   metrics.pairwise.sigmoid_kernel
   metrics.pairwise.cosine_similarity
   metrics.pairwise.cosine_distances
   metrics.pairwise.laplacian_kernel
   metrics.pairwise_distances
   metrics.pairwise_distances_argmin
   metrics.pairwise_distances_argmin_min
   metrics.pairwise.paired_euclidean_distances
   metrics.pairwise.paired_manhattan_distances
   metrics.pairwise.paired_cosine_distances
   metrics.pairwise.paired_distances


.. _mixture_ref:

:mod:`sklearn.mixture`: Gaussian Mixture Models
===============================================

.. automodule:: sklearn.mixture
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`mixture` section for further details.

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mixture.GaussianMixture
   mixture.DPGMM
   mixture.VBGMM


.. _multiclass_ref:

:mod:`sklearn.multiclass`: Multiclass and multilabel classification
===================================================================

.. automodule:: sklearn.multiclass
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`multiclass` section for further details.

.. currentmodule:: sklearn

.. autosummary::
    :toctree: generated
    :template: class.rst

    multiclass.OneVsRestClassifier
    multiclass.OneVsOneClassifier
    multiclass.OutputCodeClassifier

.. _multioutput_ref:

:mod:`sklearn.multioutput`: Multioutput regression and classification
=====================================================================

.. automodule:: sklearn.multioutput
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`multiclass` section for further details.

.. currentmodule:: sklearn

.. autosummary::
    :toctree: generated
    :template: class.rst

    multioutput.MultiOutputRegressor
    multioutput.MultiOutputClassifier

.. _naive_bayes_ref:

:mod:`sklearn.naive_bayes`: Naive Bayes
=======================================

.. automodule:: sklearn.naive_bayes
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`naive_bayes` section for further details.

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   naive_bayes.GaussianNB
   naive_bayes.MultinomialNB
   naive_bayes.BernoulliNB


.. _neighbors_ref:

:mod:`sklearn.neighbors`: Nearest Neighbors
===========================================

.. automodule:: sklearn.neighbors
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`neighbors` section for further details.

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   neighbors.NearestNeighbors
   neighbors.KNeighborsClassifier
   neighbors.RadiusNeighborsClassifier
   neighbors.KNeighborsRegressor
   neighbors.RadiusNeighborsRegressor
   neighbors.NearestCentroid
   neighbors.BallTree
   neighbors.KDTree
   neighbors.LSHForest
   neighbors.DistanceMetric
   neighbors.KernelDensity

.. autosummary::
   :toctree: generated/
   :template: function.rst

   neighbors.kneighbors_graph
   neighbors.radius_neighbors_graph

.. _neural_network_ref:

:mod:`sklearn.neural_network`: Neural network models
=====================================================

.. automodule:: sklearn.neural_network
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`neural_networks_supervised` and :ref:`neural_networks_unsupervised` sections for further details.

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   neural_network.BernoulliRBM
   neural_network.MLPClassifier
   neural_network.MLPRegressor


.. _calibration_ref:

:mod:`sklearn.calibration`: Probability Calibration
===================================================

.. automodule:: sklearn.calibration
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`calibration` section for further details.

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   calibration.CalibratedClassifierCV


.. autosummary::
   :toctree: generated/
   :template: function.rst

   calibration.calibration_curve


.. _cross_decomposition_ref:

:mod:`sklearn.cross_decomposition`: Cross decomposition
=======================================================

.. automodule:: sklearn.cross_decomposition
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`cross_decomposition` section for further details.

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   cross_decomposition.PLSRegression
   cross_decomposition.PLSCanonical
   cross_decomposition.CCA
   cross_decomposition.PLSSVD


.. _pipeline_ref:

:mod:`sklearn.pipeline`: Pipeline
=================================

.. automodule:: sklearn.pipeline
   :no-members:
   :no-inherited-members:

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   pipeline.Pipeline
   pipeline.FeatureUnion

.. autosummary::
   :toctree: generated/
   :template: function.rst

   pipeline.make_pipeline
   pipeline.make_union


.. _preprocessing_ref:

:mod:`sklearn.preprocessing`: Preprocessing and Normalization
=============================================================

.. automodule:: sklearn.preprocessing
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`preprocessing` section for further details.

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   preprocessing.Binarizer
   preprocessing.FunctionTransformer
   preprocessing.Imputer
   preprocessing.KernelCenterer
   preprocessing.LabelBinarizer
   preprocessing.LabelEncoder
   preprocessing.MultiLabelBinarizer
   preprocessing.MaxAbsScaler
   preprocessing.MinMaxScaler
   preprocessing.Normalizer
   preprocessing.OneHotEncoder
   preprocessing.PolynomialFeatures
   preprocessing.RobustScaler
   preprocessing.StandardScaler

.. autosummary::
   :toctree: generated/
   :template: function.rst

   preprocessing.add_dummy_feature
   preprocessing.binarize
   preprocessing.label_binarize
   preprocessing.maxabs_scale
   preprocessing.minmax_scale
   preprocessing.normalize
   preprocessing.robust_scale
   preprocessing.scale


.. _random_projection_ref:

:mod:`sklearn.random_projection`: Random projection
===================================================

.. automodule:: sklearn.random_projection
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`random_projection` section for further details.

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   random_projection.GaussianRandomProjection
   random_projection.SparseRandomProjection

.. autosummary::
   :toctree: generated/
   :template: function.rst

   random_projection.johnson_lindenstrauss_min_dim


.. _semi_supervised_ref:

:mod:`sklearn.semi_supervised` Semi-Supervised Learning
========================================================

.. automodule:: sklearn.semi_supervised
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`semi_supervised` section for further details.

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   semi_supervised.LabelPropagation
   semi_supervised.LabelSpreading


.. _svm_ref:

:mod:`sklearn.svm`: Support Vector Machines
===========================================

.. automodule:: sklearn.svm
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`svm` section for further details.

Estimators
----------

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   svm.SVC
   svm.LinearSVC
   svm.NuSVC
   svm.SVR
   svm.LinearSVR
   svm.NuSVR
   svm.OneClassSVM

.. autosummary::
   :toctree: generated/
   :template: function.rst

   svm.l1_min_c

Low-level methods
-----------------

.. autosummary::
   :toctree: generated
   :template: function.rst

   svm.libsvm.fit
   svm.libsvm.decision_function
   svm.libsvm.predict
   svm.libsvm.predict_proba
   svm.libsvm.cross_validation


.. _tree_ref:

:mod:`sklearn.tree`: Decision Trees
===================================

.. automodule:: sklearn.tree
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`tree` section for further details.

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   tree.DecisionTreeClassifier
   tree.DecisionTreeRegressor
   tree.ExtraTreeClassifier
   tree.ExtraTreeRegressor

.. autosummary::
   :toctree: generated/
   :template: function.rst

   tree.export_graphviz


.. _utils_ref:

:mod:`sklearn.utils`: Utilities
===============================

.. automodule:: sklearn.utils
   :no-members:
   :no-inherited-members:

**Developer guide:** See the :ref:`developers-utils` page for further details.

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   utils.check_random_state
   utils.estimator_checks.check_estimator
   utils.resample
   utils.shuffle
