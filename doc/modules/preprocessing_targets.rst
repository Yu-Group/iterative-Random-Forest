
.. currentmodule:: sklearn.preprocessing

.. _preprocessing_targets:

==========================================
Transforming the prediction target (``y``)
==========================================

Label binarization
------------------

:class:`LabelBinarizer` is a utility class to help create a label indicator
matrix from a list of multi-class labels::

    >>> from sklearn import preprocessing
    >>> lb = preprocessing.LabelBinarizer()
    >>> lb.fit([1, 2, 6, 4, 2])
    LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
    >>> lb.classes_
    array([1, 2, 4, 6])
    >>> lb.transform([1, 6])
    array([[1, 0, 0, 0],
           [0, 0, 0, 1]])

For multiple labels per instance, use :class:`MultiLabelBinarizer`::

    >>> lb = preprocessing.MultiLabelBinarizer()
    >>> lb.fit_transform([(1, 2), (3,)])
    array([[1, 1, 0],
           [0, 0, 1]])
    >>> lb.classes_
    array([1, 2, 3])

Label encoding
--------------

:class:`LabelEncoder` is a utility class to help normalize labels such that
they contain only values between 0 and n_classes-1. This is sometimes useful
for writing efficient Cython routines. :class:`LabelEncoder` can be used as
follows::

    >>> from sklearn import preprocessing
    >>> le = preprocessing.LabelEncoder()
    >>> le.fit([1, 2, 2, 6])
    LabelEncoder()
    >>> le.classes_
    array([1, 2, 6])
    >>> le.transform([1, 1, 2, 6])
    array([0, 0, 1, 2])
    >>> le.inverse_transform([0, 0, 1, 2])
    array([1, 1, 2, 6])

It can also be used to transform non-numerical labels (as long as they are
hashable and comparable) to numerical labels::

    >>> le = preprocessing.LabelEncoder()
    >>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
    LabelEncoder()
    >>> list(le.classes_)
    ['amsterdam', 'paris', 'tokyo']
    >>> le.transform(["tokyo", "tokyo", "paris"])
    array([2, 2, 1])
    >>> list(le.inverse_transform([2, 2, 1]))
    ['tokyo', 'tokyo', 'paris']
