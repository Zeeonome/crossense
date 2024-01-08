Module crossense.ensemble
=========================
The :mod:`crossense.ensemble` module includes ensemble-based methods for
classification, regression and anomaly detection.

Sub-modules
-----------
* crossense.ensemble.tests

Classes
-------

`BaseCrossBagging(estimator=None, cv=5, *, n_jobs=None, verbose=0)`
:   Base class for cross-fold Bagging meta-estimator.
    
    Warning: This class should not be used directly. Use derived classes
    instead.

    ### Ancestors (in MRO)

    * sklearn.ensemble._base.BaseEnsemble
    * sklearn.base.MetaEstimatorMixin
    * sklearn.base.BaseEstimator
    * sklearn.utils._metadata_requests._MetadataRequester

    ### Descendants

    * crossense.ensemble._bagging.CrossBaggingClassifier
    * crossense.ensemble._bagging.CrossBaggingRegressor

    ### Methods

    `fit(self, X, y, sample_weight=None)`
    :   Build a Bagging ensemble of estimators from the training set (X, y).
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        
        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).
        
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.

    `set_fit_request(self: crossense.ensemble._bagging.BaseCrossBagging, *, sample_weight: Union[bool, ForwardRef(None), str] = '$UNCHANGED$') ‑> crossense.ensemble._bagging.BaseCrossBagging`
    :   Request metadata passed to the ``fit`` method.
        
        Note that this method is only relevant if
        ``enable_metadata_routing=True`` (see :func:`sklearn.set_config`).
        Please see :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.
        
        The options for each parameter are:
        
        - ``True``: metadata is requested, and passed to ``fit`` if provided. The request is ignored if metadata is not provided.
        
        - ``False``: metadata is not requested and the meta-estimator will not pass it to ``fit``.
        
        - ``None``: metadata is not requested, and the meta-estimator will raise an error if the user provides it.
        
        - ``str``: metadata should be passed to the meta-estimator with this given alias instead of the original name.
        
        The default (``sklearn.utils.metadata_routing.UNCHANGED``) retains the
        existing request. This allows you to change the request for some
        parameters and not others.
        
        .. versionadded:: 1.3
        
        .. note::
            This method is only relevant if this estimator is used as a
            sub-estimator of a meta-estimator, e.g. used inside a
            :class:`~sklearn.pipeline.Pipeline`. Otherwise it has no effect.
        
        Parameters
        ----------
        sample_weight : str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED
            Metadata routing for ``sample_weight`` parameter in ``fit``.
        
        Returns
        -------
        self : object
            The updated object.

    `set_params(self, **params)`
    :   Set the parameters of this estimator.
        
        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        
        Returns
        -------
        self : estimator instance
            Estimator instance.

`CrossBaggingClassifier(estimator: object = None, cv: Union[int, BaseCrossValidator, Iterable] = 5, *, n_jobs: Optional[int] = None, verbose=0)`
:   A cross-validation Bagging classifier.
    
    A Bagging classifier is an ensemble meta-estimator that fits base
    classifiers each on a fold of cross-validation generator
    
    Attributes
    ----------
    estimator_ : estimator
        The base estimator from which the ensemble is grown.
    
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    
    estimators_ : list of estimators
        The collection of fitted base estimators.
    
    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by an array of the indices selected.
    
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
    
    n_classes_ : int or list
        The number of classes.
    
    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> from crossense.ensemble import CrossBaggingClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = CrossBaggingClassifier(estimator=SVC(), cv=5).fit(X, y)
    >>> clf.predict([[0, 0, 0, 0]])
    array([1])
    
    Parameters
    ----------
    estimator:
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a
        :class:`~sklearn.tree.DecisionTreeClassifier`.
    
    cv:
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
    
        - `None`, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable that generates (train, test) splits as arrays of indices.
    
        For `int`/`None` inputs, if the estimator is a classifier and `y` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.
    
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
    
    n_jobs:
        The number of jobs to run in parallel for both :meth:`fit` and
        :meth:`predict`. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See :term:`Glossary <n_jobs>` for more details.
    
    verbose:
        Controls the verbosity when fitting and predicting.

    ### Ancestors (in MRO)

    * sklearn.base.ClassifierMixin
    * crossense.ensemble._bagging.BaseCrossBagging
    * sklearn.ensemble._base.BaseEnsemble
    * sklearn.base.MetaEstimatorMixin
    * sklearn.base.BaseEstimator
    * sklearn.utils._metadata_requests._MetadataRequester

    ### Methods

    `decision_function(self, X)`
    :   Average of the decision functions of the base classifiers.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        
        Returns
        -------
        score : ndarray of shape (n_samples, k)
            The decision function of the input samples. The columns correspond
            to the classes in sorted order, as they appear in the attribute
            ``classes_``. Regression and binary classification are special
            cases with ``k == 1``, otherwise ``k==n_classes``.

    `predict(self, X)`
    :   Predict class for X.
        
        The predicted class of an input sample is computed as the class with
        the highest mean predicted probability. If base estimators do not
        implement a ``predict_proba`` method, then it resorts to voting.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.

    `predict_all_proba(self, X)`
    :   Predict class probabilities of all models for X.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        
        Returns
        -------
        p : ndarray of shape (n_estimators, n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.

    `predict_log_proba(self, X)`
    :   Predict class log-probabilities for X.
        
        The predicted class log-probabilities of an input sample is computed as
        the log of the mean predicted class probabilities of the base
        estimators in the ensemble.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        
        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.

    `predict_proba(self, X)`
    :   Predict class probabilities for X.
        
        The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the base estimators in the
        ensemble. If base estimators do not implement a ``predict_proba``
        method, then it resorts to voting and the predicted class probabilities
        of an input sample represents the proportion of estimators predicting
        each class.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        
        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.

    `set_fit_request(self: crossense.ensemble._bagging.CrossBaggingClassifier, *, sample_weight: Union[bool, ForwardRef(None), str] = '$UNCHANGED$') ‑> crossense.ensemble._bagging.CrossBaggingClassifier`
    :   Request metadata passed to the ``fit`` method.
        
        Note that this method is only relevant if
        ``enable_metadata_routing=True`` (see :func:`sklearn.set_config`).
        Please see :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.
        
        The options for each parameter are:
        
        - ``True``: metadata is requested, and passed to ``fit`` if provided. The request is ignored if metadata is not provided.
        
        - ``False``: metadata is not requested and the meta-estimator will not pass it to ``fit``.
        
        - ``None``: metadata is not requested, and the meta-estimator will raise an error if the user provides it.
        
        - ``str``: metadata should be passed to the meta-estimator with this given alias instead of the original name.
        
        The default (``sklearn.utils.metadata_routing.UNCHANGED``) retains the
        existing request. This allows you to change the request for some
        parameters and not others.
        
        .. versionadded:: 1.3
        
        .. note::
            This method is only relevant if this estimator is used as a
            sub-estimator of a meta-estimator, e.g. used inside a
            :class:`~sklearn.pipeline.Pipeline`. Otherwise it has no effect.
        
        Parameters
        ----------
        sample_weight : str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED
            Metadata routing for ``sample_weight`` parameter in ``fit``.
        
        Returns
        -------
        self : object
            The updated object.

    `set_score_request(self: crossense.ensemble._bagging.CrossBaggingClassifier, *, sample_weight: Union[bool, ForwardRef(None), str] = '$UNCHANGED$') ‑> crossense.ensemble._bagging.CrossBaggingClassifier`
    :   Request metadata passed to the ``score`` method.
        
        Note that this method is only relevant if
        ``enable_metadata_routing=True`` (see :func:`sklearn.set_config`).
        Please see :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.
        
        The options for each parameter are:
        
        - ``True``: metadata is requested, and passed to ``score`` if provided. The request is ignored if metadata is not provided.
        
        - ``False``: metadata is not requested and the meta-estimator will not pass it to ``score``.
        
        - ``None``: metadata is not requested, and the meta-estimator will raise an error if the user provides it.
        
        - ``str``: metadata should be passed to the meta-estimator with this given alias instead of the original name.
        
        The default (``sklearn.utils.metadata_routing.UNCHANGED``) retains the
        existing request. This allows you to change the request for some
        parameters and not others.
        
        .. versionadded:: 1.3
        
        .. note::
            This method is only relevant if this estimator is used as a
            sub-estimator of a meta-estimator, e.g. used inside a
            :class:`~sklearn.pipeline.Pipeline`. Otherwise it has no effect.
        
        Parameters
        ----------
        sample_weight : str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED
            Metadata routing for ``sample_weight`` parameter in ``score``.
        
        Returns
        -------
        self : object
            The updated object.

`CrossBaggingRegressor(estimator: object = None, cv: Union[int, BaseCrossValidator, Iterable] = 5, *, n_jobs: Optional[int] = None, verbose=0)`
:   A cross-validation Bagging regressor.
    
    A Bagging regressor is an ensemble meta-estimator that fits base
    regressors each on a fold of cross-validation generator
    
    Attributes
    ----------
    estimator_ : estimator
        The base estimator from which the ensemble is grown.
    
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    
    estimators_ : list of estimators
        The collection of fitted sub-estimators.
    
    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by an array of the indices selected.
    
    Examples
    --------
    >>> from sklearn.svm import SVR
    >>> from crossense.ensemble import CrossBaggingRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=4,
    ...                        n_informative=2, n_targets=1,
    ...                        random_state=0, shuffle=False)
    >>> regr = CrossBaggingRegressor(estimator=SVR(), cv=5).fit(X, y)
    >>> regr.predict([[0, 0, 0, 0]])
    array([-2.8720...])
    
    Parameters
    ----------
    estimator:
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a
        :class:`~sklearn.tree.DecisionTreeClassifier`.
    
    cv:
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
    
        - `None`, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable that generates (train, test) splits as arrays of indices.
    
        For `int`/`None` inputs, if the estimator is a classifier and `y` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.
    
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
    
    n_jobs:
        The number of jobs to run in parallel for both :meth:`fit` and
        :meth:`predict`. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See :term:`Glossary <n_jobs>` for more details.
    
    verbose:
        Controls the verbosity when fitting and predicting.

    ### Ancestors (in MRO)

    * sklearn.base.RegressorMixin
    * crossense.ensemble._bagging.BaseCrossBagging
    * sklearn.ensemble._base.BaseEnsemble
    * sklearn.base.MetaEstimatorMixin
    * sklearn.base.BaseEstimator
    * sklearn.utils._metadata_requests._MetadataRequester

    ### Methods

    `predict(self, X)`
    :   Predict regression target for X.
        
        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the estimators in the ensemble.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.

    `predict_all(self, X)`
    :   Predict regression target of all models for X.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        
        Returns
        -------
        p : ndarray of shape (n_estimators, n_samples, )
            The predicted values.

    `set_fit_request(self: crossense.ensemble._bagging.CrossBaggingRegressor, *, sample_weight: Union[bool, ForwardRef(None), str] = '$UNCHANGED$') ‑> crossense.ensemble._bagging.CrossBaggingRegressor`
    :   Request metadata passed to the ``fit`` method.
        
        Note that this method is only relevant if
        ``enable_metadata_routing=True`` (see :func:`sklearn.set_config`).
        Please see :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.
        
        The options for each parameter are:
        
        - ``True``: metadata is requested, and passed to ``fit`` if provided. The request is ignored if metadata is not provided.
        
        - ``False``: metadata is not requested and the meta-estimator will not pass it to ``fit``.
        
        - ``None``: metadata is not requested, and the meta-estimator will raise an error if the user provides it.
        
        - ``str``: metadata should be passed to the meta-estimator with this given alias instead of the original name.
        
        The default (``sklearn.utils.metadata_routing.UNCHANGED``) retains the
        existing request. This allows you to change the request for some
        parameters and not others.
        
        .. versionadded:: 1.3
        
        .. note::
            This method is only relevant if this estimator is used as a
            sub-estimator of a meta-estimator, e.g. used inside a
            :class:`~sklearn.pipeline.Pipeline`. Otherwise it has no effect.
        
        Parameters
        ----------
        sample_weight : str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED
            Metadata routing for ``sample_weight`` parameter in ``fit``.
        
        Returns
        -------
        self : object
            The updated object.

    `set_score_request(self: crossense.ensemble._bagging.CrossBaggingRegressor, *, sample_weight: Union[bool, ForwardRef(None), str] = '$UNCHANGED$') ‑> crossense.ensemble._bagging.CrossBaggingRegressor`
    :   Request metadata passed to the ``score`` method.
        
        Note that this method is only relevant if
        ``enable_metadata_routing=True`` (see :func:`sklearn.set_config`).
        Please see :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.
        
        The options for each parameter are:
        
        - ``True``: metadata is requested, and passed to ``score`` if provided. The request is ignored if metadata is not provided.
        
        - ``False``: metadata is not requested and the meta-estimator will not pass it to ``score``.
        
        - ``None``: metadata is not requested, and the meta-estimator will raise an error if the user provides it.
        
        - ``str``: metadata should be passed to the meta-estimator with this given alias instead of the original name.
        
        The default (``sklearn.utils.metadata_routing.UNCHANGED``) retains the
        existing request. This allows you to change the request for some
        parameters and not others.
        
        .. versionadded:: 1.3
        
        .. note::
            This method is only relevant if this estimator is used as a
            sub-estimator of a meta-estimator, e.g. used inside a
            :class:`~sklearn.pipeline.Pipeline`. Otherwise it has no effect.
        
        Parameters
        ----------
        sample_weight : str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED
            Metadata routing for ``sample_weight`` parameter in ``score``.
        
        Returns
        -------
        self : object
            The updated object.