Module crossense.ensemble.tests.test_bagging
============================================
Testing for the bagging ensemble module (crossense.ensemble.bagging).

Functions
---------

    
`replace(X)`
:   

    
`test_bagging_allow_nan_tag(bagging, expected_allow_nan)`
:   Check that bagging inherits allow_nan tag.

    
`test_bagging_get_estimators_indices()`
:   

    
`test_bagging_regressor_with_missing_inputs()`
:   

    
`test_bagging_sample_weight_unsupported_but_passed()`
:   

    
`test_bagging_with_pipeline()`
:   

    
`test_classification()`
:   

    
`test_error()`
:   

    
`test_estimator()`
:   

    
`test_estimators_samples()`
:   

    
`test_estimators_samples_deterministic()`
:   

    
`test_gridsearch()`
:   

    
`test_parallel_classification()`
:   

    
`test_parallel_regression()`
:   

    
`test_probability()`
:   

    
`test_regression()`
:   

Classes
-------

`DummySizeEstimator()`
:   Base class for all estimators in scikit-learn.
    
    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).

    ### Ancestors (in MRO)

    * sklearn.base.BaseEstimator
    * sklearn.utils._metadata_requests._MetadataRequester

    ### Methods

    `fit(self, X, y)`
    :

    `predict(self, X)`
    :

`DummyZeroEstimator()`
:   Base class for all estimators in scikit-learn.
    
    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).

    ### Ancestors (in MRO)

    * sklearn.base.BaseEstimator
    * sklearn.utils._metadata_requests._MetadataRequester

    ### Methods

    `fit(self, X, y)`
    :

    `predict(self, X)`
    :