# Copyright (C) 2022  Marcelo R. H. Maia <mmaia@ic.uff.br, marcelo.h.maia@ibge.gov.br>
# License: GPLv3 (https://www.gnu.org/licenses/gpl-3.0.html)


from warnings import catch_warnings, simplefilter, warn

import numpy as np
from joblib import Parallel
from pandas import DataFrame
from scipy.sparse import issparse
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier, _forest
from sklearn.ensemble._forest import _get_n_samples_bootstrap, MAX_INT
from sklearn.exceptions import DataConversionWarning
from sklearn.tree._tree import DTYPE, DOUBLE
from sklearn.utils import compute_sample_weight, check_random_state
from sklearn.utils.fixes import _joblib_parallel_args, delayed
from sklearn.utils.validation import _check_sample_weight

from ud_tree_classes import DFEDecisionTreeClassifier


def generate_biased_sample_indices(random_state, n_samples, n_samples_bootstrap, sample_proba):
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.choice(range(n_samples), size=n_samples_bootstrap, p=sample_proba)

    return sample_indices


def compute_sample_bias(x, uncertain_features):
    bias_sum = 0.0
    sample_bias = np.empty((x.shape[0],), dtype=np.float64)
    for s in range(x.shape[0]):
        entropy_sum = 0.0
        missing_values = 0
        for f in range(x.shape[1]):
            if uncertain_features[f]:
                v = x.iloc[s, f] if isinstance(x, DataFrame) else x[s, f]
                if v > 0:
                    entropy_sum += entropy([v, 1.0 - v])
                else:
                    missing_values += 1
        if x.shape[1] == missing_values:
            sample_bias[s] = 0.0001
        else:
            sample_bias[s] = 1.0 - entropy_sum / (x.shape[1] - missing_values)
            sample_bias[s] *= (x.shape[1] - missing_values) / x.shape[1]    # Known values rate
        bias_sum += sample_bias[s]

    return sample_bias / bias_sum


def compute_feature_bias(x, uncertain_features, sample_indices=None):
    if sample_indices is None:
        sample_indices = range(x.shape[0])

    feature_bias = np.empty((x.shape[1],), dtype=np.float64)
    for f in range(x.shape[1]):
        entropy_sum = 0.0
        missing_values = 0
        if uncertain_features[f]:
            for s in sample_indices:
                v = x.iloc[s, f] if isinstance(x, DataFrame) else x[s, f]
                if v > 0:
                    entropy_sum += entropy([v, 1.0 - v])
                else:
                    missing_values += 1
        nb_indices = sample_indices.stop if isinstance(sample_indices, range) else sample_indices.shape[0]
        if nb_indices == missing_values:
            feature_bias[f] = 0.0001
        else:
            feature_bias[f] = 1.0 - entropy_sum / (nb_indices - missing_values)
            feature_bias[f] *= (nb_indices - missing_values) / nb_indices    # Known values rate

    return feature_bias


def parallel_build_trees(tree, forest, X, y, sample_weight, tree_idx, n_trees, verbose=0, class_weight=None,
                         n_samples_bootstrap=None, sample_bias=None, feature_bias=None):
    """
    Private function used to fit a single tree in parallel."""
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    if forest.bootstrap:
        n_samples = X.shape[0]
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
        else:
            curr_sample_weight = sample_weight.copy()

        if sample_bias is None:
            indices = _forest._generate_sample_indices(tree.random_state, n_samples, n_samples_bootstrap)
        else:
            indices = generate_biased_sample_indices(tree.random_state, n_samples, n_samples_bootstrap, sample_bias)

        sample_counts = np.bincount(indices, minlength=n_samples)
        curr_sample_weight *= sample_counts

        if class_weight == 'subsample':
            with catch_warnings():
                simplefilter('ignore', DeprecationWarning)
                curr_sample_weight *= compute_sample_weight('auto', y,
                                                            indices=indices)
        elif class_weight == 'balanced_subsample':
            curr_sample_weight *= compute_sample_weight('balanced', y,
                                                        indices=indices)

        if forest.biased_splitting:
            if feature_bias is None:
                feature_bias = compute_feature_bias(X, forest.uncertain_features, indices)
            tree.fit(X, y, sample_weight=curr_sample_weight, check_input=False,
                     feature_bias=feature_bias)
        else:
            tree.fit(X, y, sample_weight=curr_sample_weight, check_input=False)
    else:
        if forest.biased_splitting:
            if feature_bias is None:
                feature_bias = compute_feature_bias(X, forest.uncertain_features)
            tree.fit(X, y, sample_weight=sample_weight, check_input=False,
                     feature_bias=feature_bias)
        else:
            tree.fit(X, y, sample_weight=sample_weight, check_input=False)

    return tree


class DFERandomForestClassifier(RandomForestClassifier):
    def __init__(self,
                 n_estimators=100,
                 criterion='dfe_gini',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.01,
                 max_features='auto',
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 ccp_alpha=0.0,
                 max_samples=None,
                 uncertain_features=None,
                 biased_bootstrap=False,
                 biased_splitting=False,
                 missing_values=-1.0):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples)
        self.uncertain_features = uncertain_features
        self.biased_bootstrap = biased_bootstrap
        self.biased_splitting = biased_splitting
        self.missing_values = missing_values
        self.base_estimator = DFEDecisionTreeClassifier(uncertain_features=uncertain_features,
                                                        missing_values=missing_values)

    def fit(self, X, y, sample_weight=None):
        # Validate or convert input data
        if issparse(y):
            raise ValueError(
                "sparse multilabel-indicator for y is not supported."
            )
        X, y = self._validate_data(X, y, multi_output=True,
                                   accept_sparse="csc", dtype=DTYPE)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        # Remap output
        self.n_features_ = X.shape[1]

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn("A column-vector y was passed when a 1d array was"
                 " expected. Please change the shape of y to "
                 "(n_samples,), for example using ravel().",
                 DataConversionWarning, stacklevel=2)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Get bootstrap sample size
        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples=X.shape[0],
            max_samples=self.max_samples
        )

        # Check parameters
        self._validate_estimator()

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [self._make_estimator(append=False,
                                          random_state=random_state)
                     for i in range(n_more_estimators)]

            sample_bias = None
            if self.bootstrap and self.biased_bootstrap and self.uncertain_features is not None:
                sample_bias = compute_sample_bias(X, self.uncertain_features)

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                             **_joblib_parallel_args(prefer='threads'))(
                delayed(parallel_build_trees)(
                    t, self, X, y, sample_weight, i, len(trees),
                    verbose=self.verbose, class_weight=self.class_weight,
                    n_samples_bootstrap=n_samples_bootstrap, sample_bias=sample_bias)
                for i, t in enumerate(trees))

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score:
            self._set_oob_score(X, y)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self
