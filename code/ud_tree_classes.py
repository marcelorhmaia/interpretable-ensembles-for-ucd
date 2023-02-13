# Copyright (C) 2022  Marcelo R. H. Maia <mmaia@ic.uff.br, marcelo.h.maia@ibge.gov.br>
# License: GPLv3 (https://www.gnu.org/licenses/gpl-3.0.html)


import numbers
import warnings
from math import ceil

import numpy as np
from scipy.sparse import issparse
from sklearn.base import is_classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._classes import DOUBLE, DTYPE, CRITERIA_CLF, CRITERIA_REG, SPARSE_SPLITTERS, DENSE_SPLITTERS
from sklearn.tree._criterion import Criterion
from sklearn.tree._splitter import Splitter
from sklearn.tree._tree import BestFirstTreeBuilder, Tree
from sklearn.utils import _deprecate_positional_args, check_random_state, compute_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _check_sample_weight, check_is_fitted

import ud_criterion
import ud_splitter
from ud_tree import DFEDepthFirstTreeBuilder, DFETree, UDDepthFirstTreeBuilder


class DFEDecisionTreeClassifier(DecisionTreeClassifier):
    @_deprecate_positional_args
    def __init__(self, *,
                 criterion="dfe_gini",
                 splitter="dfe_best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.01,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 class_weight=None,
                 ccp_alpha=0.0,
                 uncertain_features=None,
                 default_feature_uncertainty=False,
                 missing_values=-1.0):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            ccp_alpha=ccp_alpha)
        self.uncertain_features = uncertain_features
        self.default_feature_uncertainty = default_feature_uncertainty
        self.missing_values = missing_values
        CRITERIA_CLF["dfe_gini"] = ud_criterion.DFEGini
        DENSE_SPLITTERS["dfe_best"] = ud_splitter.DFEBestSplitter

    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted="deprecated", feature_bias=None):

        random_state = check_random_state(self.random_state)

        if self.ccp_alpha < 0.0:
            raise ValueError("ccp_alpha must be greater than or equal to 0")

        if check_input:
            # Need to validate separately here.
            # We can't pass multi_ouput=True because that would allow y to be
            # csr.
            check_X_params = dict(dtype=DTYPE, accept_sparse="csc")
            check_y_params = dict(ensure_2d=False, dtype=None)
            X, y = self._validate_data(X, y,
                                       validate_separately=(check_X_params,
                                                            check_y_params))
            if issparse(X):
                X.sort_indices()

                if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                    raise ValueError("No support for np.int64 index based "
                                     "sparse matrices")

            if self.criterion == "poisson":
                if np.any(y < 0):
                    raise ValueError("Some value(s) of y are negative which is"
                                     " not allowed for Poisson regression.")
                if np.sum(y) <= 0:
                    raise ValueError("Sum of y is not positive which is "
                                     "necessary for Poisson regression.")

        # Determine output settings
        n_samples, self.n_features_ = X.shape
        self.n_features_in_ = self.n_features_
        is_classification = is_classifier(self)

        y = np.atleast_1d(y)
        expanded_class_weight = None

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        if is_classification:
            check_classification_targets(y)
            y = np.copy(y)

            self.classes_ = []
            self.n_classes_ = []

            if self.class_weight is not None:
                y_original = np.copy(y)

            y_encoded = np.zeros(y.shape, dtype=int)
            for k in range(self.n_outputs_):
                classes_k, y_encoded[:, k] = np.unique(y[:, k],
                                                       return_inverse=True)
                self.classes_.append(classes_k)
                self.n_classes_.append(classes_k.shape[0])
            y = y_encoded

            if self.class_weight is not None:
                expanded_class_weight = compute_sample_weight(
                    self.class_weight, y_original)

            self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        # Check parameters
        max_depth = (np.iinfo(np.int32).max if self.max_depth is None
                     else self.max_depth)
        max_leaf_nodes = (-1 if self.max_leaf_nodes is None
                          else self.max_leaf_nodes)

        if isinstance(self.min_samples_leaf, numbers.Integral):
            if not 1 <= self.min_samples_leaf:
                raise ValueError("min_samples_leaf must be at least 1 "
                                 "or in (0, 0.5], got %s"
                                 % self.min_samples_leaf)
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            if not 0. < self.min_samples_leaf <= 0.5:
                raise ValueError("min_samples_leaf must be at least 1 "
                                 "or in (0, 0.5], got %s"
                                 % self.min_samples_leaf)
            min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))

        if isinstance(self.min_samples_split, numbers.Integral):
            if not 2 <= self.min_samples_split:
                raise ValueError("min_samples_split must be an integer "
                                 "greater than 1 or a float in (0.0, 1.0]; "
                                 "got the integer %s"
                                 % self.min_samples_split)
            min_samples_split = self.min_samples_split
        else:  # float
            if not 0. < self.min_samples_split <= 1.:
                raise ValueError("min_samples_split must be an integer "
                                 "greater than 1 or a float in (0.0, 1.0]; "
                                 "got the float %s"
                                 % self.min_samples_split)
            min_samples_split = int(ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                if is_classification:
                    max_features = max(1, int(np.sqrt(self.n_features_)))
                else:
                    max_features = self.n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError("Invalid value for max_features. "
                                 "Allowed string values are 'auto', "
                                 "'sqrt' or 'log2'.")
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1,
                                   int(self.max_features * self.n_features_))
            else:
                max_features = 0

        self.max_features_ = max_features

        if len(y) != n_samples:
            raise ValueError("Number of labels=%d does not match "
                             "number of samples=%d" % (len(y), n_samples))
        if not 0 <= self.min_weight_fraction_leaf <= 0.5:
            raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero. ")
        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")
        if not isinstance(max_leaf_nodes, numbers.Integral):
            raise ValueError("max_leaf_nodes must be integral number but was "
                             "%r" % max_leaf_nodes)
        if -1 < max_leaf_nodes < 2:
            raise ValueError(("max_leaf_nodes {0} must be either None "
                              "or larger than 1").format(max_leaf_nodes))

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Set min_weight_leaf from min_weight_fraction_leaf
        if sample_weight is None:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               n_samples)
        else:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               np.sum(sample_weight))

        min_impurity_split = self.min_impurity_split
        if min_impurity_split is not None:
            warnings.warn(
                "The min_impurity_split parameter is deprecated. Its default "
                "value has changed from 1e-7 to 0 in version 0.23, and it "
                "will be removed in 1.0 (renaming of 0.25). Use the "
                "min_impurity_decrease parameter instead.",
                FutureWarning
            )

            if min_impurity_split < 0.:
                raise ValueError("min_impurity_split must be greater than "
                                 "or equal to 0")
        else:
            min_impurity_split = 0

        if self.min_impurity_decrease < 0.:
            raise ValueError("min_impurity_decrease must be greater than "
                             "or equal to 0")

        # TODO: Remove in 1.1
        if X_idx_sorted != "deprecated":
            warnings.warn(
                "The parameter 'X_idx_sorted' is deprecated and has no "
                "effect. It will be removed in 1.1 (renaming of 0.26). You "
                "can suppress this warning by not passing any value to the "
                "'X_idx_sorted' parameter.",
                FutureWarning
            )

        # Build tree
        criterion = self.criterion
        if not isinstance(criterion, Criterion):
            if is_classification:
                criterion = CRITERIA_CLF[self.criterion](self.n_outputs_,
                                                         self.n_classes_)
            else:
                criterion = CRITERIA_REG[self.criterion](self.n_outputs_,
                                                         n_samples)

        SPLITTERS = SPARSE_SPLITTERS if issparse(X) else DENSE_SPLITTERS

        splitter = self.splitter
        if not isinstance(self.splitter, Splitter):
            splitter = SPLITTERS[self.splitter](criterion,
                                                self.max_features_,
                                                min_samples_leaf,
                                                min_weight_leaf,
                                                random_state)

        if is_classifier(self):
            self.tree_ = DFETree(self.n_features_,
                                 self.n_classes_, self.n_outputs_)
        else:
            self.tree_ = DFETree(self.n_features_,
                                 # TODO: tree should't need this in this case
                                 np.array([1] * self.n_outputs_, dtype=np.intp),
                                 self.n_outputs_)

        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        if max_leaf_nodes < 0:
            builder = DFEDepthFirstTreeBuilder(splitter, min_samples_split,
                                               min_samples_leaf,
                                               min_weight_leaf,
                                               max_depth,
                                               self.min_impurity_decrease,
                                               min_impurity_split)
        else:
            builder = BestFirstTreeBuilder(splitter, min_samples_split,
                                           min_samples_leaf,
                                           min_weight_leaf,
                                           max_depth,
                                           max_leaf_nodes,
                                           self.min_impurity_decrease,
                                           min_impurity_split)

        if self.uncertain_features is None:
            self.uncertain_features = np.full((self.n_features_,), self.default_feature_uncertainty)
        elif self.n_features_ > self.uncertain_features.shape[0]:
            self.uncertain_features = np.append(self.uncertain_features, np.full(
                                                                (self.n_features_ - self.uncertain_features.shape[0],),
                                                                self.default_feature_uncertainty))

        builder.build(self.tree_, X, y, sample_weight, self.uncertain_features, feature_bias, self.missing_values)

        if self.n_outputs_ == 1 and is_classifier(self):
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        self._prune_tree()

        return self

    def predict_proba(self, X, check_input=True):
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        proba = self.tree_.dfe_predict(X, self.uncertain_features, self.missing_values)

        if self.n_outputs_ == 1:
            proba = proba[:, :self.n_classes_]
            normalizer = proba.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            proba /= normalizer

            return proba

        else:
            all_proba = []

            for k in range(self.n_outputs_):
                proba_k = proba[:, k, :self.n_classes_[k]]
                normalizer = proba_k.sum(axis=1)[:, np.newaxis]
                normalizer[normalizer == 0.0] = 1.0
                proba_k /= normalizer
                all_proba.append(proba_k)

            return all_proba


class UDDecisionTreeClassifier(DecisionTreeClassifier):
    @_deprecate_positional_args
    def __init__(self, *,
                 criterion="gini",
                 splitter="ud_best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 class_weight=None,
                 ccp_alpha=0.0):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            ccp_alpha=ccp_alpha)
        DENSE_SPLITTERS["ud_best"] = ud_splitter.UDBestSplitter

    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted="deprecated", feature_bias=None):

        random_state = check_random_state(self.random_state)

        if self.ccp_alpha < 0.0:
            raise ValueError("ccp_alpha must be greater than or equal to 0")

        if check_input:
            # Need to validate separately here.
            # We can't pass multi_ouput=True because that would allow y to be
            # csr.
            check_X_params = dict(dtype=DTYPE, accept_sparse="csc")
            check_y_params = dict(ensure_2d=False, dtype=None)
            X, y = self._validate_data(X, y,
                                       validate_separately=(check_X_params,
                                                            check_y_params))
            if issparse(X):
                X.sort_indices()

                if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                    raise ValueError("No support for np.int64 index based "
                                     "sparse matrices")

            if self.criterion == "poisson":
                if np.any(y < 0):
                    raise ValueError("Some value(s) of y are negative which is"
                                     " not allowed for Poisson regression.")
                if np.sum(y) <= 0:
                    raise ValueError("Sum of y is not positive which is "
                                     "necessary for Poisson regression.")

        # Determine output settings
        n_samples, self.n_features_ = X.shape
        self.n_features_in_ = self.n_features_
        is_classification = is_classifier(self)

        y = np.atleast_1d(y)
        expanded_class_weight = None

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        if is_classification:
            check_classification_targets(y)
            y = np.copy(y)

            self.classes_ = []
            self.n_classes_ = []

            if self.class_weight is not None:
                y_original = np.copy(y)

            y_encoded = np.zeros(y.shape, dtype=int)
            for k in range(self.n_outputs_):
                classes_k, y_encoded[:, k] = np.unique(y[:, k],
                                                       return_inverse=True)
                self.classes_.append(classes_k)
                self.n_classes_.append(classes_k.shape[0])
            y = y_encoded

            if self.class_weight is not None:
                expanded_class_weight = compute_sample_weight(
                    self.class_weight, y_original)

            self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        # Check parameters
        max_depth = (np.iinfo(np.int32).max if self.max_depth is None
                     else self.max_depth)
        max_leaf_nodes = (-1 if self.max_leaf_nodes is None
                          else self.max_leaf_nodes)

        if isinstance(self.min_samples_leaf, numbers.Integral):
            if not 1 <= self.min_samples_leaf:
                raise ValueError("min_samples_leaf must be at least 1 "
                                 "or in (0, 0.5], got %s"
                                 % self.min_samples_leaf)
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            if not 0. < self.min_samples_leaf <= 0.5:
                raise ValueError("min_samples_leaf must be at least 1 "
                                 "or in (0, 0.5], got %s"
                                 % self.min_samples_leaf)
            min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))

        if isinstance(self.min_samples_split, numbers.Integral):
            if not 2 <= self.min_samples_split:
                raise ValueError("min_samples_split must be an integer "
                                 "greater than 1 or a float in (0.0, 1.0]; "
                                 "got the integer %s"
                                 % self.min_samples_split)
            min_samples_split = self.min_samples_split
        else:  # float
            if not 0. < self.min_samples_split <= 1.:
                raise ValueError("min_samples_split must be an integer "
                                 "greater than 1 or a float in (0.0, 1.0]; "
                                 "got the float %s"
                                 % self.min_samples_split)
            min_samples_split = int(ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                if is_classification:
                    max_features = max(1, int(np.sqrt(self.n_features_)))
                else:
                    max_features = self.n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError("Invalid value for max_features. "
                                 "Allowed string values are 'auto', "
                                 "'sqrt' or 'log2'.")
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1,
                                   int(self.max_features * self.n_features_))
            else:
                max_features = 0

        self.max_features_ = max_features

        if len(y) != n_samples:
            raise ValueError("Number of labels=%d does not match "
                             "number of samples=%d" % (len(y), n_samples))
        if not 0 <= self.min_weight_fraction_leaf <= 0.5:
            raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero. ")
        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")
        if not isinstance(max_leaf_nodes, numbers.Integral):
            raise ValueError("max_leaf_nodes must be integral number but was "
                             "%r" % max_leaf_nodes)
        if -1 < max_leaf_nodes < 2:
            raise ValueError(("max_leaf_nodes {0} must be either None "
                              "or larger than 1").format(max_leaf_nodes))

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Set min_weight_leaf from min_weight_fraction_leaf
        if sample_weight is None:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               n_samples)
        else:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               np.sum(sample_weight))

        min_impurity_split = self.min_impurity_split
        if min_impurity_split is not None:
            warnings.warn(
                "The min_impurity_split parameter is deprecated. Its default "
                "value has changed from 1e-7 to 0 in version 0.23, and it "
                "will be removed in 1.0 (renaming of 0.25). Use the "
                "min_impurity_decrease parameter instead.",
                FutureWarning
            )

            if min_impurity_split < 0.:
                raise ValueError("min_impurity_split must be greater than "
                                 "or equal to 0")
        else:
            min_impurity_split = 0

        if self.min_impurity_decrease < 0.:
            raise ValueError("min_impurity_decrease must be greater than "
                             "or equal to 0")

        # TODO: Remove in 1.1
        if X_idx_sorted != "deprecated":
            warnings.warn(
                "The parameter 'X_idx_sorted' is deprecated and has no "
                "effect. It will be removed in 1.1 (renaming of 0.26). You "
                "can suppress this warning by not passing any value to the "
                "'X_idx_sorted' parameter.",
                FutureWarning
            )

        # Build tree
        criterion = self.criterion
        if not isinstance(criterion, Criterion):
            if is_classification:
                criterion = CRITERIA_CLF[self.criterion](self.n_outputs_,
                                                         self.n_classes_)
            else:
                criterion = CRITERIA_REG[self.criterion](self.n_outputs_,
                                                         n_samples)

        SPLITTERS = SPARSE_SPLITTERS if issparse(X) else DENSE_SPLITTERS

        splitter = self.splitter
        if not isinstance(self.splitter, Splitter):
            splitter = SPLITTERS[self.splitter](criterion,
                                                self.max_features_,
                                                min_samples_leaf,
                                                min_weight_leaf,
                                                random_state)

        if is_classifier(self):
            self.tree_ = Tree(self.n_features_,
                                 self.n_classes_, self.n_outputs_)
        else:
            self.tree_ = Tree(self.n_features_,
                                 # TODO: tree should't need this in this case
                                 np.array([1] * self.n_outputs_, dtype=np.intp),
                                 self.n_outputs_)

        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        if max_leaf_nodes < 0:
            builder = UDDepthFirstTreeBuilder(splitter, min_samples_split,
                                               min_samples_leaf,
                                               min_weight_leaf,
                                               max_depth,
                                               self.min_impurity_decrease,
                                               min_impurity_split)
        else:
            builder = BestFirstTreeBuilder(splitter, min_samples_split,
                                           min_samples_leaf,
                                           min_weight_leaf,
                                           max_depth,
                                           max_leaf_nodes,
                                           self.min_impurity_decrease,
                                           min_impurity_split)

        builder.build(self.tree_, X, y, sample_weight, feature_bias)

        if self.n_outputs_ == 1 and is_classifier(self):
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        self._prune_tree()

        return self
