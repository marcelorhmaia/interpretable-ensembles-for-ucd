# Copyright (C) 2022  Marcelo R. H. Maia <mmaia@ic.uff.br, marcelo.h.maia@ibge.gov.br>
# License: GPLv3 (https://www.gnu.org/licenses/gpl-3.0.html)


cimport numpy as np
import numpy as np
from libc.stdint cimport SIZE_MAX
from libc.stdio cimport printf
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy
from numpy import float32 as DTYPE
from scipy.sparse import issparse
from sklearn.tree._splitter cimport SplitRecord, Splitter
from sklearn.tree._tree cimport DOUBLE_t, DTYPE_t, Node, SIZE_t, Tree, TreeBuilder
from sklearn.tree._utils cimport Stack, StackRecord

from ud_splitter cimport DFESplitter
from ud_utils cimport DFEStack, DFEStackRecord


cdef double EPSILON = np.finfo('double').eps
cdef double INFINITY = np.inf

cdef SIZE_t INITIAL_STACK_SIZE = 10
TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED


# =============================================================================
# Tree
# =============================================================================

cdef class DFETree(Tree):
    cpdef np.ndarray dfe_predict(self, object X, np.ndarray uncertain_features, DOUBLE_t missing_values):
        """Predict target for X."""
        out = self.dfe_apply(X, uncertain_features, missing_values)
        if self.n_outputs == 1:
            out = out.reshape(X.shape[0], self.max_n_classes)
        return out

    cpdef np.ndarray dfe_apply(self, object X, np.ndarray uncertain_features, DOUBLE_t missing_values):
        if issparse(X):
            raise ValueError("Not implemented")
        else:
            return self.dfe_apply_dense(X, uncertain_features, missing_values)

    cdef inline np.ndarray dfe_apply_dense(self, object X, np.ndarray uncertain_features, DOUBLE_t missing_values):
        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        n_samples = X.shape[0]

        # Initialize output
        out = np.empty((n_samples, self.n_outputs, self.max_n_classes), dtype=np.double)

        for i in range(n_samples):
            proba = self.get_proba(i, self.nodes, X, uncertain_features, missing_values)
            out[i] = proba[0]

        return out

    cdef inline np.ndarray get_proba(self, SIZE_t sample, Node* node, object X, np.ndarray uncertain_features, DOUBLE_t missing_values):
        # if node is a leaf
        if node.left_child == _TREE_LEAF:
            return self._get_value_ndarray().take([node - self.nodes], axis=0,
                                         mode='clip')

        if uncertain_features[node.feature]:
            if X[sample, node.feature] == missing_values:
                prob_right = self.nodes[node.right_child].weighted_n_node_samples / node.weighted_n_node_samples
            else:
                prob_right = X[sample, node.feature]

            prob_left = 1.0 - prob_right
            proba = self.get_proba(sample, &self.nodes[node.left_child], X, uncertain_features, missing_values) * prob_left
            if prob_right > 0:
                proba += self.get_proba(sample, &self.nodes[node.right_child], X, uncertain_features, missing_values) * prob_right

            return proba

        if X[sample, node.feature] <= node.threshold:
            return self.get_proba(sample, &self.nodes[node.left_child], X, uncertain_features, missing_values)

        return self.get_proba(sample, &self.nodes[node.right_child], X, uncertain_features, missing_values)


# Depth first builder ---------------------------------------------------------

cdef class DFEDepthFirstTreeBuilder(TreeBuilder):
    """Build a decision tree in depth-first fashion."""

    def __cinit__(self, DFESplitter splitter, SIZE_t min_samples_split,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  SIZE_t max_depth, double min_impurity_decrease,
                  double min_impurity_split):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray sample_weight=None, np.ndarray uncertain_features=None, np.ndarray feature_bias=None,
                DOUBLE_t missing_values=-1.0):
        """Build a decision tree from the training set (X, y)."""

        # check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)

        cdef DOUBLE_t* sample_weight_ptr = NULL
        if sample_weight is not None:
            sample_weight_ptr = <DOUBLE_t*> sample_weight.data

        cdef DOUBLE_t* sample_weight_ptr_curr = NULL
        cdef DOUBLE_t* sample_weight_ptr_left = NULL
        cdef DOUBLE_t* sample_weight_ptr_right = NULL

        cdef SIZE_t* samples_curr = NULL
        cdef SIZE_t* samples_left = NULL
        cdef SIZE_t* samples_right = NULL

        cdef BOOL_t* uncertain_features_ptr = NULL
        if uncertain_features is not None:
            uncertain_features_ptr = <BOOL_t*> uncertain_features.data

        cdef double* feature_bias_ptr = NULL
        if feature_bias is not None:
            feature_bias_ptr = <double*> feature_bias.data

        cdef int n_samples = <int> X.shape[0]
        cdef int n_features = <int> X.shape[1]
        cdef double* X_ptr = <double*> calloc(n_samples * n_features, sizeof(double))
        cdef int i = 0
        cdef int j = 0
        while i < n_samples:
            j = 0
            while j < n_features:
                X_ptr[i * n_features + j] = <double> X[i, j]
                j += 1
            i += 1

        # Initial capacity
        cdef int init_capacity

        if tree.max_depth <= 10:
            init_capacity = (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        tree._resize(init_capacity)

        # Parameters
        cdef DFESplitter splitter = self.splitter
        cdef SIZE_t max_depth = self.max_depth
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef SIZE_t min_samples_split = self.min_samples_split
        cdef double min_impurity_decrease = self.min_impurity_decrease
        cdef double min_impurity_split = self.min_impurity_split

        # Recursive partition (without actual recursion)
        splitter.init(X, y, sample_weight_ptr)

        cdef SIZE_t start
        cdef SIZE_t end
        cdef SIZE_t depth
        cdef SIZE_t parent
        cdef bint is_left
        cdef SIZE_t n_node_samples = splitter.n_samples
        cdef double weighted_n_samples = splitter.weighted_n_samples
        cdef double weighted_n_node_samples
        cdef SplitRecord split
        cdef SIZE_t node_id

        cdef double impurity = INFINITY
        cdef SIZE_t n_constant_features
        cdef bint is_leaf
        cdef bint first = 1
        cdef SIZE_t max_depth_seen = -1
        cdef int rc = 0

        cdef DFEStack stack = DFEStack(INITIAL_STACK_SIZE)
        cdef DFEStackRecord stack_record

        cdef int s
        cdef SIZE_t n_values
        cdef DOUBLE_t sum_values
        cdef DOUBLE_t mean_value

        with nogil:
            # push root node onto stack
            rc = stack.push(0, n_node_samples, 0, _TREE_UNDEFINED, 0, INFINITY, 0)
            if rc == -1:
                # got return code -1 - out-of-memory
                with gil:
                    raise MemoryError()

            while not stack.is_empty():
                stack.pop(&stack_record)

                start = stack_record.start
                end = stack_record.end
                depth = stack_record.depth
                parent = stack_record.parent
                is_left = stack_record.is_left
                impurity = stack_record.impurity
                n_constant_features = stack_record.n_constant_features

                n_node_samples = end - start
                if stack_record.samples == NULL:
                    splitter.node_reset(start, end, &weighted_n_node_samples)
                    samples_curr = splitter.samples
                    sample_weight_ptr_curr = sample_weight_ptr
                else:
                    splitter.dfe_node_reset_with_copy(start, end, &weighted_n_node_samples, stack_record.samples, stack_record.sample_weight)
                    samples_curr = stack_record.samples
                    sample_weight_ptr_curr = stack_record.sample_weight

                is_leaf = (depth >= max_depth or
                           n_node_samples < min_samples_split or
                           n_node_samples < 2 * min_samples_leaf or
                           weighted_n_node_samples < 2 * min_weight_leaf)

                if first:
                    impurity = splitter.node_impurity()
                    first = 0

                is_leaf = (is_leaf or
                           (impurity <= min_impurity_split))

                if not is_leaf:
                    if stack_record.samples == NULL:
                        splitter.dfe_node_split(impurity, &split, &n_constant_features, uncertain_features_ptr, feature_bias_ptr, missing_values)
                    else:
                        splitter.dfe_node_split_with_copy(impurity, &split, &n_constant_features, uncertain_features_ptr, feature_bias_ptr, missing_values, stack_record.samples)

                    # If EPSILON=0 in the below comparison, float precision
                    # issues stop splitting, producing trees that are
                    # dissimilar to v0.18
                    is_leaf = (is_leaf or split.pos >= end or
                               (split.improvement + EPSILON <
                                min_impurity_decrease))

                node_id = tree._add_node(parent, is_left, is_leaf, split.feature,
                                         split.threshold, impurity, n_node_samples,
                                         weighted_n_node_samples)

                if node_id == SIZE_MAX:
                    rc = -1
                    break

                # Store value for all nodes, to facilitate tree/model
                # inspection and interpretation
                splitter.node_value(tree.value + node_id * tree.value_stride)

                if is_leaf:
                    if stack_record.samples != NULL:
                        free(stack_record.samples)
                        free(stack_record.sample_weight)
                else:
                    if uncertain_features_ptr[split.feature]:
                        if missing_values < 0:
                            samples_right = <SIZE_t*> calloc(end - split.pos, sizeof(SIZE_t))
                            memcpy(samples_right, samples_curr + split.pos, (end - split.pos) * sizeof(SIZE_t))

                            sample_weight_ptr_right = <DOUBLE_t*> calloc(n_samples, sizeof(DOUBLE_t))
                            memcpy(sample_weight_ptr_right, sample_weight_ptr_curr, n_samples * sizeof(DOUBLE_t))

                            s = split.pos
                            while s < end:
                                sample_weight_ptr_curr[samples_curr[s]] *= 1.0 - X_ptr[samples_curr[s] * n_features + split.feature]
                                sample_weight_ptr_right[samples_curr[s]] *= X_ptr[samples_curr[s] * n_features + split.feature]
                                s += 1

                            # Push right child on stack
                            rc = stack.push_with_copy(0, end - split.pos, depth + 1, node_id, 0,
                                            split.impurity_right, n_constant_features, samples_right, sample_weight_ptr_right)
                            if rc == -1:
                                break

                            # Push left child on stack
                            if stack_record.samples == NULL:
                                rc = stack.push(start, end, depth + 1, node_id, 1,
                                            split.impurity_left, n_constant_features)
                            else:
                                rc = stack.push_with_copy(start, end, depth + 1, node_id, 1,
                                            split.impurity_left, n_constant_features, stack_record.samples, stack_record.sample_weight)

                            if rc == -1:
                                break
                        else:
                            samples_left = <SIZE_t*> calloc(end - start, sizeof(SIZE_t))
                            memcpy(samples_left, samples_curr, (end - start) * sizeof(SIZE_t))

                            sample_weight_ptr_left = <DOUBLE_t*> calloc(n_samples, sizeof(DOUBLE_t))
                            memcpy(sample_weight_ptr_left, sample_weight_ptr_curr, n_samples * sizeof(DOUBLE_t))

                            samples_right = <SIZE_t*> calloc(end - start, sizeof(SIZE_t))
                            memcpy(samples_right, samples_curr, (end - start) * sizeof(SIZE_t))

                            sample_weight_ptr_right = <DOUBLE_t*> calloc(n_samples, sizeof(DOUBLE_t))
                            memcpy(sample_weight_ptr_right, sample_weight_ptr_curr, n_samples * sizeof(DOUBLE_t))

                            sum_values = 0.0
                            n_values = 0
                            s = start
                            while s < end:
                                sum_values += X_ptr[samples_curr[s] * n_features + split.feature]
                                if X_ptr[samples_curr[s] * n_features + split.feature] != missing_values:
                                    n_values += 1
                                s += 1
                            mean_value = sum_values / n_values

                            s = start
                            while s < end:
                                if X_ptr[samples_curr[s] * n_features + split.feature] == missing_values:
                                    sample_weight_ptr_left[samples_curr[s]] *= 1.0 - mean_value
                                    sample_weight_ptr_right[samples_curr[s]] *= mean_value
                                else:
                                    sample_weight_ptr_left[samples_curr[s]] *= 1.0 - X_ptr[samples_curr[s] * n_features + split.feature]
                                    sample_weight_ptr_right[samples_curr[s]] *= X_ptr[samples_curr[s] * n_features + split.feature]
                                s += 1

                            # Push right child on stack
                            rc = stack.push_with_copy(0, end - start, depth + 1, node_id, 0,
                                            split.impurity_right, n_constant_features, samples_right, sample_weight_ptr_right)
                            if rc == -1:
                                break

                            # Push left child on stack
                            rc = stack.push_with_copy(0, end - start, depth + 1, node_id, 1,
                                            split.impurity_left, n_constant_features, samples_left, sample_weight_ptr_left)

                            if rc == -1:
                                break
                    else:
                        # Push right child on stack
                        rc = stack.push(split.pos, end, depth + 1, node_id, 0,
                                        split.impurity_right, n_constant_features)
                        if rc == -1:
                            break

                        # Push left child on stack
                        rc = stack.push(start, split.pos, depth + 1, node_id, 1,
                                        split.impurity_left, n_constant_features)
                        if rc == -1:
                            break

                if depth > max_depth_seen:
                    max_depth_seen = depth

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen
        if rc == -1:
            raise MemoryError()

        free(X_ptr)
