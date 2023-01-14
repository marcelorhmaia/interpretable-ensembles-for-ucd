# Copyright (C) 2022  Marcelo R. H. Maia <mmaia@ic.uff.br, marcelo.h.maia@ibge.gov.br>
# License: GPLv3 (https://www.gnu.org/licenses/gpl-3.0.html)


from sklearn.tree._splitter cimport SplitRecord, Splitter
from sklearn.tree._tree cimport DOUBLE_t, SIZE_t

from ud_criterion cimport DFEClassificationCriterion
from ud_tree cimport BOOL_t


cdef class DFESplitter(Splitter):
    cdef DFEClassificationCriterion dfe_c_criterion

    cdef int dfe_node_split(self, double impurity, SplitRecord* split,
                        SIZE_t* n_constant_features, BOOL_t* uncertain_features, double* feature_bias, double missing_values) nogil except -1

    cdef int dfe_node_split_with_copy(self, double impurity, SplitRecord* split,
                        SIZE_t* n_constant_features, BOOL_t* uncertain_features, double* feature_bias, double missing_values, SIZE_t* samples) nogil except -1

    cdef int dfe_node_reset_with_copy(self, SIZE_t start, SIZE_t end,
                        double* weighted_n_node_samples, SIZE_t* samples, DOUBLE_t* sample_weight) nogil except -1


cdef class UDSplitter(Splitter):
    cdef int ud_node_split(self, double impurity, SplitRecord* split,
                        SIZE_t* n_constant_features, double* feature_bias) nogil except -1