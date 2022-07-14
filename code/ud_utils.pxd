# Copyright (C) 2022  Marcelo R. H. Maia <mmaia@ic.uff.br, marcelo.h.maia@ibge.gov.br>
# License: GPLv3 (https://www.gnu.org/licenses/gpl-3.0.html)


#import numpy as np
#cimport numpy as np
from sklearn.neighbors._quad_tree cimport Cell
from sklearn.tree._tree cimport Node
from sklearn.tree._utils cimport DOUBLE_t, DTYPE_t, PriorityHeapRecord, SIZE_t, WeightedPQueueRecord

#ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
#ctypedef np.npy_intp SIZE_t              # Type for indices and counters


ctypedef fused realloc_ptr:
    # Add pointer types here as needed.
    (DTYPE_t*)
    (SIZE_t*)
    (unsigned char*)
    (WeightedPQueueRecord*)
    (DOUBLE_t*)
    (DOUBLE_t**)
    (Node*)
    (Cell*)
    (Node**)
    (DFEStackRecord*)
    (PriorityHeapRecord*)

cdef realloc_ptr safe_realloc(realloc_ptr* p, size_t nelems) nogil except *

# =============================================================================
# Stack data structure
# =============================================================================

# A record on the stack for depth-first tree growing
cdef struct DFEStackRecord:
    SIZE_t start
    SIZE_t end
    SIZE_t depth
    SIZE_t parent
    bint is_left
    double impurity
    SIZE_t n_constant_features
    SIZE_t* samples
    DOUBLE_t* sample_weight

cdef class DFEStack:
    cdef SIZE_t capacity
    cdef SIZE_t top
    cdef DFEStackRecord* stack_

    cdef bint is_empty(self) nogil
    cdef int push(self, SIZE_t start, SIZE_t end, SIZE_t depth, SIZE_t parent,
                  bint is_left, double impurity,
                  SIZE_t n_constant_features) nogil except -1
    cdef int push_with_copy(self, SIZE_t start, SIZE_t end, SIZE_t depth, SIZE_t parent,
                  bint is_left, double impurity,
                  SIZE_t n_constant_features, SIZE_t* samples, DOUBLE_t* sample_weight) nogil except -1
    cdef int pop(self, DFEStackRecord* res) nogil