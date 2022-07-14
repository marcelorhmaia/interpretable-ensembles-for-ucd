# Copyright (C) 2022  Marcelo R. H. Maia <mmaia@ic.uff.br, marcelo.h.maia@ibge.gov.br>
# License: GPLv3 (https://www.gnu.org/licenses/gpl-3.0.html)


from sklearn.tree._criterion cimport ClassificationCriterion
from sklearn.tree._tree cimport DTYPE_t, SIZE_t


cdef class DFEClassificationCriterion(ClassificationCriterion):
    cdef int update_uncertain(self, SIZE_t new_pos, DTYPE_t* values, double missing_values) nogil except -1
