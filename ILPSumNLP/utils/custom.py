#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import scipy.sparse as sp

# noinspection PyProtectedMember
from sklearn.feature_extraction.text import TfidfTransformer, _document_frequency


class CustomizedTfidfTransformer(TfidfTransformer):
    # noinspection PyPep8Naming,PyIncorrectDocstring
    def fit(self, X, y=None):
        """Learn the idf vector (global term weights)

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        """
        if not sp.issparse(X):
            X = sp.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)

            # perform idf smoothing if required
            df += int(self.smooth_idf)
            # n_samples += int(self.smooth_idf)

            # log+1 instead of log makes sure terms with zero idf don't get
            # suppressed entirely.
            idf = np.log(float(n_samples) / df) + 1.0

            # noinspection PyAttributeOutsideInit,PyTypeChecker
            self._idf_diag = sp.spdiags(idf, diags=0, m=n_features, n=n_features, format='csr')

        return self
