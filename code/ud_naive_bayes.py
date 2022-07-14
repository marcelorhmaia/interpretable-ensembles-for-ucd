# Copyright (C) 2022  Marcelo R. H. Maia <mmaia@ic.uff.br, marcelo.h.maia@ibge.gov.br>
# License: GPLv3 (https://www.gnu.org/licenses/gpl-3.0.html)


import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB, _BaseDiscreteNB
from sklearn.utils import _deprecate_positional_args
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted


def binarize(X, threshold=0.0, missing_values=None):
    cond = X > threshold
    not_cond = X <= threshold
    if missing_values is not None:
        cond &= X != missing_values
        not_cond &= X != missing_values
    X[cond] = 1
    X[not_cond] = 0
    return X


class UDGaussianNB(GaussianNB):

    def __init__(self, *, priors=None, var_smoothing=1e-9, missing_values=-1.0):
        super().__init__(priors=priors, var_smoothing=var_smoothing)
        self.missing_values = missing_values

    def _update_mean_variance(self, n_past, mu, var, X, sample_weight=None):
        if X.shape[0] == 0:
            return mu, var

        mx = np.ma.masked_equal(X, self.missing_values)

        # Compute (potentially weighted) mean and variance of new datapoints
        if sample_weight is not None:
            n_new = float(sample_weight.sum())
            new_mu = np.average(mx, axis=0, weights=sample_weight)
            new_var = np.average((mx - new_mu) ** 2, axis=0,
                                 weights=sample_weight)
        else:
            n_new = X.shape[0]
            new_var = np.var(mx, axis=0)
            new_mu = np.mean(mx, axis=0)

        if n_past == 0:
            return new_mu, new_var

        n_total = float(n_past + n_new)

        # Combine mean of old and new data, taking into consideration
        # (weighted) number of observations
        total_mu = (n_new * new_mu + n_past * mu) / n_total

        # Combine variance of old and new data, taking into consideration
        # (weighted) number of observations. This is achieved by combining
        # the sum-of-squared-differences (ssd)
        old_ssd = n_past * var
        new_ssd = n_new * new_var
        total_ssd = (old_ssd + new_ssd +
                     (n_new * n_past / n_total) * (mu - new_mu) ** 2)
        total_var = total_ssd / n_total

        return total_mu, total_var

    def _joint_log_likelihood(self, X):
        joint_log_likelihood = []
        mx = np.ma.masked_equal(X, self.missing_values)
        for i in range(np.size(self.classes_)):
            jointi = np.log(self.class_prior_[i])
            n_ij = - 0.5 * np.sum(np.log(2. * np.pi * self.sigma_[i, :]))
            n_ij -= 0.5 * np.sum(((mx - self.theta_[i, :]) ** 2) /
                                 (self.sigma_[i, :]), 1)
            joint_log_likelihood.append(jointi + n_ij)

        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood


class UDBernoulliNB(_BaseDiscreteNB):

    @_deprecate_positional_args
    def __init__(self, *, alpha=1.0, binarize=.0, fit_prior=True,
                 class_prior=None, missing_values=None):
        self.alpha = alpha
        self.binarize = binarize
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.missing_values = missing_values

    def _check_X(self, X):
        X = super()._check_X(X)
        if self.missing_values is not None:
            X[X == self.missing_values] = -1
        if self.binarize is not None:
            if self.missing_values is not None:
                X = binarize(X, threshold=self.binarize, missing_values=-1)
            else:
                X = binarize(X, threshold=self.binarize)
        return X

    def _check_X_y(self, X, y):
        X, y = super()._check_X_y(X, y)
        if self.missing_values is not None:
            X[X == self.missing_values] = -1
        if self.binarize is not None:
            if self.missing_values is not None:
                X = binarize(X, threshold=self.binarize, missing_values=-1)
            else:
                X = binarize(X, threshold=self.binarize)
        return X, y

    def _init_counters(self, n_effective_classes, n_features):
        super()._init_counters(n_effective_classes, n_features)
        self.neg_feature_count_ = np.zeros((n_effective_classes, n_features),
                                           dtype=np.float64)

    def _count(self, X, Y):
        """Count and smooth feature occurrences."""
        mx = np.ma.masked_equal(X, -1)
        self.feature_count_ += np.ma.dot(Y.T, mx)
        self.class_count_ += Y.sum(axis=0)

        neg_mx = 1 - mx
        self.neg_feature_count_ += np.ma.dot(Y.T, neg_mx)

    def _update_feature_log_prob(self, alpha):
        """Apply smoothing to raw counts and recompute log probabilities"""
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = self.class_count_ + alpha * 2

        self.feature_log_prob_ = (np.log(smoothed_fc) -
                                  np.log(smoothed_cc.reshape(-1, 1)))

        smoothed_neg_fc = self.neg_feature_count_ + alpha

        self.neg_feature_log_prob_ = (np.log(smoothed_neg_fc) -
                                      np.log(smoothed_cc.reshape(-1, 1)))

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        n_classes, n_features = self.feature_log_prob_.shape
        n_samples, n_features_X = X.shape

        if n_features_X != n_features:
            raise ValueError("Expected input with %d features, got %d instead"
                             % (n_features, n_features_X))

        neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
        # Compute  neg_prob · (1 - X).T  as  ∑neg_prob - X · neg_prob
        mx = np.ma.masked_equal(X, -1)
        jll = np.ma.dot(mx, (self.feature_log_prob_ - neg_prob).T)
        jll += self.class_log_prior_ + neg_prob.sum(axis=1)

        if self.missing_values is not None:
            neg_neg_prob = np.log(1 - np.exp(self.neg_feature_log_prob_))
            neg_mx = 1 - mx
            jll += np.ma.dot(neg_mx, (self.neg_feature_log_prob_ - neg_neg_prob).T)
            jll += neg_neg_prob.sum(axis=1)

        return jll.filled()


class InterpretableBernoulliNB(BernoulliNB):

    @property
    def feature_importances_(self):
        check_is_fitted(self)

        return self.compute_feature_importances_()

    def compute_feature_importances_(self, normalize=True):
        """Computes the importance of each feature (aka variable)."""
        feature_prob = np.exp(self.feature_log_prob_)
        importances = np.abs(feature_prob[0] - feature_prob[1])

        if normalize:
            normalizer = np.sum(importances)

            if normalizer > 0.0:
                importances /= normalizer

        return importances

    def sufficiency_based_feature_importances(self, X, normalize=True):
        """Computes the importance of each feature (aka variable) based on its `sufficiency` for the examples in X."""
        X = self._check_X(X)
        minimal_sufficient = self.minimal_sufficient_features(X)
        importances = np.count_nonzero(minimal_sufficient, axis=0) / X.shape[0]
        # cardinalities = np.count_nonzero(minimal_sufficient, axis=1)
        # unique, indices, counts = np.unique(minimal_sufficient, axis=0, return_index=True, return_counts=True)
        # print('Inst\tFreq\tCard')
        # for i in range(unique.shape[0]):
        #     print(indices[i], '\t', counts[i], '\t', cardinalities[indices[i]])

        if normalize:
            normalizer = np.sum(importances)

            if normalizer > 0.0:
                importances /= normalizer

        return importances

    def minimal_sufficient_features(self, X):
        """Returns a `minimal sufficient` set of features for each example in X."""
        X = self._check_X(X)
        y = self.predict(X)
        minimal_sufficient = self.supporting_features(X, y)
        sorted_indices = np.argsort(self.feature_importances_)
        for i in range(X.shape[0]):
            for j in sorted_indices:
                if minimal_sufficient[i, j]:
                    minimal_sufficient[i, j] = False
                    jll = self.reduced_joint_log_likelihood(X[i], minimal_sufficient[i])
                    if self.classes_[np.argmax(jll)] != y[i]:
                        minimal_sufficient[i, j] = True
                        break

        return minimal_sufficient

    def supporting_features(self, X, y):
        """Returns the `supporting` features for each example in X, given the corresponding classes in y."""
        support_features = np.empty(X.shape, dtype=np.bool)
        feature_prob = np.exp(self.feature_log_prob_)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if X[i, j] == 1:
                    support_features[i, j] = feature_prob[y[i], j] > feature_prob[1 - y[i], j]
                else:
                    support_features[i, j] = 1 - feature_prob[y[i], j] > 1 - feature_prob[1 - y[i], j]

        return support_features

    def reduced_joint_log_likelihood(self, x, mask):
        """Calculate the posterior log probability of the sample x considering a reduced model with the features
        filtered by an index mask """
        feature_log_prob = self.feature_log_prob_[:, mask]
        x = x[mask]

        neg_prob = np.log(1 - np.exp(feature_log_prob))
        # Compute  neg_prob · (1 - X).T  as  ∑neg_prob - X · neg_prob
        jll = safe_sparse_dot(x, (feature_log_prob - neg_prob).T)
        jll += self.class_log_prior_ + neg_prob.sum(axis=1)

        return jll
