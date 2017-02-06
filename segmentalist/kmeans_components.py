"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2016
"""

import logging
import numpy as np
import random

logger = logging.getLogger(__name__)


#-----------------------------------------------------------------------------#
#                           K-MEANS COMPONENTS CLASS                          #
#-----------------------------------------------------------------------------#

class KMeansComponents(object):
    """
    Components of a k-means model.

    At present, this model does not support deleting components, and if no item
    is assigned to a component, its mean is implicitly set to a random data
    vector.

    Parameters
    ----------
    X : NxD matrix
        A matrix of N data vectors, each of dimension D.
    assignments : Nx1 vector of int
        The initial component assignments. If this values is None, then all
        data vectors are left unassigned indicated with -1 in the vector.
        Components should be labelled from 0.
    K_max : int
        The maximum number of components.

    Global attributes
    -----------------
    N : int
        Number of data vectors.
    D : int 
        Dimensionality of data vectors.
    K : int
        Number of components.

    Component attributes
    --------------------
    mean_numerators : KxD matrix
        The sum of all the data vectors assigned to each component, i.e. the
        component means without normalization .
    counts : Kx1 vector of int
        Counts for each of the K components.
    """

    def __init__(self, X, assignments, K_max):

        # Attributes from parameters
        self.X = X
        self.N, self.D = X.shape
        self.K_max = K_max

        # Initialize attributes
        self.mean_numerators = np.zeros((self.K_max, self.D), np.float)
        self.counts = np.zeros(self.K_max, np.int)
        # self.means = np.zeros((self.K_max, self.D), np.float)

        # Check that assignments are valid
        self.K = 0
        assignments = np.asarray(assignments, np.int)
        assert (self.N, ) == assignments.shape
        # Apart from unassigned (-1), components should be labelled from 0
        assert set(assignments).difference([-1]) == set(range(assignments.max() + 1))
        self.assignments = -1*np.ones(self.N, dtype=np.int)

        self.setup_random_means()
        self.means = self.random_means.copy()

        # Add the data items
        for k in range(assignments.max() + 1):
            for i in np.where(assignments == k)[0]:
                self.add_item(i, k)

        # Temp: used if no mean is assigned
        # This is wrong!! Make sure to pick point from valid X
        # replace = self.K_max < self.N
        # self.random_means = self.X[np.random.choice(range(self.N), self.K_max, replace=True), :]
        # self.random_means = self.X[np.random.choice(np.where(self.assignments != -1)[0], self.K_max, replace=True), :]
        # self.prior_mean = np.zeros((self.K_max, self.D), np.float)

    def setup_random_means(self):
        self.random_means = self.X[np.random.choice(range(self.N), self.K_max, replace=True), :]

    def add_item(self, i, k):
        """
        Add data vector `X[i]` to component `k`.

        If `k` is `K`, then a new component is added. No checks are performed
        to make sure that `X[i]` is not already assigned to another component.
        """
        assert not i == -1
        assert self.assignments[i] == -1

        if k > self.K:
            k = self.K
        if k == self.K:
            self.K += 1

        self.mean_numerators[k, :] += self.X[i]
        self.counts[k] += 1
        self.means[k, :] = self.mean_numerators[k, :] / self.counts[k]
        self.assignments[i] = k

    def del_item(self, i):
        """
        Remove data vector `X[i]` from its component.

        The `no_empty` parameter ensures that empty components always contain
        at least one item. If a item is the last one to be removed from a
        component, a random item is assigned to that component to re-initialize
        it.
        """
        
        assert not i == -1
        k = self.assignments[i]

        # Only do something if the data vector has been assigned
        if k != -1:
            self.counts[k] -= 1
            self.assignments[i] = -1
            self.mean_numerators[k, :] -= self.X[i]
            if self.counts[k] != 0:
                self.means[k, :] = self.mean_numerators[k, :] / self.counts[k]

            # # Temp 3
            # if no_empty and self.counts[k] == 0:
            #     i_rnd = random.choice(np.where(self.assignments != -1)[0])
            #     self.del_item(i_rnd, True)
            #     self.add_item(i_rnd, k)

            # if self.counts[k] == 0:
            #     # Can just delete the component, don't have to update anything
            #     self.del_component(k)
            # else:
            #     # Update the component
            #     self.mu_N_numerators[k, :] -= self.precision*self.X[i]
            #     self.precision_Ns[k, :] -= self.precision
            #     self._update_log_prod_precision_pred_and_precision_pred(k)

    def del_component(self, k):
        """Remove the component `k`."""

        assert k < self.K

        logger.debug("Deleting component " + str(k))
        self.K -= 1
        if k != self.K:
            # Put stats from last component into place of the one being removed
            self.mean_numerators[k] = self.mean_numerators[self.K]
            self.counts[k] = self.counts[self.K]
            self.means[k, :] = self.mean_numerators[self.K, :] / self.counts[self.K]
            self.assignments[np.where(self.assignments == self.K)] = k

        # Empty out stats for last component
        self.mean_numerators[self.K].fill(0.)
        self.counts[self.K] = 0
        self.means[self.K] = self.random_means[self.K]

    # @profile
    def neg_sqrd_norm(self, i):
        """
        Return the vector of the negative squared distances of `X[i]` to the
        mean of each of the components.
        """
        # self.random_means = self.X[np.random.choice(np.where(self.assignments != -1)[0], self.K_max, replace=True), :]        
        # means = np.zeros((self.K_max, self.D), np.float)  # implict zero prior on mean
        # means = self.X[np.random.choice(range(self.N), self.K_max, replace=True), :]  # another option

        # means = self.random_means.copy()
        # means[:self.K] = self.mean_numerators[:self.K]/self.counts[:self.K, None]

        # means = np.zeros((self.K_max, self.D))
        # means[:self.K] = self.mean_numerators[:self.K]/self.counts[:self.K, None]
        # means[self.K:] = self.random_means[self.K:]
        # assert np.all(means == self.means)

        # # Temp
        # active_means = self.mean_numerators[:self.K]/self.counts[:self.K, None]
        # means = np.zeros((self.K_max + 1, self.D))
        # means[:self.K] = active_means
        # means[-1, :] = np.mean(active_means, axis=0)

        # print np.mean(active_means, axis=0).shape
        # assert False
        # means = self.random_means.copy()
        # means[:self.K] = active_means

        # # Temp
        # if self.K < self.K_max:
        #     means = self.random_means
        #     means[:self.K] = self.mean_numerators[:self.K]/self.counts[:self.K, None]
        #     # means[self.K:] = self.random_means[self.K:]
        #     # print np.where(self.counts == 0)[0]
        #     # assert False
        # # assert False
        # else:
        #     means = self.mean_numerators[:self.K]/self.counts[:self.K, None]

        # # Temp 2
        # if self.K < self.K_max:
        #     means = self.random_means[:self.K + 1]
        #     means[:self.K] = self.mean_numerators[:self.K]/self.counts[:self.K, None]
        #     # print "jip"
        # else:
        #     means = self.mean_numerators[:self.K]/self.counts[:self.K, None]

        # # Temp 4
        # deltas = means - self.X[i]
        # neg_sqrd_norms = -(deltas*deltas).sum(axis=1)
        # constant = -0.6
        # neg_sqrd_norms[self.K:] = constant
        # return neg_sqrd_norms
        # assert False

        # deltas = means - self.X[i]
        deltas = self.means - self.X[i]
        return -(deltas*deltas).sum(axis=1)  # equavalent to np.linalg.norm(deltas, axis=1)**2

    def max_neg_sqrd_norm_i(self, i):
        return np.max(self.neg_sqrd_norm(i))

    def argmax_neg_sqrd_norm_i(self, i):
        return np.argmax(self.neg_sqrd_norm(i))

    def sum_neg_sqrd_norm(self):
        """
        Return the k-means maximization objective: the sum of the negative
        squared norms of all the items.
        """
        objective = 0
        for k in xrange(self.K):
            # if self.counts[k] == 0:
            #     continue
            X = self.X[np.where(self.assignments == k)]
            mean = self.mean_numerators[k, :]/self.counts[k]
            deltas = mean - X
            objective += -np.sum(deltas*deltas)
        return objective

    def get_assignments(self, list_of_i):
        """
        Return a vector of the current assignments for the data vector indices
        in `list_of_i`.
        """
        return self.assignments[np.asarray(list_of_i)]

    def get_max_assignments(self, list_of_i):
        """
        Return a vector of the best assignments for the data vector indices in
        `list_of_i`.
        """
        return [self.argmax_neg_sqrd_norm_i(i) for i in list_of_i]

    def clean_components(self):
        """Remove all empty components."""
        for k in np.where(self.counts[:self.K] == 0)[0][::-1]:
            self.del_component(k)



#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    
    import numpy.testing as npt

    np.random.seed(1)

    # Generate data
    D = 3           # dimensions
    N = 10          # number of points to generate
    K_true = 4      # the true number of components
    mu_scale = 4.0
    covar_scale = 0.7
    z_true = np.random.randint(0, K_true, N)
    mu = np.random.randn(D, K_true)*mu_scale
    X = mu[:, z_true] + np.random.randn(D, N)*covar_scale
    X = X.T

    # Model
    K = 3
    assignments = np.random.randint(0, K, N)
    components = KMeansComponents(X, assignments, K)

    # Test mean_numerators
    n = 0
    for k in xrange(components.K):
        component_mean = components.mean_numerators[k]/components.counts[k]
        X_k = X[np.where(components.assignments == k)]
        n += X_k.shape[0]

        npt.assert_almost_equal(np.mean(X_k, axis=0), component_mean)
    assert n == N

    # Test neg_sqrd_norm
    for i in xrange(N):
        x_i = X[i, :]
        expected_sqrd_norms = []
        for k in xrange(components.K):
            component_mean = components.mean_numerators[k]/components.counts[k]
            expected_sqrd_norms.append(np.linalg.norm(x_i - component_mean)**2)
        npt.assert_almost_equal(components.neg_sqrd_norm(i), -np.array(expected_sqrd_norms))

    # Test sum_neg_sqrd_norm
    expected_sum_neg_sqrd_norm = 0.
    for i in xrange(N):
        x_i = X[i, :]
        expected_sqrd_norms = []
        for k in xrange(components.K):
            component_mean = components.mean_numerators[k]/components.counts[k]
            expected_sqrd_norms.append(np.linalg.norm(x_i - component_mean)**2)
        expected_sum_neg_sqrd_norm += -np.array(expected_sqrd_norms)[components.assignments[i]]
    npt.assert_almost_equal(components.sum_neg_sqrd_norm(), expected_sum_neg_sqrd_norm)


if __name__ == "__main__":
    main()
