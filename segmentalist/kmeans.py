"""
K-means model.

This is the parallel of `fbgmm` but for the segmental k-means model rather
than the segmental Bayesian GMM model.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2016
"""

import logging
import numpy as np
import random
import time

logger = logging.getLogger(__name__)

from kmeans_components import KMeansComponents


#-----------------------------------------------------------------------------#
#                             K-MEANS MODEL CLASS                             #
#-----------------------------------------------------------------------------#

class KMeans(object):
    """
    A k-means model.

    See `KMeansComponents` for an overview of the parameters not mentioned
    below.

    Parameters
    ----------
    K : int
        The number of mixture components.
    assignments : vector of int or str
        If vector of int, this gives the initial component assignments. The
        vector should therefore have N entries between 0 and `K`. Values of
        -1 is also allowed, indicating that the data vector does not belong to
        any component. Alternatively, `assignments` can take one of the
        following values:
        - "rand": Vectors are assigned randomly to one of `K` components.
        - "each-in-own": Each vector is assigned to a component of its own.
        - "spread": Vectors are also randomly assigned, but here an attempt is
          made to spread the items over the different components.
    """

    def __init__(self, X, K, assignments="rand"):
        self.setup_components(K, assignments, X)

    def setup_components(self, K, assignments="rand", X=None):
        """
        Setup the `components` attribute.

        See parameters of `KMeans` for parameters not described below. This
        function is also useful for resetting the `components`, e.g. if you
        want to change the maximum number of possible components.

        Parameters
        ----------
        X : NxD matrix or None
            The data matrix. If None, then it is assumed that the `components`
            attribute has already been initialized and that this function is
            called to reset the `components`; in this case the data is taken
            from the previous initialization.
        """
        if X is None:
            assert hasattr(self, "components")
            X = self.components.X

        N, D = X.shape

        # Initial component assignments
        if isinstance(assignments, basestring) and assignments == "rand":
            assignments = np.random.randint(0, K, N)
        elif isinstance(assignments, basestring) and assignments == "each-in-own":
            assignments = np.arange(N)
        elif isinstance(assignments, basestring) and assignments == "spread":
            assignment_list = (range(K)*int(np.ceil(float(N)/K)))[:N]
            random.shuffle(assignment_list)
            assignments = np.array(assignment_list)
        else:
            # `assignments` is a vector
            pass

        # Make sure we have consequetive values
        for k in xrange(assignments.max()):
            while len(np.nonzero(assignments == k)[0]) == 0:
                assignments[np.where(assignments > k)] -= 1
            if assignments.max() == k:
                break

        self.components = KMeansComponents(X, assignments, K)

    # @profile
    def fit(self, n_iter, consider_unassigned=True, no_empty=True):
        """
        Perform `n_iter` iterations of k-means optimization.

        Parameters
        ----------
        consider_unassigned : bool
            Whether unassigned vectors (-1 in `assignments`) should be
            considered during optimization.

        Return
        ------
        record_dict : dict
            Contains several fields describing the optimization iterations.
            Each field is described by its key and statistics are given in a
            list covering the iterations.
        """

        # Setup record dictionary
        record_dict = {}
        record_dict["sum_neg_sqrd_norm"] = []
        record_dict["components"] = []
        record_dict["n_mean_updates"] = []
        record_dict["sample_time"] = []
        
        # Loop over iterations
        start_time = time.time()
        for i_iter in xrange(n_iter):

            # List of tuples (i, k) where i is the data item and k is the new
            # component to which it should be assigned
            mean_numerator_updates = []

            # Assign data items
            for i in xrange(self.components.N):
                
                # Keep track of old value in case we do not have to update
                k_old = self.components.assignments[i]
                if not consider_unassigned and k_old == -1:
                    continue

                # Pick the new component
                # scores = np.zeros(self.components.K_max)
                # scores[:self.components.K] = self.components.neg_sqrd_norm(i)
                scores = self.components.neg_sqrd_norm(i)
                # k = np.nanargmax(scores)
                k = np.argmax(scores)

                if k != k_old:
                    mean_numerator_updates.append((i, k))

            # Update means
            for i, k in mean_numerator_updates:
                self.components.del_item(i) #, no_empty=no_empty)
                self.components.add_item(i, k)

            # Remove empty components
            self.components.clean_components()
            # self.components.setup_random_means()

            # Update record
            record_dict["sum_neg_sqrd_norm"].append(self.components.sum_neg_sqrd_norm())
            record_dict["components"].append(self.components.K)
            record_dict["n_mean_updates"].append(len(mean_numerator_updates))
            record_dict["sample_time"].append(time.time() - start_time)
            start_time = time.time()

            # Log info
            info = "iteration: " + str(i_iter)
            for key in sorted(record_dict):
                info += ", " + key + ": " + str(record_dict[key][-1])
            logger.info(info)

            if len(mean_numerator_updates) == 0:
                break

        return record_dict

    def get_n_assigned(self):
        """Return the number of assigned data vectors."""
        return len(np.where(self.components.assignments != -1)[0])



#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#


def main():

    import random

    logging.basicConfig(level=logging.INFO)

    random.seed(1)
    np.random.seed(1)

    # Data parameters
    D = 2           # dimensions
    N = 10          # number of points to generate
    K_true = 4      # the true number of components

    # Model parameters
    K = 6           # number of components
    n_iter = 10

    # Generate data
    mu_scale = 4.0
    covar_scale = 0.7
    z_true = np.random.randint(0, K_true, N)
    mu = np.random.randn(D, K_true)*mu_scale
    X = mu[:, z_true] + np.random.randn(D, N)*covar_scale
    X = X.T

    # Setup k-means model
    model = KMeans(X, K, "rand")


if __name__ == "__main__":
    main()
