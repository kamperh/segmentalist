"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2016
"""

import logging
import numpy as np
import random
import time

from gaussian_components import GaussianComponents
from gaussian_components_diag import GaussianComponentsDiag
from gaussian_components_fixedvar import GaussianComponentsFixedVar

logger = logging.getLogger(__name__)


class BigramFBGMM(object):
    """
    A bigram-based finite Bayesian Gaussian mixture model (FBGMM).

    Parameters
    ----------
    K : int
        The number of mixture components. This is actually a maximum number,
        and it is possible to empty out some of these components.
    assignments : vector of int or str
        If vector of int, this gives the initial component assignments. The
        vector should therefore have N entries between 0 and `K`. Values of -1
        is also allowed, indicating that the data vector does not belong to any
        component. Alternatively, `assignments` can take one of the following
        values:
        - "rand": Vectors are assigned randomly to one of `K` components.
        - "each-in-own": Each vector is assigned to a component of its own.
    covariance_type : str
        String describing the type of covariance parameters to use. Must be
        one of "full", "diag" or "fixed".
    """

    def __init__(self, X, prior, K, assignments="rand",
            covariance_type="fixed", lms=1.0, lm=None):
        self.prior = prior
        self.covariance_type = covariance_type
        self.lms = lms
        self.setup_components(K, assignments, X, lm)

    def setup_components(self, K, assignments="rand", X=None, lm=None):
        """
        Setup the `components` attribute.

        See parameters of `FBGMM` for parameters not described below. This
        function is also useful for resetting the `components`, e.g. if you
        want to change the maximum number of possible components.

        Parameters
        ----------
        X : NxD matrix or None
            The data matrix. If None, then this it is assumed that the
            `components` attribute has already been initialized and that this
            function is called to reset the `components`; in this case the data
            is taken from the previous initialization.
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
        else:
            # `assignments` is a vector
            pass
        # Make sure we have consequetive values
        for k in xrange(assignments.max()):
            while len(np.nonzero(assignments == k)[0]) == 0:
                assignments[np.where(assignments > k)] -= 1
            if assignments.max() == k:
                break

        if self.covariance_type == "full":
            self.components = GaussianComponents(X, self.prior, assignments, K_max=K)
        elif self.covariance_type == "diag":
            self.components = GaussianComponentsDiag(X, self.prior, assignments, K_max=K)
        elif self.covariance_type == "fixed":
            self.components = GaussianComponentsFixedVar(X, self.prior, assignments, K_max=K, lm=lm)
        else:
            assert False, "Invalid covariance type."

    def log_prob_X_given_z(self):
        """Return the log probability of data in each component p(X|z)."""
        return self.components.log_marg()

    def get_n_assigned(self):
        """Return the number of assigned data vectors."""
        return len(np.where(self.components.assignments != -1)[0])
