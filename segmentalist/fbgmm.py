"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2014-2015
"""

from scipy.misc import logsumexp
from scipy.special import gammaln
import logging
import numpy as np
import random
import time

from gaussian_components import GaussianComponents
from gaussian_components_diag import GaussianComponentsDiag
from gaussian_components_fixedvar import GaussianComponentsFixedVar
import _cython_utils
import utils

logger = logging.getLogger(__name__)


#-----------------------------------------------------------------------------#
#                                 FBGMM CLASS                                 #
#-----------------------------------------------------------------------------#

class FBGMM(object):
    """
    A finite Bayesian Gaussian mixture model (FBGMM).

    See `GaussianComponents` or `GaussianComponentsDiag` for an overview of the
    parameters not mentioned below.

    Parameters
    ----------
    alpha : float
        Concentration parameter for the symmetric Dirichlet prior over the
        mixture weights.
    K : int
        The number of mixture components. This is actually a maximum number,
        and it is possible to empty out some of these components.
    assignments : vector of int or str
        If vector of int, this gives the initial component assignments. The
        vector should therefore have N entries between 0 and `K`. Values of
        -1 is also allowed, indicating that the data vector does not belong to
        any component. Alternatively, `assignments` can take one of the
        following values:
        - "rand": Vectors are assigned randomly to one of `K` components.
        - "each-in-own": Each vector is assigned to a component of its own.
    covariance_type : str
        String describing the type of covariance parameters to use. Must be
        one of "full", "diag" or "fixed".
    lms : float
        Language model scaling factor.
    """

    def __init__(self, X, prior, alpha, K, assignments="rand",
            covariance_type="full", lms=1.0):
        self.alpha = alpha
        self.prior = prior
        self.covariance_type = covariance_type
        self.lms = lms

        self.setup_components(K, assignments, X)

        # N, D = X.shape

        # # Initial component assignments
        # if assignments == "rand":
        #     assignments = np.random.randint(0, K, N)

        #     # Make sure we have consequetive values
        #     for k in xrange(assignments.max()):
        #         while len(np.nonzero(assignments == k)[0]) == 0:
        #             assignments[np.where(assignments > k)] -= 1
        #         if assignments.max() == k:
        #             break
        # elif assignments == "each-in-own":
        #     assignments = np.arange(N)
        # else:
        #     # `assignments` is a vector
        #     pass

        # if covariance_type == "full":
        #     self.components = GaussianComponents(X, prior, assignments, K_max=K)
        # elif covariance_type == "diag":
        #     self.components = GaussianComponentsDiag(X, prior, assignments, K_max=K)
        # elif covariance_type == "fixed":
        #     self.components = GaussianComponentsFixedVar(X, prior, assignments, K_max=K)
        # else:
        #     assert False, "Invalid covariance type."

    def setup_components(self, K, assignments="rand", X=None):
        """
        Setup the `components` attribute.

        See parameters of `FBGMM` for parameters not described below. This
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
            self.components = GaussianComponentsFixedVar(X, self.prior, assignments, K_max=K)
        else:
            assert False, "Invalid covariance type."

    def set_K(self, K, reassign=True):
        """
        Set the number of components `K`.

        The `K` largest existing components are kept, and the rest of the data
        vectors are re-assigned to one of these (if `reassign` is True).
        """

        if self.components.K <= K:
            # The active components are already less than the new K
            self.components.K_max = K
            return

        sizes = self.components.counts
        old_assignments = self.components.assignments

        # Keep only the `K` biggest assignments
        assignments_to_keep = list(np.argsort(sizes)[-K:])
        new_assignments = [
            i if i in assignments_to_keep else -1 for i in old_assignments
            ]
        mapping = dict([(assignments_to_keep[i], i) for i in range(K)])
        mapping[-1] = -1
        new_assignments = np.array([mapping[i] for i in new_assignments])

        # Make sure we have consequetive assignment values
        for k in xrange(new_assignments.max()):
            while len(np.nonzero(new_assignments == k)[0]) == 0:
                new_assignments[np.where(new_assignments > k)] -= 1
            if new_assignments.max() == k:
                break

        # Create new `components` attribute
        self.setup_components(K, list(new_assignments))

        # Now add back those vectors which were assigned before but are unassigned now
        if reassign:
            for i, old_assignment in enumerate(old_assignments):
                new_assignment = new_assignments[i]
                if old_assignment == -1 or new_assignment != -1:
                    continue
                self.gibbs_sample_inside_loop_i(i)

        # sizes = self.components.counts
        # cur_assignments = self.components.assignments
        # print cur_assignments[:100]

        # # Keep only the `K` biggest assignments
        # assignments_to_keep = list(np.argsort(sizes)[-K:])
        # print assignments_to_keep
        # new_assignments = [
        #     i if i in assignments_to_keep else -1 if i == -1 else
        #     random.choice(assignments_to_keep) for i in cur_assignments
        #     ]
        # mapping = dict([(assignments_to_keep[i], i) for i in range(K)])
        # print mapping
        # mapping[-1] = -1
        # new_assignments = np.array([mapping[i] for i in new_assignments])
        # print new_assignments[:100]

        # # Make sure we have consequetive assignment values
        # for k in xrange(new_assignments.max()):
        #     while len(np.nonzero(new_assignments == k)[0]) == 0:
        #         new_assignments[np.where(new_assignments > k)] -= 1
        #     if new_assignments.max() == k:
        #         break

        # self.setup_components(K, list(new_assignments))

    def log_prob_z(self):
        """
        Return the log marginal probability of component assignment P(z).

        See (24.24) in Murphy, p. 842.
        """
        log_prob_z = (
            gammaln(self.alpha)
            - gammaln(self.alpha + np.sum(self.components.counts))
            + np.sum(
                gammaln(
                    self.components.counts
                    + float(self.alpha)/self.components.K_max
                    )
                - gammaln(self.alpha/self.components.K_max)
                )
            )
        return log_prob_z

    def log_prob_X_given_z(self):
        """Return the log probability of data in each component p(X|z)."""
        return self.components.log_marg()

    def log_marg(self):
        """Return log marginal of data and component assignments: p(X, z)"""

        log_prob_z = self.log_prob_z()
        log_prob_X_given_z = self.log_prob_X_given_z()

        # # Log probability of component assignment, (24.24) in Murphy, p. 842
        # log_prob_z = (
        #     gammaln(self.alpha)
        #     - gammaln(self.alpha + np.sum(self.components.counts))
        #     + np.sum(
        #         gammaln(
        #             self.components.counts
        #             + float(self.alpha)/self.components.K_max
        #             )
        #         - gammaln(self.alpha/self.components.K_max)
        #         )
        #     )

        # # Log probability of data in each component
        # log_prob_X_given_z = self.components.log_marg()

        return log_prob_z + log_prob_X_given_z

    # @profile
    def log_marg_i(self, i): #, lms=1.):
        """
        Return the log marginal of the i'th data vector: p(x_i)

        Here it is assumed that x_i is not currently in the acoustic model,
        so the -1 term used in the denominator in (24.26) in Murphy, p. 843
        is dropped (since x_i is already not included in the counts).
        """
        assert i != -1

        # Compute log probability of `X[i]` belonging to each component
        # (24.26) in Murphy, p. 843
        log_prob_z = self.lms * (
            np.log(float(self.alpha)/self.components.K_max + self.components.counts)
            # - np.log(_cython_utils.sum_ints(self.components.counts) + self.alpha - 1.)
            - np.log(_cython_utils.sum_ints(self.components.counts) + self.alpha)
            )
        # log_prob_z = lms * (
        #     np.ones(self.components.K_max)*(
        #         np.log(float(self.alpha)/self.components.K_max + self.components.counts)
        #         - np.log(np.sum(self.components.counts) + self.alpha - 1.)
        #         )
        #     )
        # logger.info("log_prob_z: " + str(log_prob_z))

        # (24.23) in Murphy, p. 842
        log_prob_z[:self.components.K] += self.components.log_post_pred(i)
        # Empty (unactive) components
        log_prob_z[self.components.K:] += self.components.log_prior(i)
        return _cython_utils.logsumexp(log_prob_z)
        # return logsumexp(log_prob_z)

    def gibbs_sample(self, n_iter, consider_unassigned=True,
            anneal_schedule=None, anneal_start_temp_inv=0.1,
            anneal_end_temp_inv=1, n_anneal_steps=-1): #, lms=1.0):
        """
        Perform `n_iter` iterations Gibbs sampling on the FBGMM.

        Parameters
        ----------
        consider_unassigned : bool
            Whether unassigned vectors (-1 in `assignments`) should be
            considered during sampling.
        anneal_schedule : str
            Can be one of the following:
            - None: A constant temperature of `anneal_end_temp_inv` is used
              throughout; if `anneal_end_temp_inv` is left at default (1), then
              this is equivalent to not performing annealing.
            - "linear": Linearly take the inverse temperature from
              `anneal_start_temp_inv` to `anneal_end_temp_inv` in
              `n_anneal_steps`. If `n_anneal_steps` is -1 for this schedule,
              annealing is performed over all `n_iter` iterations.
            - "step": Piecewise schedule in which the inverse temperature is
              taken from `anneal_start_temp_inv` to `anneal_end_temp_inv` in
              `n_anneal_steps` steps (annealing will be performed over all
              `n_iter` iterations; it might be worth adding an additional
              variable for this case to allow the step schedule to stop early).

        Return
        ------
        record_dict : dict
            Contains several fields describing the sampling process. Each field
            is described by its key and statistics are given in a list which
            covers the Gibbs sampling iterations.
        """

        # Setup record dictionary
        record_dict = {}
        record_dict["sample_time"] = []
        start_time = time.time()
        record_dict["log_marg"] = []
        record_dict["log_prob_z"] = []
        record_dict["log_prob_X_given_z"] = []
        record_dict["anneal_temp"] = []
        record_dict["components"] = []

        # Setup annealing iterator
        if anneal_schedule is None:
            get_anneal_temp = iter([])
        elif anneal_schedule == "linear":
            if n_anneal_steps == -1:
                n_anneal_steps = n_iter
            anneal_list = 1./np.linspace(anneal_start_temp_inv, anneal_end_temp_inv, n_anneal_steps)
            get_anneal_temp = iter(anneal_list)
        elif anneal_schedule == "step":
            assert not n_anneal_steps == -1, (
                "`n_anneal_steps` of -1 not allowed for step annealing schedule"
                )
            n_iter_per_step = int(round(float(n_iter)/n_anneal_steps))
            anneal_list = np.linspace(anneal_start_temp_inv, anneal_end_temp_inv, n_anneal_steps)
            anneal_list = 1./anneal_list
            anneal_list = np.repeat(anneal_list, n_iter_per_step)
            get_anneal_temp = iter(anneal_list)

        # Loop over iterations
        for i_iter in range(n_iter):

            # Get anneal temperature
            anneal_temp = next(get_anneal_temp, anneal_end_temp_inv)

            # Loop over data items
            for i in xrange(self.components.N):

                # Cache some old values for possible future use
                k_old = self.components.assignments[i]
                if not consider_unassigned and k_old == -1:
                    continue
                K_old = self.components.K
                stats_old = self.components.cache_component_stats(k_old)

                # Remove data vector `X[i]` from its current component
                self.components.del_item(i)

                # Compute log probability of `X[i]` belonging to each component
                # (24.26) in Murphy, p. 843
                log_prob_z = self.lms * (
                    np.ones(self.components.K_max)*np.log(
                        float(self.alpha)/self.components.K_max + self.components.counts
                        )
                    )
                # (24.23) in Murphy, p. 842
                log_prob_z[:self.components.K] += self.components.log_post_pred(i)
                # Empty (unactive) components
                log_prob_z[self.components.K:] += self.components.log_prior(i)
                if anneal_temp != 1:
                    log_prob_z = log_prob_z - logsumexp(log_prob_z)
                    log_prob_z_anneal = 1./anneal_temp * log_prob_z - logsumexp(1./anneal_temp * log_prob_z)
                    prob_z = np.exp(log_prob_z_anneal)
                else:
                    prob_z = np.exp(log_prob_z - logsumexp(log_prob_z))
                # prob_z = np.exp(log_prob_z - logsumexp(log_prob_z))

                # Sample the new component assignment for `X[i]`
                k = utils.draw(prob_z)

                # There could be several empty, unactive components at the end
                if k > self.components.K:
                    k = self.components.K
                # print prob_z, k, prob_z[k]

                # Add data item X[i] into its component `k`
                if k == k_old and self.components.K == K_old:
                    # Assignment same and no components have been removed
                    self.components.restore_component_from_stats(k_old, *stats_old)
                    self.components.assignments[i] = k_old
                else:
                    # Add data item X[i] into its new component `k`
                    self.components.add_item(i, k)

            # Update record
            record_dict["sample_time"].append(time.time() - start_time)
            start_time = time.time()
            record_dict["log_marg"].append(self.log_marg())
            record_dict["log_prob_z"].append(self.log_prob_z())
            record_dict["log_prob_X_given_z"].append(self.log_prob_X_given_z())
            record_dict["anneal_temp"].append(anneal_temp)
            record_dict["components"].append(self.components.K)

            # Log info
            info = "iteration: " + str(i_iter)
            for key in sorted(record_dict):
                info += ", " + key + ": " + str(record_dict[key][-1])
            logger.info(info)

        return record_dict

    def gibbs_sample_inside_loop_i(self, i, anneal_temp=1): #, lms=1.):
        """
        Perform the inside loop of Gibbs sampling for data vector `i`.

        This is the inside of `gibbs_sample` and can be used by outside objects
        to perform only the inside loop part of the Gibbs sampling operation.
        The step in the loop is sample a new assignment for data vector `i`.
        The reason for not replacing the actual inner part of `gibbs_sample` by
        a call to this function is because this won't allow for caching the old
        component stats.
        """

        # Compute log probability of `X[i]` belonging to each component
        # (24.26) in Murphy, p. 843
        log_prob_z = self.lms * (
            np.ones(self.components.K_max)*np.log(
                float(self.alpha)/self.components.K_max + self.components.counts
                )
            )

        # (24.23) in Murphy, p. 842
        log_prob_z[:self.components.K] += self.components.log_post_pred(i)
        # Empty (unactive) components
        log_prob_z[self.components.K:] += self.components.log_prior(i)
        if anneal_temp != 1:
            log_prob_z = log_prob_z - logsumexp(log_prob_z)
            log_prob_z_anneal = 1./anneal_temp * log_prob_z - logsumexp(1./anneal_temp * log_prob_z)
            prob_z = np.exp(log_prob_z_anneal)
        else:
            prob_z = np.exp(log_prob_z - logsumexp(log_prob_z))
        # prob_z = np.exp(log_prob_z - logsumexp(log_prob_z))
        assert not np.isnan(np.sum(prob_z))

        # Sample the new component assignment for `X[i]`
        k = utils.draw(prob_z)

        # There could be several empty, unactive components at the end
        if k > self.components.K:
            k = self.components.K

        logger.debug("Adding item " + str(i) + " to acoustic model component " + str(k))
        self.components.add_item(i, k)

    def map_assign_i(self, i):
        """
        Assign data vector `i` to the component giving the maximum posterior.

        This function is very similar to `gibbs_sample_inside_loop_i`, but
        instead of sampling the assignment, the MAP estimate is used.
        """

        # Compute log probability of `X[i]` belonging to each component
        # (24.26) in Murphy, p. 843
        log_prob_z = (
            np.ones(self.components.K_max)*np.log(
                float(self.alpha)/self.components.K_max + self.components.counts
                )
            )
        # (24.23) in Murphy, p. 842
        log_prob_z[:self.components.K] += self.components.log_post_pred(i)
        # Empty (unactive) components
        log_prob_z[self.components.K:] += self.components.log_prior(i)
        prob_z = np.exp(log_prob_z - logsumexp(log_prob_z))

        # Take the MAP assignment for `X[i]`
        k = np.argmax(prob_z)

        # There could be several empty, unactive components at the end
        if k > self.components.K:
            k = self.components.K

        logger.debug("Adding item " + str(i) + " to acoustic model component " + str(k))
        self.components.add_item(i, k)

    def get_n_assigned(self):
        """Return the number of assigned data vectors."""
        return len(np.where(self.components.assignments != -1)[0])


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():

    import random

    from niw import NIW

    logging.basicConfig(level=logging.INFO)

    random.seed(1)
    np.random.seed(1)

    # Data parameters
    D = 2           # dimensions
    N = 10          # number of points to generate
    K_true = 4      # the true number of components

    # Model parameters
    alpha = 1.
    K = 6           # number of components
    n_iter = 10

    # Generate data
    mu_scale = 4.0
    covar_scale = 0.7
    z_true = np.random.randint(0, K_true, N)
    mu = np.random.randn(D, K_true)*mu_scale
    X = mu[:, z_true] + np.random.randn(D, N)*covar_scale
    X = X.T

    # Intialize prior
    m_0 = np.zeros(D)
    k_0 = covar_scale**2/mu_scale**2
    v_0 = D + 3
    S_0 = covar_scale**2*v_0*np.eye(D)
    prior = NIW(m_0, k_0, v_0, S_0)

    # Setup FBGMM
    fmgmm = FBGMM(X, prior, alpha, K, "rand")

    # Perform Gibbs sampling
    logger.info("Initial log marginal prob: " + str(fmgmm.log_marg()))
    record = fmgmm.gibbs_sample(n_iter)


if __name__ == "__main__":
    main()
