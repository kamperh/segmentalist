"""
Bigram language model classes.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2016
"""

import numpy as np
import _cython_utils


#-----------------------------------------------------------------------------#
#           BIGRAM SMOOTHED MAXIMUM LIKELIHOOD LANGUAGE MODEL CLASS           #
#-----------------------------------------------------------------------------#

class BigramSmoothLM(object):
    """
    A smoothed, interpolated maximum likelihood bigram language model.

    Parameters
    ----------
    intrp_lambda : float
        With a value of 0, no interpolation is performed.
    a : float
        Unigram smoothing parameter.
    b : float
        Bigram smoothing parameter.
    K : int

    Attributes
    ----------
    unigram_counts : Kx1 vector of int
        Counts for each of the K components.
    bigram_counts : KxK matrix of int
        Element (j, i) is the count N_i_given_j of the component i following
        the component j.
    """

    def __init__(self, intrp_lambda, a, b, K):
        self.intrp_lambda = intrp_lambda
        self.a = a
        self.b = b
        self.K = K

        self.unigram_counts = np.zeros(K, np.int)
        self.bigram_counts = np.zeros((K, K), np.int)

    def prob_i(self, i):
        """The unigram probability of observing `i`."""
        return (
            (self.unigram_counts[i] + float(self.a)/self.K) / 
            (_cython_utils.sum_ints(self.unigram_counts) + self.a)
            )

    def prob_i_given_j(self, i, j):
        """The conditional bigram probability of observing `i` given `j`."""
        prob_i_ = self.prob_i(i)
        prob_i_given_j_ = (
            (self.bigram_counts[j, i] + float(self.b)/self.K) / (self.unigram_counts[j] + float(self.b))
            )
        return self.intrp_lambda * prob_i_ + (1 - self.intrp_lambda) * prob_i_given_j_

    def log_prob_vec_i(self):
        """Return a vector of the log unigram probabilities."""
        return (
            np.log(self.unigram_counts + float(self.a)/self.K)
            - np.log(_cython_utils.sum_ints(self.unigram_counts) + self.a)
            )

    def prob_vec_i(self):
        """Return a vector of the unigram probabilities."""
        return (
            (self.unigram_counts + float(self.a)/self.K) /
            (_cython_utils.sum_ints(self.unigram_counts) + self.a)
            )

    def log_prob_vec_given_j(self, j):
        """
        Return a vector of the log conditional bigram probabilities given `j`.
        """
        return np.log(self.prob_vec_given_j(j))

    def prob_vec_given_j(self, j):
        """
        Return a vector of the conditional bigram probabilities given `j`.
        """
        return (
            self.intrp_lambda * self.prob_vec_i() + (1 - self.intrp_lambda) *
            (self.bigram_counts[j, :] + float(self.b)/self.K) / (self.unigram_counts[j] + float(self.b))
            )

    def counts_from_data(self, data):
        """Taking in a list of list of `data` and setting the counts."""
        for utterance in data:
            self.counts_from_utterance(utterance)

    def counts_from_utterance(self, utterance):
        """Update the counts according to the given utterance."""
        j_prev = None
        for i_cur in utterance:
            self.unigram_counts[i_cur] += 1
            if j_prev is not None:
                self.bigram_counts[j_prev, i_cur] += 1
            j_prev = i_cur

    def remove_counts_from_utterance(self, utterance):
        """Remove the counts according to the given utterance."""
        j_prev = None
        for i_cur in utterance:
            self.unigram_counts[i_cur] -= 1
            if j_prev is not None:
                self.bigram_counts[j_prev, i_cur] -= 1
            j_prev = i_cur


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    
    # Test BigramSmoothLM
    intrp_lambda = 0.1
    a = 1
    b = 2
    K = 5
    lm = BigramSmoothLM(intrp_lambda, a, b, K)

    data = [
        [1, 1, 3, 4, 0],
        [4, 4],
        [1, 0, 2, 2, 2, 2, 3, 1],
        [3, 3, 1]
        ]

    lm.counts_from_data(data)

    assert (
        lm.prob_i_given_j(1, 3) == intrp_lambda * lm.prob_i(1) + (1 -
        intrp_lambda) * (2. + b/K) / (4 + b)
        )
    assert lm.prob_i(1) == (5. + a/K) / (18 + a)

    prob_vec_i = lm.prob_vec_i()
    for i in range(5):
        assert prob_vec_i[i] == lm.prob_i(i)

    j = 3
    prob_vec_given_j = lm.prob_vec_given_j(j)
    for i in range(5):
        assert prob_vec_given_j[i] == lm.prob_i_given_j(i, j)


if __name__ == "__main__":
    main()
