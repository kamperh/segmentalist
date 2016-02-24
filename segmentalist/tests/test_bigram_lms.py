"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2016
"""

from segmentalist.bigram_lms import BigramSmoothLM

def test_bigram_smooth_lm():
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


def test_bigram_smooth_lm_vecs():
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
    prob_vec_i = lm.prob_vec_i()
    for i in range(5):
        assert prob_vec_i[i] == lm.prob_i(i)
    j = 3
    prob_vec_given_j = lm.prob_vec_given_j(j)
    for i in range(5):
        assert prob_vec_given_j[i] == lm.prob_i_given_j(i, j)
