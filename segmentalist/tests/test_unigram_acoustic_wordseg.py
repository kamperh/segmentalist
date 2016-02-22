"""
Author: Herman Kamper
Contact: h.kamper@sms.ed.ac.uk
Date: 2014
"""

import numpy as np
import numpy.testing as npt
import random

from segmentalist import unigram_acoustic_wordseg
from segmentalist import fbgmm
from segmentalist import gaussian_components_fixedvar


def setup_module():
    """
    Initialize a three-embedding test dataset.

    It has three slice positions (including the last).
    """
    global embedding_mats, vec_ids_dict, seed_bounds_dict, landmarks_dict, durations_dict

    # Embedding matrix
    embedding_mat = np.array([
        [-0.2702691 , -0.12348549, -0.20069546, -0.10067126, -0.32822475,
         -0.24878924, -0.17988801, -0.13201745,  0.66409844, -0.44816282],
        [-0.27186683, -0.12384345, -0.20049213, -0.10272419, -0.32618827,
         -0.24660945, -0.17784701, -0.13362537,  0.66524321, -0.44805479],
        [-0.2465426 , -0.06354388, -0.22458388,  0.79060942,  0.48230717,
         -0.11888564,  0.06724239, -0.04977163,  0.06908087,  0.03395205]], dtype=np.float32
        )

    # Vector IDs
    n_slices = embedding_mat.shape[0] - 1
    vec_ids = -1*np.ones((n_slices**2 + n_slices)/2, dtype=int)
    i_embed = 0
    n_slices_max = 20
    for cur_start in range(n_slices):
        for cur_end in range(cur_start, min(n_slices, cur_start + n_slices_max)):
            cur_end += 1
            t = cur_end
            i = t*(t - 1)/2
            vec_ids[i + cur_start] = i_embed
            # print cur_start, cur_end, i + cur_start, i_embed
            i_embed += 1

    embedding_mats = {}
    vec_ids_dict = {}
    seed_bounds_dict = {}
    durations_dict = {}
    landmarks_dict = {}
    embedding_mats["test"] = embedding_mat
    vec_ids_dict["test"] = vec_ids
    seed_bounds_dict["test"] = [2]
    landmarks_dict["test"] = [1, 2]
    durations_dict["test"] = [1, 2, 1]


def test_simple_vec_embed_log_probs():

    random.seed(1)
    np.random.seed(1)

    # Data parameters
    D = 10  # dimensionality of embeddings

    # Acoustic model parameters
    am_class = fbgmm.FBGMM
    am_alpha = 10.
    am_K = 2
    m_0 = np.zeros(D)
    k_0 = 0.05
    # S_0 = 0.025*np.ones(D)
    S_0 = 0.002*np.ones(D)
    am_param_prior = gaussian_components_fixedvar.FixedVarPrior(S_0, m_0, S_0/k_0)

    # Initialize model
    segmenter = unigram_acoustic_wordseg.UnigramAcousticWordseg(
        am_class, am_alpha, am_K, am_param_prior, embedding_mats, vec_ids_dict,
        durations_dict, landmarks_dict, seed_boundaries_dict=seed_bounds_dict, beta_sent_boundary=-1
        )

    segmenter.gibbs_sample_i(0)
    vec_embed_log_probs = segmenter.get_vec_embed_log_probs(
        segmenter.utterances.vec_ids[0], segmenter.utterances.durations[0]
        )
    expected_vec_embed_log_probs = np.array([17.64191118, 35.27798971, 17.64191119])

    npt.assert_almost_equal(vec_embed_log_probs, expected_vec_embed_log_probs)


def test_simple_sampling():
    """
    Simple sampling in unigram_acoustic_wordseg.

    Compares to output from a previous version of the code.
    """

    random.seed(1)
    np.random.seed(1)

    unigram_acoustic_wordseg.i_debug_monitor = 0

    # Data parameters
    D = 10  # dimensionality of embeddings

    # Acoustic model parameters
    am_class = fbgmm.FBGMM
    am_alpha = 10.
    am_K = 2
    m_0 = np.zeros(D)
    k_0 = 0.05
    # S_0 = 0.025*np.ones(D)
    S_0 = 0.002*np.ones(D)
    am_param_prior = gaussian_components_fixedvar.FixedVarPrior(S_0, m_0, S_0/k_0)

    # Initialize model
    segmenter = unigram_acoustic_wordseg.UnigramAcousticWordseg(
        am_class, am_alpha, am_K, am_param_prior, embedding_mats, vec_ids_dict,
        durations_dict, landmarks_dict, seed_boundaries_dict=seed_bounds_dict, beta_sent_boundary=-1
        )

    # Perform sampling
    record = segmenter.gibbs_sample(6)

    expected_log_marg = np.array([
        -11.969040866436707, -11.969040866436707, -11.969040866436707,
        -5.9368664797514707, -11.969040866436707, -5.9368664797514707
        ])
    expected_log_prob_z = np.array([
        -1.4816045409242173, -1.4816045409242173, -1.4816045409242173,
        -0.69314718055994673, -1.4816045409242173, -0.69314718055994673
        ])
    expected_log_prob_X_given_z = np.array([
        -10.48743632551249, -10.48743632551249, -10.48743632551249,
        -5.2437192991915236, -10.48743632551249, -5.2437192991915236
        ])

    npt.assert_almost_equal(record["log_marg"], expected_log_marg)
    npt.assert_almost_equal(record["log_prob_z"], expected_log_prob_z)
    npt.assert_almost_equal(record["log_prob_X_given_z"], expected_log_prob_X_given_z)


def test_simple_sampling2():
    """More severe `n_slices_max`."""

    # Test setup

    embedding_mat1 = np.array(
        [[ 1.55329044,  0.82568932,  0.56011276],
       [ 1.10640768, -0.41715366,  0.30323529],
       [ 1.24183824, -2.39021548,  0.02369367],
       [ 1.26094544, -0.27567053,  1.35731148],
       [ 1.59711416, -0.54917262, -0.56074459],
       [-0.4298405 ,  1.39010761, -1.2608597 ]], dtype=np.float32
        )
    embedding_mat2 = np.array(
        [[ 1.63075195,  0.25297823, -1.75406467],
       [-0.59324473,  0.96613426, -0.20922202],
       [ 0.97066059, -1.22315308, -0.37979187],
       [-0.31613254, -0.07262261, -1.04392799],
       [-1.11535652,  0.33905751,  1.85588856],
       [-1.08211738,  0.88559445,  0.2924617 ]], dtype=np.float32
        )

    n_slices = 3
    vec_ids = -1*np.ones((n_slices**2 + n_slices)/2, dtype=int)
    i_embed = 0
    n_slices_max = 20
    for cur_start in range(n_slices):
        for cur_end in range(cur_start, min(n_slices, cur_start + n_slices_max)):
            cur_end += 1
            t = cur_end
            i = t*(t - 1)/2
            vec_ids[i + cur_start] = i_embed
            # print cur_start, cur_end, i + cur_start, i_embed
            i_embed += 1

    embedding_mats = {}
    vec_ids_dict = {}
    seed_bounds_dict = {}
    durations_dict = {}
    landmarks_dict = {}
    embedding_mats["test1"] = embedding_mat1
    vec_ids_dict["test1"] = vec_ids
    seed_bounds_dict["test1"] = [2]
    landmarks_dict["test1"] = [1, 2, 3]
    durations_dict["test1"] = [1, 2, 1, 3, 2, 1]
    embedding_mats["test2"] = embedding_mat2
    vec_ids_dict["test2"] = vec_ids
    seed_bounds_dict["test2"] = [2]
    landmarks_dict["test2"] = [1, 2, 3]
    durations_dict["test2"] = [1, 2, 1, 3, 2, 1]

    # Run test

    random.seed(1)
    np.random.seed(1)

    unigram_acoustic_wordseg.i_debug_monitor = 0

    # Data parameters
    D = 3  # dimensionality of embeddings

    # Acoustic model parameters
    am_class = fbgmm.FBGMM
    am_alpha = 10.
    am_K = 2
    m_0 = np.zeros(D)
    k_0 = 0.05
    # S_0 = 0.025*np.ones(D)
    S_0 = 0.002*np.ones(D)
    am_param_prior = gaussian_components_fixedvar.FixedVarPrior(S_0, m_0, S_0/k_0)

    # Initialize model
    segmenter = unigram_acoustic_wordseg.UnigramAcousticWordseg(
        am_class, am_alpha, am_K, am_param_prior, embedding_mats, vec_ids_dict,
        durations_dict, landmarks_dict, p_boundary_init=0.5, beta_sent_boundary=-1, n_slices_max=2
        )

    # Perform sampling
    record = segmenter.gibbs_sample(3)

    expected_log_marg = np.array([-1520.885395538874, -435.84314783538349, -435.84314783538349])
    expected_log_prob_z = np.array([-3.641088790277589, -2.7937909298903829, -2.7937909298903829])
    expected_log_prob_X_given_z = np.array([-1517.2443067485965, -433.04935690549308, -433.04935690549308])

    npt.assert_almost_equal(record["log_marg"], expected_log_marg)
    npt.assert_almost_equal(record["log_prob_z"], expected_log_prob_z)
    npt.assert_almost_equal(record["log_prob_X_given_z"], expected_log_prob_X_given_z)
