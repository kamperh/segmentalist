"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2016
"""

import logging
import math
import numpy as np
import random
import time

logger = logging.getLogger(__name__)


#-----------------------------------------------------------------------------#
#                        BIGRAM ACOUSTIC WORDSEG CLASS                        #
#-----------------------------------------------------------------------------#

class BigramAcousticWordseg(object):
    """
    Unigram word segmentation of speech using acoustic word embeddings.

    Segmentation and sampling operations are carried out in this class.
    Segmentation results are mainly stored in `utterances`, which deals with
    all utterance-level information, but knows nothing about the acoustics. The
    `acoustic_model` deals with all the acoustic embedding operations. Blocked
    Gibbs sampling is used for inference. In the member functions, the index
    `i` generally refers to the index of an utterance.

    Parameters
    ----------
    am_K : int
        Acoustic model parameter.
    am_param_prior : e.g. instance of `FixedVarPrior`
        The acoustic model prior on the mean and covariance parameters.
    lm_params : dict
        A dictionary with at least an entry for "type", which can be
        "maxlikelihood", and the other entries giving the hyperparameters for
        that particular kind of language model.
    embedding_mats : dict of matrix
        The matrices of embeddings for every utterance.
    vec_ids_dict : dict of vector of int
        For every utterance, the vector IDs (see `Utterances`).
    landmarks_dict : dict of list of int
        For every utterance, the landmark points at which word boundaries are
        considered, given in the number of frames (10 ms units) from the start
        of each utterance. There is an implicit landmark at the start of every
        utterance.
    durations_dict : dict of vector of int
        The shape of this dict is the same as that of `vec_ids_dict`, but here
        the duration (in frames) of each of the embeddings are given.
    seed_boundaries_dict : dict of list of tuple
        Every tuple is the start (inclusive) and end (exclusive) embedding
        slice index of a seed token, giving its boundaries. If not given, no
        seeding is used.
    seed_assignments_dict : dict of list of int
        Every int is a cluster assignment for the corresponding seed token in
        `seed_boundaries_dict`. If not given, no seeding is used.
    seed_boundaries_dict : dict of list of int
        For every utterance, seed boundaries in 10 ms units (same format as
        `landmarks_dict`). If not given, no seeding is used.
    seed_assignments_dict : dict of list of int
        Every int is a cluster assignment for the corresponding seed token in
        `seed_boundaries_dict`. If not given, no seeding is used.
    n_slices_min : int
        The minimum number of landmarks over which an embedding can be
        calculated.
    n_slices_max : int
        The maximum number of landmarks over which an embedding can be
        calculated.
    min_duration : int
        Minimum duration of a segment.
    p_boundary_init : float
        See `Utterances`.
    beta_sent_boundary : float
        The symmetric Beta prior on the end of sentence probability; if this is
        set to -1, sentence boundary probabilities are not taken into account.
    lms : float
        Language model scaling factor.
    wip : float
        Word insertion penalty.
    fb_type : str
        The type of forward-backward algorithm to use:
        - "unigram": In this case, segmentation is carried out as it is done in
          the unigram case; i.e. only assignments are sampled using the bigram
          model.
        - "bigram": Sample assignments using the bigram language model.
    init_am_assignments : str
        This setting determines how the initial acoustic model assignments are
        determined:
        - "rand": Randomly assigned.
        - "one-by-one": Data vectors are added one at a time to the acoustic
          model.
    time_power_term : float
        Scaling the per-frame scaling; with 1.2 instead of 1, we get less words
        (prefer longer words).

    Attributes
    ----------
    utterances : Utterances
        Knows nothing about the acoustics. The indices in the `vec_ids`
        attribute refers to the embedding at the corresponding row in
        `acoustic_model.components.X`.
    acoustic_model : BigramFBGMM
        Knows nothing about utterance-level information. All embeddings are
        stored in this class as the data `components.X` attribute.
    ids_to_utterance_labels : list of str
        Keeps track of utterance labels for a specific utterance ID.
    """

    def __init__(self, am_K, am_param_prior, lm_params,
            embedding_mats, vec_ids_dict, durations_dict, landmarks_dict,
            seed_boundaries_dict=None, seed_assignments_dict=None,
            covariance_type="fixed", n_slices_min=0,
            n_slices_max=20, min_duration=0, p_boundary_init=0.5,
            beta_sent_boundary=2.0, lms=1., wip=0., fb_type="standard",
            init_am_assignments="rand",
            time_power_term=1.):

        logger.info("Initializing")

        # Check parameters
        assert seed_assignments_dict is None or seed_boundaries_dict is not None


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():

    logging.basicConfig(level=logging.DEBUG)

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

    # Vector IDs
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
    logger.info("vec_ids: " + str(vec_ids))

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

    random.seed(1)
    np.random.seed(1)

    global i_debug_monitor
    i_debug_monitor = 0


if __name__ == "__main__":
    main()
