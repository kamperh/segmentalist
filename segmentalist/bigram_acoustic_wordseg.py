"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2016
"""

# from scipy.misc import logsumexp
import logging
import math
import numpy as np
import numpy.testing as npt
import random
import time

from bigram_fbgmm import BigramFBGMM
from bigram_lms import BigramSmoothLM
from unigram_acoustic_wordseg import process_embeddings
from utterances import Utterances
import _cython_utils
import unigram_acoustic_wordseg
import utils

logger = logging.getLogger(__name__)
i_debug_monitor = 81  # 466  # the index of an utterance which is to be monitored
debug_gibbs_only = False  # only sample the debug utterance


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
    unigram_counts : Kx1 vector of int
        Counts for each of the K components.
    bigram_counts : KxK matrix of int
        Element (j, i) is the count N_i_given_j of the component i following
        the component j.
    """

    def __init__(self, am_K, am_param_prior, lm_params,
            embedding_mats, vec_ids_dict, durations_dict, landmarks_dict,
            seed_boundaries_dict=None, seed_assignments_dict=None,
            covariance_type="fixed", n_slices_min=0,
            n_slices_max=20, min_duration=0, p_boundary_init=0.5,
            beta_sent_boundary=2.0, lms=1., wip=0., fb_type="bigram",
            init_am_assignments="rand",
            time_power_term=1.):

        logger.info("Initializing")

        # Check parameters
        assert seed_assignments_dict is None or seed_boundaries_dict is not None

        # Initialize simple attributes
        self.n_slices_min = n_slices_min
        self.n_slices_max = n_slices_max
        self.beta_sent_boundary = beta_sent_boundary
        self.wip = wip
        self.lms = lms
        self.time_power_term = time_power_term
        self.set_fb_type(fb_type)

        # Process embeddings into a single matrix, and vec_ids into a list (entry for each utterance)
        embeddings, vec_ids, ids_to_utterance_labels = process_embeddings(embedding_mats, vec_ids_dict)
        self.ids_to_utterance_labels = ids_to_utterance_labels
        N = embeddings.shape[0]

        # Initialize `utterances`
        if seed_boundaries_dict is not None:
            seed_boundaries = [seed_boundaries_dict[i] for i in ids_to_utterance_labels]
        else:
            seed_boundaries = None
        lengths = [len(landmarks_dict[i]) for i in ids_to_utterance_labels]
        landmarks = [landmarks_dict[i] for i in ids_to_utterance_labels]
        durations = [durations_dict[i] for i in ids_to_utterance_labels]
        self.utterances = Utterances(
            lengths, vec_ids, durations, landmarks, seed_boundaries=seed_boundaries,
            p_boundary_init=p_boundary_init, n_slices_min=n_slices_min,
            n_slices_max=n_slices_max, min_duration=min_duration
            )

        # Find all the embeddings that are in the initial segmentation
        init_embeds = []
        for i in range(self.utterances.D):
            init_embeds.extend(self.utterances.get_segmented_embeds_i(i))
        init_embeds = np.array(init_embeds, dtype=int)
        init_embeds = init_embeds[np.where(init_embeds != -1)]

        # Setup language model
        if lm_params["type"] == "smooth":
            intrp_lambda = lm_params["intrp_lambda"]
            a = lm_params["a"]
            b = lm_params["b"]
            K = am_K
            self.lm = BigramSmoothLM(intrp_lambda, a, b, K)

        # Provide the initial acoustic model assignments and initialize the model accordingly
        assignments = -1*np.ones(N, dtype=int)
        if seed_assignments_dict is not None:

            # Use seed assignments if provided
            logger.info("Using seed assignments")
            self.seed_to_cluster = {}
            i_cluster = 0
            for i_utt, utt in enumerate(ids_to_utterance_labels):
                utt_init_embeds = np.array(self.utterances.get_segmented_embeds_i(i_utt), dtype=int)
                utt_init_assignments = np.array(seed_assignments_dict[utt][:])
                utt_init_assignments = utt_init_assignments[np.where(utt_init_embeds != -1)]
                utt_init_embeds = utt_init_embeds[np.where(utt_init_embeds != -1)]
                for seed in utt_init_assignments:
                    if not seed in self.seed_to_cluster:
                        if isinstance(seed, (int, long)):
                            self.seed_to_cluster[seed] = seed
                        else:
                            self.seed_to_cluster[seed] = i_cluster
                            i_cluster += 1
                utt_init_assignments = [self.seed_to_cluster[i] for i in utt_init_assignments]
                assignments[utt_init_embeds] = utt_init_assignments
            if am_K is None:
                am_K = max(self.seed_to_cluster.values()) + 1
            else:
                assert am_K >= max(self.seed_to_cluster.values()) + 1

            # Initialize `acoustic_model`
            self.acoustic_model = BigramFBGMM(
                embeddings, am_param_prior, am_K, assignments,
                covariance_type=covariance_type, lms=lms, lm=self.lm
                )           

        elif init_am_assignments == "rand":

            # Assign each of the above embeddings randomly to one of the `am_K` clusters
            logger.info("Using random initial component assignments")
            init_embeds_assignments = np.random.randint(0, am_K, len(init_embeds))
            # Make sure we have consecutive values
            for k in xrange(init_embeds_assignments.max()):
                while len(np.nonzero(init_embeds_assignments == k)[0]) == 0:
                    init_embeds_assignments[np.where(init_embeds_assignments > k)] -= 1
                if init_embeds_assignments.max() == k:
                    break
            assignments[init_embeds] = init_embeds_assignments

            # Initialize `acoustic_model`
            self.acoustic_model = BigramFBGMM(
                embeddings, am_param_prior, am_K, assignments,
                covariance_type=covariance_type, lms=lms, lm=self.lm
                )

        elif init_am_assignments == "one-by-one":
            assert False
            # # Initialize `acoustic_model`
            # logger.info("Using a one-by-one initial assignment")
            # self.acoustic_model = am_class(
            #     embeddings, am_param_prior, am_alpha, am_K, assignments,
            #     covariance_type=covariance_type, lms=lms
            #     )

            # # Assign the embeddings one-by-one
            # for i_embed in init_embeds:
            #     # print i_embed
            #     self.acoustic_model.gibbs_sample_inside_loop_i(i_embed)

        else:
            assert False, "invalid value for `init_am_assignments`: " + init_am_assignments

        # Setup initial language model counts
        self.set_lm_counts()

    def set_fb_type(self, fb_type):
        self.fb_type = fb_type

        # Assign forward-backward function
        if fb_type == "bigram":
            self.fb_func = forward_backward
            self.get_vec_embed_log_probs = self.get_vec_embed_log_probs_bigram
        elif fb_type == "unigram":
            self.fb_func = unigram_acoustic_wordseg.forward_backward
            self.get_vec_embed_log_probs = self.get_vec_embed_log_probs_unigram
        else:
            assert False, "invalid `fb_type`: " + fb_type

    def set_lm_counts(self):
        # K = self.acoustic_model.components.K_max
        # unigram_counts = np.zeros(K, np.int)
        # bigram_counts = np.zeros((K, K), np.int)
        for i_utt in xrange(self.utterances.D):
            self.lm.counts_from_utterance(self.get_unsup_transcript_i(i_utt))
            # print 
            # print i_utt, "-"*5, self.get_unsup_transcript_i(i_utt)
            # j_prev = None
            # for i_cur in self.get_unsup_transcript_i(i_utt):
            #     self.lm.unigram_counts[i_cur] += 1
            #     if j_prev is not None:
            #         self.lm.bigram_counts[j_prev, i_cur] += 1
            #     j_prev = i_cur
        # npt.assert_equal(self.acoustic_model.components.counts, self.lm.unigram_counts)

    def log_prob_z(self):
        """
        Return the log marginal probability of component assignment P(z).
        """
        lm_tmp = BigramSmoothLM(
            intrp_lambda=self.lm.intrp_lambda, a=self.lm.a, b=self.lm.b,
            K=self.lm.K
            )
        log_prob_z = 0.
        for i_utt in xrange(self.utterances.D):
            j_prev = None
            for i_cur in self.get_unsup_transcript_i(i_utt):
                if j_prev is not None:
                    log_prob_z += np.log(lm_tmp.prob_i_given_j(i_cur, j_prev))
                    lm_tmp.bigram_counts[j_prev, i_cur] += 1
                else:
                    log_prob_z += np.log(lm_tmp.prob_i(i_cur))
                lm_tmp.unigram_counts[i_cur] += 1
        return log_prob_z

    def log_marg(self):
        """Return log marginal of data and component assignments: p(X, z)"""
        log_prob_z = self.log_prob_z()
        log_prob_X_given_z = self.acoustic_model.log_prob_X_given_z()
        return log_prob_z + log_prob_X_given_z

    # @profile
    def log_marg_i_embed_unigram(self, i_embed):
        """Return the unigram log marginal of the i'th data vector: p(x_i)"""
        assert i_embed != -1

        # Compute log probability of `X[i]` belonging to each component
        # (24.26) in Murphy, p. 843
        log_prob_z = self.lms * self.lm.log_prob_vec_i()
        # logger.info("log_prob_z: " + str(log_prob_z))

        # (24.23) in Murphy, p. 842`
        log_prob_z[:self.acoustic_model.components.K] += self.acoustic_model.components.log_post_pred(
            i_embed
            )
        # Empty (unactive) components
        log_prob_z[self.acoustic_model.components.K:] += self.acoustic_model.components.log_prior(i_embed)
        return _cython_utils.logsumexp(log_prob_z)

    # @profile
    def gibbs_sample_inside_loop_i_embed(self, i_embed, j_prev_assignment=None, anneal_temp=1, i_utt=None):
        """
        Perform the inside loop of Gibbs sampling for data vector `i_embed`.
        """

        # Temp
        # print "j_prev_assignment", j_prev_assignment
        # print self.lm.unigram_counts
        # print self.lm.bigram_counts
        # print

        # Compute log probability of `X[i]` belonging to each component; this
        # is the bigram version of (24.26) in Murphy, p. 843.
        if j_prev_assignment is not None:
            log_prob_z = np.log(self.lm.prob_vec_given_j(j_prev_assignment))
        else:
            log_prob_z = self.lm.log_prob_vec_i()
        # print log_prob_z

        # Scale with language model scaling factor
        log_prob_z *= self.lms
        # print log_prob_z
        if i_utt is not None and i_utt == i_debug_monitor:
            logger.debug("lms * log(P(z=i|z_prev=j)): " + str(log_prob_z))
            logger.debug("log(p(x|z=i)): " + str(self.acoustic_model.components.log_post_pred(i_embed)))

        # Bigram version of (24.23) in Murphy, p. 842
        log_prob_z[:self.acoustic_model.components.K] += self.acoustic_model.components.log_post_pred(i_embed)
        # Empty (unactive) components
        log_prob_z[self.acoustic_model.components.K:] += self.acoustic_model.components.log_prior(i_embed)
        if anneal_temp != 1:
            log_prob_z = log_prob_z - _cython_utils.logsumexp(log_prob_z)
            log_prob_z_anneal = 1./anneal_temp * log_prob_z - _cython_utils.logsumexp(1./anneal_temp * log_prob_z)
            prob_z = np.exp(log_prob_z_anneal)
        else:
            prob_z = np.exp(log_prob_z - _cython_utils.logsumexp(log_prob_z))
        assert not np.isnan(np.sum(prob_z))

        if i_utt is not None and i_utt == i_debug_monitor:
            logger.debug("P(z=i|x): " + str(prob_z))

        # Sample the new component assignment for `X[i]`
        k = utils.draw(prob_z)

        # There could be several empty, unactive components at the end
        if k > self.acoustic_model.components.K:
            k = self.acoustic_model.components.K

        if i_utt is not None and i_utt == i_debug_monitor:
            logger.debug("Adding item " + str(i_embed) + " to acoustic model component " + str(k))
        self.acoustic_model.components.add_item(i_embed, k)

        return k

    def gibbs_sample_i(self, i, anneal_temp=1, anneal_gibbs_am=False,
            assignments_only=False):
        """
        Block Gibbs sample new boundaries and embedding assignments for
        utterance `i`.

        Return
        ------
        log_prob : float
        """

        # # Temp
        # print i, self.ids_to_utterance_labels[i], str(self.get_unsup_transcript_i(i))

        # Debug trace
        logger.debug("Gibbs sampling utterance: " + str(i))
        if i == i_debug_monitor:
            logger.debug("-"*39)
            logger.debug("log p(X) before sampling: " + str(self.log_marg()))
            logger.debug("Unsupervised transcript before sampling: " + str(self.get_unsup_transcript_i(i)))
            logger.debug("Unigram counts before sampling: " + str(self.lm.unigram_counts))
            logger.debug("Bigram counts before sampling: " + str(self.lm.bigram_counts))

        # Remove counts from the `lm`
        self.lm.remove_counts_from_utterance(self.get_unsup_transcript_i(i))

        # Remove embeddings from utterance `i` from the `acoustic_model`
        for i_embed in self.utterances.get_segmented_embeds_i(i):
            if i_embed == -1:
                continue  # don't remove a non-embedding (would accidently remove the last embedding)
            self.acoustic_model.components.del_item(i_embed)

        # Sample segmentation
        if not assignments_only:

            # Get the log probabilities of the embeddings
            N = self.utterances.lengths[i]
            vec_embed_log_probs = self.get_vec_embed_log_probs(
                self.utterances.vec_ids[i, :(N**2 + N)/2],
                self.utterances.durations[i, :(N**2 + N)/2]
                )
            # assert False, "vec_embed_log_probs should be calculated differently based on unigram or bigram segmentation" 

            # Debug trace
            if i == i_debug_monitor:
                logger.debug("Statistics before sampling, but after removing, is given below")
                if self.fb_type == "unigram":
                    log_margs = [
                        self.log_marg_i_embed_unigram(j) for j in
                        self.utterances.get_segmented_embeds_i(i) if j != -1
                        ]
                else:
                    assert False, "to-do"
                embeddings = self.utterances.get_segmented_embeds_i(i)
                lengths = self.utterances.get_segmented_durations_i(i)
                logger.debug("Embeddings: " + str(embeddings))
                logger.debug("Utterance embeddings: " + str(self.utterances.get_original_segmented_embeds_i(i)))
                logger.debug("Landmark indices: " + str(self.utterances.get_segmented_landmark_indices(i)))
                logger.debug("Durations: " + str(self.utterances.get_segmented_durations_i(i)))
                logger.debug("log_margs: " + str(log_margs))
                logger.debug("sum(log_margs*lengths): " + str(np.sum(log_margs*np.array(lengths))))
                logger.debug("log p(X): " + str(self.log_marg()))

            # Draw new boundaries for utterance `i`
            log_p_continue = math.log(self.calc_p_continue())
            log_prob, self.utterances.boundaries[i, :N] = self.fb_func(
                vec_embed_log_probs, log_p_continue, N, self.n_slices_min, self.n_slices_max, i, anneal_temp
                )

            # Debug trace
            if i == i_debug_monitor:
                logger.debug("Statistics after sampling, but before adding new embeddings to `acoustic_model`")
                if self.fb_type == "unigram":
                    log_margs = [
                        self.log_marg_i_embed_unigram(j) for j in
                        self.utterances.get_segmented_embeds_i(i) if j != -1
                        ]
                else:
                    assert False, "to-do"
                lengths = self.utterances.get_segmented_durations_i(i)
                logger.debug("Embeddings: " + str(self.utterances.get_segmented_embeds_i(i)))
                logger.debug("Utterance embeddings: " + str(self.utterances.get_original_segmented_embeds_i(i)))
                logger.debug("Landmark indices: " + str(self.utterances.get_segmented_landmark_indices(i)))
                logger.debug("Durations: " + str(self.utterances.get_segmented_durations_i(i)))
                logger.debug("log_margs: " + str(log_margs))
                logger.debug("sum(log_margs*lengths): " + str(np.sum(log_margs*np.array(lengths))))
                logger.debug("log p(X): " + str(self.log_marg()))

        # # Temp
        # print self.lm.unigram_counts
        # print self.lm.bigram_counts
        # print

        # Assign new embeddings to components in `acoustic_model`
        if i == i_debug_monitor:
            logger.debug("Sampling component assignments")
        j_prev_assignment = None
        for i_embed in self.utterances.get_segmented_embeds_i(i):
            if i_embed == -1:
                # This only happens because of backtracking in the forward-backward functions
                continue  # don't assign a non-embedding (accidently the last embedding)
            if anneal_gibbs_am:
                anneal_temp = anneal_temp
            else:
                anneal_temp = 1

            j_prev_assignment = self.gibbs_sample_inside_loop_i_embed(
                i_embed, j_prev_assignment, anneal_temp=anneal_temp, i_utt=i
                )

        self.lm.counts_from_utterance(self.get_unsup_transcript_i(i))

        # logger.info("!!!")
        # logger.info(str(self.lm.unigram_counts))
        # logger.info(str(self.acoustic_model.components.counts))
        # logger.info(str(self.lm.bigram_counts))
        # logger.info("!!!")

        # print "!!!", self.lm.unigram_counts
        # print self.acoustic_model.components.counts
        # print "bigram_counts", self.lm.bigram_counts

        # npt.assert_equal(self.acoustic_model.components.counts, self.lm.unigram_counts)

        # import copy
        # lm = copy.copy(self.lm)
        # lm.unigram_counts.fill(0.0)
        # lm.bigram_counts.fill(0.0)
        # for i_utt in xrange(self.utterances.D):
        #     lm.counts_from_utterance(self.get_unsup_transcript_i(i_utt))
        # npt.assert_equal(lm.unigram_counts, self.lm.unigram_counts)
        # npt.assert_equal(lm.bigram_counts, self.lm.bigram_counts)
        # assert False

            # print self.lm.unigram_counts
            # print self.acoustic_model.components.lm.unigram_counts
            # print self.acoustic_model.components.counts
            # print self.lm.bigram_counts
        # assert False

        # Temp
        # print self.utterances.get_segmented_embeds_i(i)
        # print self.get_unsup_transcript_i(i)

        # Update `lm` counts
        # self.lm.counts_from_utterance(self.get_unsup_transcript_i(i))
        # assert False

        # # # Temp
        # print self.lm.unigram_counts
        # print self.lm.bigram_counts
        # print self.acoustic_model.components.lm.unigram_counts

        # Debug trace
        if i == i_debug_monitor:
            logger.debug("log p(X) after sampling: " + str(self.log_marg()))
            logger.debug("Unsupervised transcript after sampling: " + str(self.get_unsup_transcript_i(i)))
            logger.debug("Unigram counts after sampling: " + str(self.lm.unigram_counts))
            logger.debug("Bigram counts after sampling: " + str(self.lm.bigram_counts))
            logger.debug("-"*39)

        if assignments_only:
            # Segmentation is not performed, so frame-scaled marginals does not make gibbs_sample_inside_loop_i_embed
            return 0.
        else:
            return log_prob

    def gibbs_sample(self, n_iter, am_n_iter=0, anneal_schedule=None,
            anneal_start_temp_inv=0.1, anneal_end_temp_inv=1,
            n_anneal_steps=-1, anneal_gibbs_am=False, assignments_only=False):
        """
        Perform blocked Gibbs sampling on all utterances.

        Parameters
        ----------
        n_iter : int
            Number of Gibbs sampling iterations of segmentation.
        am_n_iter : int
            Number of acoustic model Gibbs sampling iterations inbetween
            segmentation sampling iterations.
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
        assignments_only : bool
            Whether only component assignments should be sampled, or whether
            both component assignment and segmentation should be performed.

        Return
        ------
        record_dict : dict
            Contains several fields describing the sampling process. Each field
            is described by its key and statistics are given in a list which
            covers the Gibbs sampling iterations.
        """

        logger.info("Gibbs sampling for " + str(n_iter) + " iterations")
        logger.debug(
            "Monitoring utterance " + self.ids_to_utterance_labels[i_debug_monitor]
            + " (index=" + str(i_debug_monitor) + ")"
            )

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
            # anneal_list = [100.0, 10.0, 3.0, 1.0, 0.1]
            anneal_list = np.repeat(anneal_list, n_iter_per_step)
            get_anneal_temp = iter(anneal_list)

        # Setup record dictionary
        record_dict = {}
        record_dict["sample_time"] = []
        record_dict["log_marg"] = []
        record_dict["log_marg*length"] = []
        record_dict["log_prob_z"] = []
        record_dict["log_prob_X_given_z"] = []
        record_dict["anneal_temp"] = []
        record_dict["components"] = []
        record_dict["n_tokens"] = []

        # Loop over sampling iterations
        for i_iter in xrange(n_iter):

            start_time = time.time()

            # Perform intermediate acoustic model re-sampling
            if am_n_iter > 0:
                assert False, "to-do"
                self.acoustic_model.gibbs_sample(
                    am_n_iter, consider_unassigned=False
                    )

            # Get anneal temperature
            anneal_temp = next(get_anneal_temp, anneal_end_temp_inv)

            # Loop over utterances
            utt_order = range(self.utterances.D)
            random.shuffle(utt_order)
            if debug_gibbs_only:
                utt_order = [i_debug_monitor]
            log_prob = 0
            for i_utt in utt_order:
                log_prob += self.gibbs_sample_i(i_utt, anneal_temp, anneal_gibbs_am, assignments_only)

            record_dict["sample_time"].append(time.time() - start_time)
            start_time = time.time()
            record_dict["log_marg"].append(self.log_marg())
            record_dict["log_marg*length"].append(log_prob)
            record_dict["log_prob_z"].append(self.log_prob_z())
            record_dict["log_prob_X_given_z"].append(self.acoustic_model.log_prob_X_given_z())
            record_dict["anneal_temp"].append(anneal_temp)
            record_dict["components"].append(self.acoustic_model.components.K)
            record_dict["n_tokens"].append(self.acoustic_model.get_n_assigned())

            info = "iteration: " + str(i_iter)
            for key in sorted(record_dict):
                info += ", " + key + ": " + str(record_dict[key][-1])
            logger.info(info)

            logger.debug("Unigram counts after inference: " + str(self.lm.unigram_counts))
            logger.debug("Bigram counts after inference: " + str(self.lm.bigram_counts))

        return record_dict

    # @profile
    def get_vec_embed_log_probs_unigram(self, vec_ids, durations):
        """
        Return the unigram log marginal probs of the `vec_ids` embeddings,
        scaled by the given `durations`.
        """

        # Get marginals
        vec_embed_log_probs = -np.inf*np.ones(len(vec_ids))
        for i, embed_id in enumerate(vec_ids):
            if embed_id == -1:
                continue
            vec_embed_log_probs[i] = self.log_marg_i_embed_unigram(embed_id)

            # Scale log marginals by number of frames
            if np.isnan(durations[i]):
                vec_embed_log_probs[i] = -np.inf
            else:
                vec_embed_log_probs[i] *= durations[i]**self.time_power_term

        return vec_embed_log_probs + self.wip

    def get_vec_embed_log_probs_bigram(self, vec_ids, durations):
        pass

    def calc_p_continue(self):
        """
        Return the probability of not having an utterance break.

        It is assumed that the number of utterances are one less than the total
        number, since the current utterance is excluded from the calculation.
        """
        if self.beta_sent_boundary != -1:
            assert False, "to check"
            n_tokens = sum(self.acoustic_model.components.counts)  # number of assigned tokens
            n_sentences = self.utterances.D - 1
            n_continue = n_tokens - n_sentences
            p_continue = (
                (n_continue + self.beta_sent_boundary / 2.0) /
                (n_tokens + self.beta_sent_boundary)
                )
        else:
            p_continue = 1.0
        return p_continue

    def get_unsup_transcript_i(self, i):
        """Return a list of the components for current segmentation of `i`."""
        return list(
            self.acoustic_model.components.get_assignments(self.utterances.get_segmented_embeds_i(i))
            )


#-----------------------------------------------------------------------------#
#                     FORWARD-BACKWARD INFERENCE FUNCTIONS                    #
#-----------------------------------------------------------------------------#

def forward_backward(vec_embed_log_probs, log_p_continue, N, n_slices_min=0,
        n_slices_max=0, i_utt=None, anneal_temp=1):
    """
    Segment an utterance of length `N` based on its `vec_embed_log_probs`
    vector and return a bool vector of boundaries.

    Parameters
    ----------
    vec_embed_log_probs : N(N + 1)/2 length vector
        For t = 1, 2, ..., N the entries `vec_embed_log_probs[i:i + t]`
        contains the log probabilties of sequence[0:t] up to sequence[t - 1:t],
        with i = t(t - 1)/2. If you have a NxN matrix where the upper
        triangular (i, j)'th entry is the log probability of sequence[i:j + 1],
        then by stacking the upper triangular terms column-wise, you get
        vec_embed_log_probs`. Written out: `vec_embed_log_probs` =
        [log_prob(seq[0:1]), log_prob(seq[0:2]), log_prob(seq[1:2]),
        log_prob(seq[0:3]), ..., log_prob(seq[N-1:N])].
    n_slices_max : int
        See `UnigramAcousticWordseg`. If 0, then the full length are
        considered. This won't necessarily lead to problems, since unassigned
        embeddings would still be ignored since their assignments are -1 and
        the would therefore have a log probability of -inf.

    Return
    ------
    (log_prob, boundaries) : (float, vector of bool)
        The `log_prob` is the sum of the log probabilties in
        `vec_embed_log_probs` for the embeddings for the final segmentation.
    """

    pass


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():

    import gaussian_components_fixedvar

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
    i_debug_monitor = 1

    # Data parameters
    D = 3  # dimensionality of embeddings

    # Acoustic model parameters
    am_K = 3
    m_0 = np.zeros(D)
    k_0 = 0.05
    S_0 = 0.002*np.ones(D)
    am_param_prior = gaussian_components_fixedvar.FixedVarPrior(S_0, m_0, S_0/k_0)
    lm_params = {
        "type": "smooth",
        "intrp_lambda": 0,
        "a": 0.5,
        "b": 0.5
        }

    # Initialize model
    segmenter = BigramAcousticWordseg(
        am_K, am_param_prior, lm_params, embedding_mats, vec_ids_dict,
        durations_dict, landmarks_dict, p_boundary_init=0.9,
        beta_sent_boundary=-1, n_slices_max=2, fb_type="unigram", lms=1.0
        )

    logger.info("Transcriptions before sampling:")
    for i_utt in xrange(segmenter.utterances.D):
        logger.info(str(i_utt) + ": " + str(segmenter.get_unsup_transcript_i(i_utt)))

    # Perform sampling
    segmenter.gibbs_sample(5, assignments_only=False)

    logger.info("Transcriptions after sampling:")
    for i_utt in xrange(segmenter.utterances.D):
        logger.info(str(i_utt) + ": " + str(segmenter.get_unsup_transcript_i(i_utt)))


if __name__ == "__main__":
    main()
