"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2014-2016
"""

import logging
import math
import numpy as np
import random
import time

from utterances import Utterances
import _cython_utils

# TIME_POWER_TERM = 1.6  # with 1.2 instead of 1, we get less words (prefer longer words)

logger = logging.getLogger(__name__)
i_debug_monitor = 0  # 466  # the index of an utterance which is to be monitored
debug_gibbs_only = False  # only sample the debug utterance


#-----------------------------------------------------------------------------#
#                        UNIGRAM ACOUSTIC WORDSEG CLASS                       #
#-----------------------------------------------------------------------------#

class UnigramAcousticWordseg(object):
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
    am_class : e.g. `FBGMM`
    am_alpha : float
        Acoustic model parameter.
    am_K : int
        Acoustic model parameter.
    am_param_prior : e.g. instance of `FixedVarPrior`
        The acoustic model prior on the mean and covariance parameters.
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
        - "standard": The normal forward filtering backward sampling algorithm.
        - "viterbi": The Viterbi version of the forward backward algorithm,
          using MAP assignments instead of sampling segmentation of embedding
          component assignments.
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
    acoustic_model : FBGMM or IGMM
        Knows nothing about utterance-level information. All embeddings are
        stored in this class as the data `components.X` attribute.
    ids_to_utterance_labels : list of str
        Keeps track of utterance labels for a specific utterance ID.
    """

    def __init__(self, am_class, am_alpha, am_K, am_param_prior,
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

        # Initialize simple attributes
        self.n_slices_min = n_slices_min
        self.n_slices_max = n_slices_max
        self.beta_sent_boundary = beta_sent_boundary
        # self.lms = lms
        self.wip = wip
        self.time_power_term = time_power_term
        self.set_fb_type(fb_type)

        # Process embeddings into a single matrix, and vec_ids into a list (entry for each utterance)
        embeddings, vec_ids, ids_to_utterance_labels = process_embeddings(
            embedding_mats, vec_ids_dict#, n_slices_min=n_slices_min
            )
        self.ids_to_utterance_labels = ids_to_utterance_labels
        N = embeddings.shape[0]

        # lengths = [
        #     int(-1 + np.sqrt(1 + 4 * 2 * i)) / 2 for i in
        #     [len(vec_ids_dict[j]) for j in ids_to_utterance_labels]
        #     ]

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
            self.acoustic_model = am_class(
                embeddings, am_param_prior, am_alpha, am_K, assignments,
                covariance_type=covariance_type, lms=lms
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
            self.acoustic_model = am_class(
                embeddings, am_param_prior, am_alpha, am_K, assignments,
                covariance_type=covariance_type, lms=lms
                )

        elif init_am_assignments == "one-by-one":
            # Initialize `acoustic_model`
            logger.info("Using a one-by-one initial assignment")
            self.acoustic_model = am_class(
                embeddings, am_param_prior, am_alpha, am_K, assignments,
                covariance_type=covariance_type, lms=lms
                )

            # Assign the embeddings one-by-one
            for i_embed in init_embeds:
                # print i_embed
                self.acoustic_model.gibbs_sample_inside_loop_i(i_embed)

        else:
            assert False, "invalid value for `init_am_assignments`: " + init_am_assignments

    def set_fb_type(self, fb_type):
        self.fb_type = fb_type

        # Assign forward-backward function
        if fb_type == "standard":
            self.fb_func = forward_backward
        elif fb_type == "viterbi":
            self.fb_func = forward_backward_viterbi
        else:
            assert False, "invalid `fb_type`: " + fb_type

    def gibbs_sample_i(self, i, anneal_temp=1, anneal_gibbs_am=False):
        """
        Block Gibbs sample new boundaries and embedding assignments for
        utterance `i`.

        Return
        ------
        log_prob : float
        """

        # Debug trace
        logger.debug("Gibbs sampling utterance: " + str(i))
        if i == i_debug_monitor:
            logger.debug("-"*39)
            logger.debug("log p(X) before sampling: " + str(self.acoustic_model.log_marg()))
            logger.debug("Unsupervised transcript before sampling: " + str(self.get_unsup_transcript_i(i)))

        # Remove embeddings from utterance `i` from the `acoustic_model`
        for i_embed in self.utterances.get_segmented_embeds_i(i):
            if i_embed == -1:
                continue  # don't remove a non-embedding (would accidently remove the last embedding)
            self.acoustic_model.components.del_item(i_embed)

        # Get the log probabilities of the embeddings
        N = self.utterances.lengths[i]
        vec_embed_log_probs = self.get_vec_embed_log_probs(
            self.utterances.vec_ids[i, :(N**2 + N)/2],
            self.utterances.durations[i, :(N**2 + N)/2]
            )

        # Debug trace
        if i == i_debug_monitor:
            logger.debug("Statistics before sampling, but after removing, is given below")
            log_margs = [
                self.acoustic_model.log_marg_i(j) for j in
                self.utterances.get_segmented_embeds_i(i) if j != -1
                ]
            where_bounds = np.where(self.utterances.boundaries[i, :N])[0]
            embeddings = self.utterances.get_segmented_embeds_i(i)
            lengths = self.utterances.get_segmented_durations_i(i)
            # lengths = []
            # i_bound = -1
            # for embed, bound in zip(embeddings, where_bounds):
            #     if embed == -1:
            #         continue
            #     lengths.append(bound - i_bound)
            #     i_bound = bound
            # print lengths
            # print self.utterances.get_segmented_durations_i(i)
            logger.debug("Embeddings: " + str(embeddings))
            logger.debug("Utterance embeddings: " + str(self.utterances.get_original_segmented_embeds_i(i)))
            logger.debug("Landmark indices: " + str(self.utterances.get_segmented_landmark_indices(i)))
            logger.debug("Durations: " + str(self.utterances.get_segmented_durations_i(i)))
            logger.debug("log_margs: " + str(log_margs))
            logger.debug("sum(log_margs*lengths): " + str(np.sum(log_margs*np.array(lengths))))
            logger.debug("log p(X): " + str(self.acoustic_model.log_marg()))

        # Draw new boundaries for utterance `i`
        log_p_continue = math.log(self.calc_p_continue())
        log_prob, self.utterances.boundaries[i, :N] = self.fb_func(
            vec_embed_log_probs, log_p_continue, N, self.n_slices_min, self.n_slices_max, i, anneal_temp
            )

        # Debug trace
        if i == i_debug_monitor:
            logger.debug("Statistics after sampling, but before adding new embeddings to `acoustic_model`")
            log_margs = [
                self.acoustic_model.log_marg_i(j) for j in
                self.utterances.get_segmented_embeds_i(i) if j != -1
                ]
            where_bounds = np.where(self.utterances.boundaries[i, :N])[0]
            lengths = self.utterances.get_segmented_durations_i(i)
            # lengths = []
            # i_bound = -1
            # for bound in where_bounds:
            #     lengths.append(bound - i_bound)
            #     i_bound = bound
            logger.debug("Embeddings: " + str(self.utterances.get_segmented_embeds_i(i)))
            logger.debug("Utterance embeddings: " + str(self.utterances.get_original_segmented_embeds_i(i)))
            logger.debug("Landmark indices: " + str(self.utterances.get_segmented_landmark_indices(i)))
            logger.debug("Durations: " + str(self.utterances.get_segmented_durations_i(i)))
            logger.debug("log_margs: " + str(log_margs))
            logger.debug("sum(log_margs*lengths): " + str(np.sum(log_margs*np.array(lengths))))
            logger.debug("log p(X): " + str(self.acoustic_model.log_marg()))
            # npt.assert_almost_equal(np.sum(log_margs*np.array(lengths)), log_prob)

        # Assign new embeddings to components in `acoustic_model`
        for i_embed in self.utterances.get_segmented_embeds_i(i):
            if i_embed == -1:
                # This only happens because of backtracking in the forward-backward functions
                continue  # don't assign a non-embedding (accidently the last embedding)
            if self.fb_type == "standard":
                if anneal_gibbs_am:
                    self.acoustic_model.gibbs_sample_inside_loop_i(i_embed, anneal_temp)
                else:
                    self.acoustic_model.gibbs_sample_inside_loop_i(i_embed, anneal_temp=1)
            elif self.fb_type == "viterbi":
                self.acoustic_model.map_assign_i(i_embed)

        # Debug trace
        if i == i_debug_monitor:
            logger.debug("log p(X) after sampling: " + str(self.acoustic_model.log_marg()))
            logger.debug("Unsupervised transcript after sampling: " + str(self.get_unsup_transcript_i(i)))
            logger.debug("-"*39)

        # # temp
        # print str(self.get_unsup_transcript_i(i))

        return log_prob

    def gibbs_sample(self, n_iter, am_n_iter=0, anneal_schedule=None,
            anneal_start_temp_inv=0.1, anneal_end_temp_inv=1,
            n_anneal_steps=-1, anneal_gibbs_am=False):
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
                log_prob += self.gibbs_sample_i(i_utt, anneal_temp, anneal_gibbs_am)

            record_dict["sample_time"].append(time.time() - start_time)
            start_time = time.time()
            record_dict["log_marg"].append(self.acoustic_model.log_marg())
            record_dict["log_marg*length"].append(log_prob)
            record_dict["log_prob_z"].append(self.acoustic_model.log_prob_z())
            record_dict["log_prob_X_given_z"].append(self.acoustic_model.log_prob_X_given_z())
            record_dict["anneal_temp"].append(anneal_temp)
            record_dict["components"].append(self.acoustic_model.components.K)
            record_dict["n_tokens"].append(self.acoustic_model.get_n_assigned())

            info = "iteration: " + str(i_iter)
            for key in sorted(record_dict):
                info += ", " + key + ": " + str(record_dict[key][-1])
            logger.info(info)

        return record_dict

    def get_vec_embed_log_probs(self, vec_ids, durations):
        """
        Return the log marginal probs of the `vec_ids` embeddings, scaled by
        the given `durations`.
        """

        # Get marginals
        vec_embed_log_probs = -np.inf*np.ones(len(vec_ids))
        for i, embed_id in enumerate(vec_ids):
            if embed_id == -1:
                continue
            vec_embed_log_probs[i] = self.acoustic_model.log_marg_i(embed_id)

            # Scale log marginals by number of frames
            if np.isnan(durations[i]):
                vec_embed_log_probs[i] = -np.inf
            else:
                vec_embed_log_probs[i] *= durations[i]**self.time_power_term

        # # Scale log marginals by number of frames
        # N = int(-1 + np.sqrt(1 + 4 * 2 * len(vec_ids))) / 2  # see `__init__`
        # i_ = 0
        # for t in xrange(1, N + 1):
        #     # Per-frame scaling
        #     vec_embed_log_probs[i_:i_ + t] = vec_embed_log_probs[i_:i_ + t] * (
        #         np.arange(t, 0, -1)
        #         )

        #     # # Add duration prior
        #     # if not self.dur_gamma_a_loc_scale is None:
        #     #     duration_prior_log = gamma.logpdf(
        #     #         np.arange(t, 0, -1), self.dur_gamma_a_loc_scale[0],
        #     #         loc=self.dur_gamma_a_loc_scale[1], scale=self.dur_gamma_a_loc_scale[2]
        #     #         )
        #     #     vec_embed_log_probs[i_:i_ + t] += self.dur_scaling_factor*duration_prior_log

        #     i_ += t
        return vec_embed_log_probs + self.wip

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

    def get_log_margs_i(self, i):
        """
        Get the log marginals for the current segmentation of utterance `i`.

        The segments from utterance `i` is removed and then added back in. This
        function is used for monitoring and post-processing.
        """

        # Remove embeddings from utterance `i` from the `acoustic_model`
        segmented_embeds = self.utterances.get_segmented_embeds_i(i)
        assignments = self.acoustic_model.components.get_assignments(segmented_embeds)
        for i_embed in segmented_embeds:
            if i_embed == -1:
                continue  # don't remove a non-embedding (would accidently remove the last embedding)
            self.acoustic_model.components.del_item(i_embed)

        log_margs = [
            self.acoustic_model.log_marg_i(j) for j in
            self.utterances.get_segmented_embeds_i(i) if j != -1
            ]

        # Add the embeddings back into the model
        for embed, assignment in zip(segmented_embeds, assignments):
            self.acoustic_model.components.add_item(embed, assignment)

        return log_margs


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def process_embeddings(embedding_mats, vec_ids_dict):
    """
    Process the embeddings and vector IDs into single data structures.

    Parameters
    ----------
    n_slices_min : int
        Minimum number of landmarks over which an embedding can be calculated.
        To-do: maybe remove this here, since it is already handled directly in
        the forward-backward functions.

    Return
    ------
    (embeddings, vec_ids, utterance_labels_to_ids) : 
            (matrix of float, list of vector of int, list of str)
        All the embeddings are returned in a single matrix, with a vec_id
        vector for every utterance and a list of str indicating which vec_id
        goes with which original utterance label.
    """

    embeddings = []
    vec_ids = []
    ids_to_utterance_labels = []
    i_embed = 0
    n_disregard = 0
    n_embed = 0

    # Loop over utterances
    for i_utt, utt in enumerate(sorted(embedding_mats)):
        ids_to_utterance_labels.append(utt)
        cur_vec_ids = vec_ids_dict[utt].copy()

        # Loop over rows
        for i_row, row in enumerate(embedding_mats[utt]):

            n_embed += 1

            # # If a row is all-zero, generate random embedding or disregard
            # if np.all(row == 0):
            #     if init_zero_embeds == "rand":
            #         # For sampling from a unit sphere, see: http://en.wikipedia
            #         # .org/wiki/N-sphere#Uniformly_at_random_from_the_.28n.C2.A
            #         # 0.E2.88.92.C2.A01.29-sphere
            #         row = np.random.randn(len(row))
            #         row = row/np.linalg.norm(row)
            #         # row = np.random.rand(len(row)) - 0.5
            #         # row = row/np.linalg.norm(row)
            #     elif init_zero_embeds == "disregard":
            #         cur_vec_ids[np.where(vec_ids_dict[utt] == i_row)[0]] = -1
            #         n_disregard += 1
            #         continue
            #     else:
            #         assert False, "invalid `init_zero_embeds`"

            # Add it to the embeddings
            embeddings.append(row)

            # Update vec_ids_dict so that the index points to i_embed
            cur_vec_ids[np.where(vec_ids_dict[utt] == i_row)[0]] = i_embed
            i_embed += 1

        # # Now set `cur_vec_ids` so that embeddings shorter than `n_slices_min` gets disregarded
        # if n_slices_min > 0:
        #     N = int(-1 + np.sqrt(1 + 4 *2 * len(cur_vec_ids))) / 2
        #     i_ = 0
        #     for t in xrange(1, N + 1):
        #         cur_vec_ids[i_:i_ + t][-(n_slices_min - 1):] = -1
        #         i_ += t

        # Add the updated entry in vec_ids_dict to the overall vec_ids list
        vec_ids.append(cur_vec_ids)

    # if init_zero_embeds == "disregard":
    #     logger.info("Disregarded " + str(n_disregard) + " out of " + str(n_embed) + " embeddings")

    return (np.asarray(embeddings), vec_ids, ids_to_utterance_labels)


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

    n_slices_min_cut = -(n_slices_min - 1) if n_slices_min > 1 else None

    boundaries = np.zeros(N, dtype=bool)
    boundaries[-1] = True
    log_alphas = np.ones(N)
    log_alphas[0] = 0.0

    # Forward filtering
    i = 0
    for t in xrange(1, N):
        if np.all(vec_embed_log_probs[i:i + t][-n_slices_max:] +
                log_alphas[:t][-n_slices_max:] == -np.inf):
            log_alphas[t] = -np.inf
        else:
            log_alphas[t] = (
                _cython_utils.logsumexp(vec_embed_log_probs[i:i + t][-n_slices_max:n_slices_min_cut] +
                log_alphas[:t][-n_slices_max:n_slices_min_cut]) + log_p_continue
                )
        # if i_utt == i_debug_monitor:
        #     logger.debug("...")
        i += t

    if i_utt == i_debug_monitor:
        logger.debug("log_alphas: " + str(log_alphas))

    # Backward sampling
    t = N
    log_prob = np.float64(0.)
    while True:
        i = int(0.5*(t - 1)*t)
        log_p_k = (
            vec_embed_log_probs[i:i + t][-n_slices_max:n_slices_min_cut] +
            log_alphas[:t][-n_slices_max:n_slices_min_cut]
            )
        assert not np.isnan(np.sum(log_p_k))
        if np.all(log_p_k == -np.inf):
            logger.debug(
                "Only impossible solutions for initial back-sampling for utterance " + str(i_utt)
                )
            # Look for first point where we can actually sample and insert a boundary at this point
            while np.all(log_p_k == -np.inf):
                t = t - 1
                if t == 0:
                    break  # this is a very crappy utterance
                i = int(0.5*(t - 1)*t)
                log_p_k = (vec_embed_log_probs[i:i + t][-n_slices_max:] + log_alphas[:t][-n_slices_max:])
            logger.debug("Backtracked to cut " + str(t))
            boundaries[t - 1] = True  # insert the boundary
        if anneal_temp != 1:
            log_p_k = log_p_k[::-1] - _cython_utils.logsumexp(log_p_k)
            log_p_k_anneal = (
                1./anneal_temp * log_p_k - _cython_utils.logsumexp(1./anneal_temp * log_p_k)
                )
            p_k = np.exp(log_p_k_anneal)
        else:
            p_k = np.exp(log_p_k[::-1] - _cython_utils.logsumexp(log_p_k))
        k = _cython_utils.draw(p_k) + 1
        if n_slices_min_cut is not None:
            k += n_slices_min - 1
        if i_utt == i_debug_monitor:
            logger.debug("log P(k): " + str(log_p_k))
            logger.debug("P(k): " + str(p_k))
            logger.debug("k sampled from P(k): " + str(k))
            logger.debug("Embedding log prob: " + str(vec_embed_log_probs[i + t - k]))
        log_prob += vec_embed_log_probs[i + t - k]
        if t - k - 1 < 0:
            break
        boundaries[t - k - 1] = True
        t = t - k

    if log_prob == -np.inf:
        assert False

    return log_prob, boundaries


def forward_backward_viterbi(vec_embed_log_probs, log_p_continue, N,
        n_slices_min=0, n_slices_max=0, i_utt=None, anneal_temp=None):
    """
    Viterbi segment an utterance of length `N` based on its
    `vec_embed_log_probs` vector and return a bool vector of boundaries.

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
    i_utt : int
        If provided, index of the utterance for which to print a debug trace;
        this happens if it matches the global `i_debug_monitor`.
    anneal_temp : None
        This parameter is ignored in this function (duck typing).

    Return
    ------
    (log_prob, boundaries) : (float, vector of bool)
        The `log_prob` is the sum of the log probabilties in
        `vec_embed_log_probs` for the embeddings for the final segmentation.
    """

    boundaries = np.zeros(N, dtype=bool)
    boundaries[-1] = True
    log_alphas = np.ones(N)
    log_alphas[0] = 0.0

    n_slices_min_cut = -(n_slices_min - 1) if n_slices_min > 1 else None

    # Forward filtering
    i = 0
    for t in xrange(1, N):
        if np.all(vec_embed_log_probs[i:i + t][-n_slices_max:] +
                log_alphas[:t][-n_slices_max:] == -np.inf):
            # logger.debug("log_alphas[:t] " + str(log_alphas[:t]))
            log_alphas[t] = -np.inf
        else:
            log_alphas[t] = np.max(
                vec_embed_log_probs[i:i + t][-n_slices_max:n_slices_min_cut]
                + log_alphas[:t][-n_slices_max:n_slices_min_cut]
                )
        # if i_utt == i_debug_monitor:
        #     logger.debug("...")
        i += t

    if i_utt == i_debug_monitor:
        logger.debug("log_alphas: " + str(log_alphas))

    # Backward sampling
    t = N
    log_prob = 0.
    while True:
        i = 0.5*(t - 1)*t
        log_p_k = (
            vec_embed_log_probs[i:i + t][-n_slices_max:n_slices_min_cut] +
            log_alphas[:t][-n_slices_max:n_slices_min_cut]
            )
        if np.all(log_p_k == -np.inf):
            logger.debug(
                "Only impossible solutions for initial back-sampling for utterance " + str(i_utt)
                )
            # Look for first point where we can actually sample and insert a boundary at this point
            while np.all(log_p_k == -np.inf):
                t = t - 1
                if t == 0:
                    break  # this is a very crappy utterance
                i = 0.5*(t - 1)*t
                log_p_k = (vec_embed_log_probs[i:i + t][-n_slices_max:] + log_alphas[:t][-n_slices_max:])
            logger.debug("Backtracked to cut " + str(t))
            boundaries[t - 1] = True  # insert the boundary

        p_k = np.exp(log_p_k[::-1] - _cython_utils.logsumexp(log_p_k))
        k = np.argmax(p_k) + 1
        if n_slices_min_cut is not None:
            k += n_slices_min - 1
        if i_utt == i_debug_monitor:
            # logger.debug(
            #     "log p(y|h-): " + np.array_repr(vec_embed_log_probs[i:i + t][-n_slices_max:n_slices_min_cut][::-1])
            #     )
            # logger.debug(
            #     "log alphas: " + np.array_repr(log_alphas[:t][-n_slices_max:n_slices_min_cut][::-1])
            #     )
            logger.debug("log P(k): " + str(log_p_k))
            logger.debug("P(k): " + str(p_k))
            logger.debug("argmax P(k) for k: " + str(k))
            logger.debug("Embedding log prob: " + str(vec_embed_log_probs[i + t - k]))
        log_prob += vec_embed_log_probs[i + t - k]
        if t - k - 1 < 0:
            break
        boundaries[t - k - 1] = True
        t = t - k

    return log_prob, boundaries


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    import fbgmm
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
    # print vec_ids

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
    segmenter = UnigramAcousticWordseg(
        am_class, am_alpha, am_K, am_param_prior, embedding_mats, vec_ids_dict,
        durations_dict, landmarks_dict, p_boundary_init=0.5, beta_sent_boundary=-1, n_slices_max=2
        )

    # Perform sampling
    record = segmenter.gibbs_sample(3)

    expected_log_marg = np.array([-1039.5191206670606, -1085.1651755363787, -435.84314783538349])
    expected_log_prob_z = np.array([-2.1747517214841601, -2.947941609717641, -2.7937909298903829])
    expected_log_prob_X_given_z = np.array([-1037.3443689455764, -1082.217233926661, -433.04935690549308])

    print record["log_marg"]
    print record["log_prob_z"]
    print record["log_prob_X_given_z"]

    # npt.assert_almost_equal(record["log_marg"], expected_log_marg)
    # npt.assert_almost_equal(record["log_prob_z"], expected_log_prob_z)
    # npt.assert_almost_equal(record["log_prob_X_given_z"], expected_log_prob_X_given_z)


if __name__ == "__main__":
    main()
