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

from unigram_acoustic_wordseg import process_embeddings
from utterances import Utterances
import _cython_utils
import kmeans

logger = logging.getLogger(__name__)
i_debug_monitor = 0  # 466  # the index of an utterance which is to be monitored
segment_debug_only = False  # only sample the debug utterance


#-----------------------------------------------------------------------------#
#                  SEGMENTAL K-MEANS WORD SEGMENTATION CLASS                  #
#-----------------------------------------------------------------------------#

class SegmentalKMeansWordseg(object):
    """
    Segmental k-menas word segmentation using acoustic word embeddings.

    Segmentation and sampling operations are carried out in this class.
    Segmentation results are mainly stored in `utterances`, which deals with
    all utterance-level information, but knows nothing about the acoustics. The
    `acoustic_model` deals with all the acoustic embedding operations. In the
    member functions, the index `i` generally refers to the index of an
    utterance.

    Parameters
    ----------
    am_K : int
        Acoustic model parameter.
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
    wip : float
        Word insertion penalty.
    p_boundary_init : float
        See `Utterances`.
    init_am_assignments : str
        This setting determines how the initial acoustic model assignments are
        determined:
        - "rand": Randomly assigned.
        - "one-by-one": Data vectors are added one at a time to the acoustic
          model.
        - "spread": Vectors are also randomly assigned, but here an attempt is
          made to spread the items over the different components.

    Attributes
    ----------
    utterances : Utterances
        Knows nothing about the acoustics. The indices in the `vec_ids`
        attribute refers to the embedding at the corresponding row in
        `acoustic_model.components.X`.
    acoustic_model : KMeans
        Knows nothing about utterance-level information. All embeddings are
        stored in this class as the data `components.X` attribute.
    ids_to_utterance_labels : list of str
        Keeps track of utterance labels for a specific utterance ID.
    """

    def __init__(self, am_K, embedding_mats, vec_ids_dict, durations_dict,
            landmarks_dict, seed_boundaries_dict=None,
            seed_assignments_dict=None, n_slices_min=0, n_slices_max=20,
            min_duration=0, p_boundary_init=0.5, init_am_assignments="rand",
            wip=0):

        logger.info("Initializing")

        # Check parameters
        assert seed_assignments_dict is None or seed_boundaries_dict is not None

        # Initialize simple attributes
        self.n_slices_min = n_slices_min
        self.n_slices_max = n_slices_max
        self.wip = wip

        # Process embeddings into a single matrix, and vec_ids into a list (entry for each utterance)
        embeddings, vec_ids, ids_to_utterance_labels = process_embeddings(
            embedding_mats, vec_ids_dict#, n_slices_min=n_slices_min
            )
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
        logger.info("No. initial embeddings: " + str(init_embeds.shape[0]))

        # Provide the initial acoustic model assignments and initialize the model accordingly
        assignments = -1*np.ones(N, dtype=int)
        if seed_assignments_dict is not None:
            assert False, "to-do"
    #         # Use seed assignments if provided
    #         logger.info("Using seed assignments")
    #         self.seed_to_cluster = {}
    #         i_cluster = 0
    #         for i_utt, utt in enumerate(ids_to_utterance_labels):
    #             utt_init_embeds = np.array(self.utterances.get_segmented_embeds_i(i_utt), dtype=int)
    #             utt_init_assignments = np.array(seed_assignments_dict[utt][:])
    #             utt_init_assignments = utt_init_assignments[np.where(utt_init_embeds != -1)]
    #             utt_init_embeds = utt_init_embeds[np.where(utt_init_embeds != -1)]
    #             for seed in utt_init_assignments:
    #                 if not seed in self.seed_to_cluster:
    #                     if isinstance(seed, (int, long)):
    #                         self.seed_to_cluster[seed] = seed
    #                     else:
    #                         self.seed_to_cluster[seed] = i_cluster
    #                         i_cluster += 1
    #             utt_init_assignments = [self.seed_to_cluster[i] for i in utt_init_assignments]
    #             assignments[utt_init_embeds] = utt_init_assignments
    #         if am_K is None:
    #             am_K = max(self.seed_to_cluster.values()) + 1
    #         else:
    #             assert am_K >= max(self.seed_to_cluster.values()) + 1

    #         # Initialize `acoustic_model`
    #         self.acoustic_model = kmeans.KMeans(
    #             embeddings, am_param_prior, am_alpha, am_K, assignments,
    #             covariance_type=covariance_type, lms=lms
    #             )                

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
            self.acoustic_model = kmeans.KMeans(embeddings, am_K, assignments)

        elif init_am_assignments == "spread":

            logger.info("Spreading component assignments")
            n_init_embeds = len(init_embeds)
            assignment_list = (range(am_K)*int(np.ceil(float(n_init_embeds)/am_K)))[:n_init_embeds]
            random.shuffle(assignment_list)
            assignments[init_embeds] = np.array(assignment_list)

            # Initialize `acoustic_model`
            self.acoustic_model = kmeans.KMeans(embeddings, am_K, assignments)

        elif init_am_assignments == "one-by-one":
            assert False, "to-do"

    #         # Initialize `acoustic_model`
    #         logger.info("Using a one-by-one initial assignment")
    #         self.acoustic_model = kmeans.KMeans(
    #             embeddings, am_param_prior, am_alpha, am_K, assignments,
    #             covariance_type=covariance_type, lms=lms
    #             )

    #         # Assign the embeddings one-by-one
    #         for i_embed in init_embeds:
    #             # print i_embed
    #             self.acoustic_model.gibbs_sample_inside_loop_i(i_embed)

        else:
            assert False, "invalid value for `init_am_assignments`: " + init_am_assignments

    def segment_i(self, i):
        """
        Segment new boundaries for utterance `i`.

        Return
        ------
        sum_neg_len_sqrd_norm : float
            The length-weighted k-means objective for this utterance.
        """

        # Debug trace
        logger.debug("Segmeting utterance: " + str(i))
        if i == i_debug_monitor:
            logger.debug("-"*39)
            logger.debug("Statistics before sampling")
            logger.debug(
                "sum_neg_sqrd_norm before sampling: " +
                str(self.acoustic_model.components.sum_neg_sqrd_norm())
                )
            # logger.debug(
            #     "sum_neg_sqrd_norm before sampling: " +
            #     str(self.acoustic_model.components.sum_neg_sqrd_norm())
            #     )
            # logger.debug("Unsupervised transcript before sampling: " + str(self.get_unsup_transcript_i(i)))
            logger.debug("Unsupervised transcript: " + str(self.get_unsup_transcript_i(i)))
            logger.debug("Unsupervised max transcript: " + str(self.get_max_unsup_transcript_i(i)))

        # Note the embeddings before segmentation
        old_embeds = self.utterances.get_segmented_embeds_i(i)
        # # Temp ----
        # for i_embed in old_embeds:
        #     if i_embed == -1:
        #         continue  # don't remove a non-embedding (would accidently remove the last embedding)
        #     self.acoustic_model.components.del_item(i_embed)
        # self.acoustic_model.components.clean_components()
        # # ---- Temp

        # Get the scores of the embeddings
        N = self.utterances.lengths[i]
        vec_embed_neg_len_sqrd_norms = self.get_vec_embed_neg_len_sqrd_norms(
            self.utterances.vec_ids[i, :(N**2 + N)/2],
            self.utterances.durations[i, :(N**2 + N)/2]
            )

        # Debug trace
        if i == i_debug_monitor:
            logger.debug("vec_embed_neg_len_sqrd_norms: " + str(vec_embed_neg_len_sqrd_norms))
            neg_sqrd_norms = [
                self.acoustic_model.components.max_neg_sqrd_norm_i(j) for j in
                self.utterances.get_segmented_embeds_i(i) if j != -1
                ]
            where_bounds = np.where(self.utterances.boundaries[i, :N])[0]
            embeddings = self.utterances.get_segmented_embeds_i(i)
            lengths = self.utterances.get_segmented_durations_i(i)
            logger.debug("Embeddings: " + str(embeddings))
            logger.debug("Utterance embeddings: " + str(self.utterances.get_original_segmented_embeds_i(i)))
            logger.debug("Landmark indices: " + str(self.utterances.get_segmented_landmark_indices(i)))
            logger.debug("Durations: " + str(self.utterances.get_segmented_durations_i(i)))
            logger.debug("neg_sqrd_norms: " + str(neg_sqrd_norms))
            logger.debug("neg_len_sqrd_norms: " + str(neg_sqrd_norms*np.array(lengths)))
            logger.debug("sum_neg_len_sqrd_norms: " + str(np.sum(neg_sqrd_norms*np.array(lengths))))

        # Draw new boundaries for utterance i
        sum_neg_len_sqrd_norm, self.utterances.boundaries[i, :N] = forward_backward_kmeans_viterbi(
            vec_embed_neg_len_sqrd_norms, N, self.n_slices_min, self.n_slices_max, i
            )

        # Debug trace
        if i == i_debug_monitor:
            logger.debug("Statistics after sampling, but before adding new embeddings to acoustic model")
            neg_sqrd_norms = [
                self.acoustic_model.components.max_neg_sqrd_norm_i(j) for j in
                self.utterances.get_segmented_embeds_i(i) if j != -1
                ]
            where_bounds = np.where(self.utterances.boundaries[i, :N])[0]
            embeddings = self.utterances.get_segmented_embeds_i(i)
            lengths = self.utterances.get_segmented_durations_i(i)
            logger.debug("Embeddings: " + str(embeddings))
            logger.debug("Utterance embeddings: " + str(self.utterances.get_original_segmented_embeds_i(i)))
            logger.debug("Landmark indices: " + str(self.utterances.get_segmented_landmark_indices(i)))
            logger.debug("Durations: " + str(self.utterances.get_segmented_durations_i(i)))
            logger.debug("neg_sqrd_norms: " + str(neg_sqrd_norms))
            logger.debug("neg_len_sqrd_norms: " + str(neg_sqrd_norms*np.array(lengths)))
            logger.debug("sum_neg_len_sqrd_norms: " + str(np.sum(neg_sqrd_norms*np.array(lengths))))

        # Remove old embeddings and add new ones; this is equivalent to
        # assigning the new embeddings and updating the means.
        new_embeds = self.utterances.get_segmented_embeds_i(i)
        new_k = self.get_max_unsup_transcript_i(i)
        for i_embed in old_embeds:
            if i_embed == -1:
                continue  # don't remove a non-embedding (would accidently remove the last embedding)
            self.acoustic_model.components.del_item(i_embed)
        for i_embed, k in zip(new_embeds, new_k):
            self.acoustic_model.components.add_item(i_embed, k)
        self.acoustic_model.components.clean_components()
        # self.acoustic_model.components.setup_random_means()

        # Debug trace
        if i == i_debug_monitor:
            logger.debug(
                "sum_neg_sqrd_norm after sampling: " +
                str(self.acoustic_model.components.sum_neg_sqrd_norm())
                )
            logger.debug("Unsupervised transcript after sampling: " + str(self.get_unsup_transcript_i(i)))
            logger.debug("-"*39)

        return sum_neg_len_sqrd_norm  # technically, this is with the old means (before updating, above)

    def get_vec_embed_neg_len_sqrd_norms(self, vec_ids, durations):

        # Get scores
        vec_embed_neg_len_sqrd_norms = -np.inf*np.ones(len(vec_ids))
        for i, embed_id in enumerate(vec_ids):
            if embed_id == -1:
                continue
            vec_embed_neg_len_sqrd_norms[i] = self.acoustic_model.components.max_neg_sqrd_norm_i(
                embed_id
                )

            # Scale log marginals by number of frames
            if np.isnan(durations[i]):
                vec_embed_neg_len_sqrd_norms[i] = -np.inf
            else:
                vec_embed_neg_len_sqrd_norms[i] *= durations[i]#**self.time_power_term

        return vec_embed_neg_len_sqrd_norms + self.wip

    def segment(self, n_iter, n_iter_inbetween_kmeans=0):
        """
        Perform segmentation of all utterances and update the k-means model.

        Parameters
        ----------
        n_iter : int
            Number of iterations of segmentation.
        n_iter_inbetween_kmeans : int
            Number of k-means iterations inbetween segmentation iterations.

        Return
        ------
        record_dict : dict
            Contains several fields describing the optimization iterations.
            Each field is described by its key and statistics are given in a
            list covering the iterations.
        """

        logger.info("Segmenting for " + str(n_iter) + " iterations")
        logger.debug(
            "Monitoring utterance " + self.ids_to_utterance_labels[i_debug_monitor]
            + " (index=" + str(i_debug_monitor) + ")"
            )

        # Setup record dictionary
        record_dict = {}
        record_dict["sum_neg_sqrd_norm"] = []
        record_dict["sum_neg_len_sqrd_norm"] = []
        record_dict["components"] = []
        # record_dict["n_mean_updates"] = []
        record_dict["sample_time"] = []
        record_dict["n_tokens"] = []

        # Loop over sampling iterations
        for i_iter in xrange(n_iter):

            start_time = time.time()

            # Loop over utterances
            utt_order = range(self.utterances.D)
            random.shuffle(utt_order)
            if segment_debug_only:
                utt_order = [i_debug_monitor]
            sum_neg_len_sqrd_norm = 0
            for i_utt in utt_order:
                sum_neg_len_sqrd_norm += self.segment_i(i_utt)

            record_dict["sample_time"].append(time.time() - start_time)
            start_time = time.time()
            record_dict["sum_neg_sqrd_norm"].append(self.acoustic_model.components.sum_neg_sqrd_norm())
            record_dict["sum_neg_len_sqrd_norm"].append(sum_neg_len_sqrd_norm)
            record_dict["components"].append(self.acoustic_model.components.K)
            record_dict["n_tokens"].append(self.acoustic_model.get_n_assigned())

            info = "iteration: " + str(i_iter)
            for key in sorted(record_dict):
                info += ", " + key + ": " + str(record_dict[key][-1])
            logger.info(info)

            # Perform intermediate acoustic model re-sampling
            if n_iter_inbetween_kmeans > 0:
                self.acoustic_model.fit(
                    n_iter_inbetween_kmeans, consider_unassigned=False
                    )
                # if i_iter == n_iter:
                # # Remove empty components
                # for k in np.where(
                #         self.acoustic_model.components.counts[:self.acoustic_model.components.K] == 0
                #         )[0][::-1]:
                #     self.acoustic_model.components.del_component(k)

        return record_dict

    def get_unsup_transcript_i(self, i):
        """
        Return a list of the current component assignments for current
        segmentation of `i`.
        """
        return list(
            self.acoustic_model.components.get_assignments(self.utterances.get_segmented_embeds_i(i))
            )

    def get_max_unsup_transcript_i(self, i):
        """
        Return a list of the best components for current segmentation of `i`.
        """
        return self.acoustic_model.components.get_max_assignments(
            self.utterances.get_segmented_embeds_i(i)
            )


#-----------------------------------------------------------------------------#
#                     FORWARD-BACKWARD INFERENCE FUNCTIONS                    #
#-----------------------------------------------------------------------------#

def forward_backward_kmeans_viterbi(vec_embed_neg_len_sqrd_norms, N,
        n_slices_min=0, n_slices_max=0, i_utt=None):
    """
    Segmental k-means viterbi segmentation of an utterance of length `N` based
    on its `vec_embed_neg_len_sqrd_norms` vector and return a bool vector of
    boundaries.

    Parameters
    ----------
    vec_embed_neg_len_sqrd_norms : N(N + 1)/2 length vector
        For t = 1, 2, ..., N the entries `vec_embed_neg_len_sqrd_norms[i:i + t]`
        contains the log probabilties of sequence[0:t] up to sequence[t - 1:t],
        with i = t(t - 1)/2. If you have a NxN matrix where the upper
        triangular (i, j)'th entry is the log probability of sequence[i:j + 1],
        then by stacking the upper triangular terms column-wise, you get
        vec_embed_neg_len_sqrd_norms`. Written out:
        `vec_embed_neg_len_sqrd_norms` = [neg_len_sqrd_norm(seq[0:1]),
        neg_len_sqrd_norm(seq[0:2]), neg_len_sqrd_norm(seq[1:2]),
        neg_len_sqrd_norm(seq[0:3]), ..., neg_len_sqrd_norm(seq[N-1:N])].
    n_slices_max : int
        If 0, then the full length are considered. This won't necessarily lead
        to problems, since unassigned embeddings would still be ignored since
        their assignments are -1 and the would therefore have a log probability
        of -inf.
    i_utt : int
        If provided, index of the utterance for which to print a debug trace;
        this happens if it matches the global `i_debug_monitor`.

    Return
    ------
    (sum_neg_len_sqrd_norm, boundaries) : (float, vector of bool)
        The `sum_neg_len_sqrd_norm` is the sum of the scores in
        `vec_embed_neg_len_sqrd_norms` for the embeddings for the final
        segmentation.
    """

    n_slices_min_cut = -(n_slices_min - 1) if n_slices_min > 1 else None

    boundaries = np.zeros(N, dtype=bool)
    boundaries[-1] = True
    gammas = np.ones(N)
    gammas[0] = 0.0

    # Forward filtering
    i = 0
    for t in xrange(1, N):
        if np.all(vec_embed_neg_len_sqrd_norms[i:i + t][-n_slices_max:] +
                gammas[:t][-n_slices_max:] == -np.inf):
            # logger.debug("gammas[:t] " + str(gammas[:t]))
            gammas[t] = -np.inf
        else:
            gammas[t] = np.max(
                vec_embed_neg_len_sqrd_norms[i:i + t][-n_slices_max:n_slices_min_cut] +
                gammas[:t][-n_slices_max:n_slices_min_cut]
                )
        # if i_utt == i_debug_monitor:
        #     logger.debug("...")
        i += t

    if i_utt == i_debug_monitor:
        logger.debug("gammas: " + str(gammas))

    # Backward segmentation
    t = N
    sum_neg_len_sqrd_norm = 0.
    while True:
        i = int(0.5*(t - 1)*t)
        q_t = (
            vec_embed_neg_len_sqrd_norms[i:i + t][-n_slices_max:n_slices_min_cut] +
            gammas[:t][-n_slices_max:n_slices_min_cut]
            )
        assert not np.isnan(np.sum(q_t))
        if np.all(q_t == -np.inf):
            logger.debug(
                "Only impossible solutions for initial back-sampling for utterance " + str(i_utt)
                )
            # Look for first point where we can actually sample and insert a boundary at this point
            while np.all(q_t == -np.inf):
                t = t - 1
                if t == 0:
                    break  # this is a very crappy utterance
                i = 0.5*(t - 1)*t
                q_t = (vec_embed_neg_len_sqrd_norms[i:i + t][-n_slices_max:] + gammas[:t][-n_slices_max:])
            logger.debug("Backtracked to cut " + str(t))
            boundaries[t - 1] = True  # insert the boundary

        q_t = q_t[::-1]
        k = np.argmax(q_t) + 1
        if n_slices_min_cut is not None:
            k += n_slices_min - 1
        if i_utt == i_debug_monitor:
            # logger.debug(
            #     "log p(y|h-): " + np.array_repr(vec_embed_neg_len_sqrd_norms[i:i + t][-n_slices_max:n_slices_min_cut][::-1])
            #     )
            # logger.debug(
            #     "log alphas: " + np.array_repr(gammas[:t][-n_slices_max:n_slices_min_cut][::-1])
            #     )
            logger.debug("q_t: " + str(q_t))
            logger.debug("argmax q_t: " + str(k))
            logger.debug("Embedding neg_len_sqrd_norms: " + str(vec_embed_neg_len_sqrd_norms[i + t - k]))
        sum_neg_len_sqrd_norm += vec_embed_neg_len_sqrd_norms[i + t - k]
        if t - k - 1 < 0:
            break
        boundaries[t - k - 1] = True
        t = t - k

    return sum_neg_len_sqrd_norm, boundaries


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    # import fbgmm
    # import gaussian_components_fixedvar

    logging.basicConfig(level=logging.DEBUG)

    # embedding_mat1 = np.array(
    #     [[ 1.55329044,  0.82568932,  0.56011276],
    #    [ 1.10640768, -0.41715366,  0.30323529],
    #    [ 1.24183824, -2.39021548,  0.02369367],
    #    [ 1.26094544, -0.27567053,  1.35731148],
    #    [ 1.59711416, -0.54917262, -0.56074459],
    #    [-0.4298405 ,  1.39010761, -1.2608597 ]], dtype=np.float32
    #     )
    # embedding_mat2 = np.array(
    #     [[ 1.63075195,  0.25297823, -1.75406467],
    #    [-0.59324473,  0.96613426, -0.20922202],
    #    [ 0.97066059, -1.22315308, -0.37979187],
    #    [-0.31613254, -0.07262261, -1.04392799],
    #    [-1.11535652,  0.33905751,  1.85588856],
    #    [-1.08211738,  0.88559445,  0.2924617 ]], dtype=np.float32
    #     )

    # # Vector IDs
    # n_slices = 3
    # vec_ids = -1*np.ones((n_slices**2 + n_slices)/2, dtype=int)
    # i_embed = 0
    # n_slices_max = 20
    # for cur_start in range(n_slices):
    #     for cur_end in range(cur_start, min(n_slices, cur_start + n_slices_max)):
    #         cur_end += 1
    #         t = cur_end
    #         i = t*(t - 1)/2
    #         vec_ids[i + cur_start] = i_embed
    #         # print cur_start, cur_end, i + cur_start, i_embed
    #         i_embed += 1
    # # print vec_ids

    # embedding_mats = {}
    # vec_ids_dict = {}
    # seed_bounds_dict = {}
    # durations_dict = {}
    # landmarks_dict = {}
    # embedding_mats["test1"] = embedding_mat1
    # vec_ids_dict["test1"] = vec_ids
    # seed_bounds_dict["test1"] = [2]
    # landmarks_dict["test1"] = [1, 2, 3]
    # durations_dict["test1"] = [1, 2, 1, 3, 2, 1]
    # embedding_mats["test2"] = embedding_mat2
    # vec_ids_dict["test2"] = vec_ids
    # seed_bounds_dict["test2"] = [2]
    # landmarks_dict["test2"] = [1, 2, 3]
    # durations_dict["test2"] = [1, 2, 1, 3, 2, 1]

    # random.seed(1)
    # np.random.seed(1)

    # global i_debug_monitor
    # i_debug_monitor = 0

    # # Data parameters
    # D = 3  # dimensionality of embeddings

    # # Acoustic model parameters
    # am_class = fbgmm.FBGMM
    # am_alpha = 10.
    # am_K = 2
    # m_0 = np.zeros(D)
    # k_0 = 0.05
    # # S_0 = 0.025*np.ones(D)
    # S_0 = 0.002*np.ones(D)
    # am_param_prior = gaussian_components_fixedvar.FixedVarPrior(S_0, m_0, S_0/k_0)

    # # Initialize model
    # segmenter = UnigramAcousticWordseg(
    #     am_class, am_alpha, am_K, am_param_prior, embedding_mats, vec_ids_dict,
    #     durations_dict, landmarks_dict, p_boundary_init=0.5, beta_sent_boundary=-1, n_slices_max=2
    #     )

    # # Perform sampling
    # record = segmenter.gibbs_sample(3)

    # expected_log_marg = np.array([-1039.5191206670606, -1085.1651755363787, -435.84314783538349])
    # expected_log_prob_z = np.array([-2.1747517214841601, -2.947941609717641, -2.7937909298903829])
    # expected_log_prob_X_given_z = np.array([-1037.3443689455764, -1082.217233926661, -433.04935690549308])

    # print record["log_marg"]
    # print record["log_prob_z"]
    # print record["log_prob_X_given_z"]

    # npt.assert_almost_equal(record["log_marg"], expected_log_marg)
    # npt.assert_almost_equal(record["log_prob_z"], expected_log_prob_z)
    # npt.assert_almost_equal(record["log_prob_X_given_z"], expected_log_prob_X_given_z)


if __name__ == "__main__":
    main()
