Segmentalist
============

Overview
--------
Perform unsupervised acoustic word segmentation using both unigram and
(partially) bigram language models. The latter is only partially implemented;
specifically, only bigram cluster assignments are sampled, and only using a
maximum likelihood bigram language model.


Dependencies
------------
- [Cython](http://cython.org/)


Building the code
-----------------
Run `make`.


References
----------
- H. Kamper, A. Jansen, and S. J. Goldwater, "Unsupervised word segmentation
  and lexicon discovery using acoustic word embeddings," *IEEE Trans. Audio,
  Speech, Language Process.*, vol. 24, no. 4, pp. 669-679, 2016.
- H. Kamper, A. Jansen, and S. J. Goldwater, "Fully unsupervised
  small-vocabulary speech recognition using a segmental Bayesian model," in
  *Proc. Interspeech*, 2015.
- H. Kamper, A. Jansen, S. King, and S. Goldwater, "Unsupervised lexical
  clustering of speech segments using fixed-dimensional acoustic embeddings,"
  in *Proc. SLT*, 2014.


Collaborators
-------------
- [Herman Kamper](http://www.kamperh.com/)
- [Aren Jansen](http://www.clsp.jhu.edu/~ajansen/)
- [Sharon Goldwater](http://homepages.inf.ed.ac.uk/sgwater/)
