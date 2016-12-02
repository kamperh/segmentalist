Segmentalist
============

Overview
--------
Perform unsupervised acoustic word segmentation using both unigram and
(partially) bigram language models. The latter is only partially implemented;
specifically, only bigram cluster assignments are sampled, and only using a
maximum likelihood bigram language model.

If you use this code, please cite one of the references below (or all of them
if you are kind). The code here relies on prior feature extraction and a number
of preprocessed files for your corpus which is not illustrated here. See
[bucktsong_segmentalist](https://github.com/kamperh/bucktsong_segmentalist/)
for a complete recipe using this code.


Dependencies
------------
- [Cython](http://cython.org/)


Building and testing the code
-----------------------------
Run `make` to build the Cython components. Run `make test` to run unit tests.


Examples
--------
An IPython notebook example of clustering is given in
`examples/clustering_examples.ipynb`. This is just to illustrate some of the
differences between FBGMM and k-means clustering, and no segmentation is
performed on the generated toy data.


References
----------
- H. Kamper, A. Jansen, and S. J. Goldwater, "A segmental framework for
  fully-unsupervised large-vocabulary speech recognition," *arXiv preprint
  arXiv:1606.06950*, 2016.
- H. Kamper, A. Jansen, and S. J. Goldwater, "Unsupervised word segmentation
  and lexicon discovery using acoustic word embeddings," *IEEE Trans. Audio,
  Speech, Language Process.*, vol. 24, no. 4, pp. 669-679, 2016.
- H. Kamper, A. Jansen, and S. J. Goldwater, "Fully unsupervised
  small-vocabulary speech recognition using a segmental Bayesian model," in
  *Proc. Interspeech*, 2015.


Contributors
------------
- [Herman Kamper](http://www.kamperh.com/)
- [Aren Jansen](http://www.clsp.jhu.edu/~ajansen/)
- [Sharon Goldwater](http://homepages.inf.ed.ac.uk/sgwater/)
