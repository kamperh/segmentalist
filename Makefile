all:
	python setup.py build_ext --inplace

test:
	nosetests -v

coverage:
	nosetests --with-coverage --cover-package=unigram_segmentalist .
