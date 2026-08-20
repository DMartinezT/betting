PYTHON ?= python3

.PHONY: install test check-figures figures check

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m unittest discover -s tests -t . -v

check-figures:
	$(PYTHON) reproduce_paper_figures.py --check-inputs

figures:
	$(PYTHON) reproduce_paper_figures.py

check: check-figures test
