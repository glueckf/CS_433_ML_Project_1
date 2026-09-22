.PHONY: help setup lint format test check run grading-tests check-data hash-data clean

PY ?= python3.11
VENV := .venv
BIN := $(VENV)/bin

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## Create .venv, install all dependencies, install git hooks
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements-dev.txt
	$(BIN)/pre-commit install
	@echo "Done. Activate with: source $(VENV)/bin/activate"

lint:  ## Check formatting and lint rules (read-only, same as CI)
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:  ## Auto-fix lint issues and reformat
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

test:  ## Run the local test suite
	$(BIN)/pytest

check: lint test  ## Everything CI runs. Run this before opening a pull request.

run:  ## Regenerate the submission csv from scratch
	$(BIN)/python run.py

grading-tests:  ## Run the official EPFL signature tests against this working tree
	@rm -rf grading_tests
	git clone --depth 1 https://github.com/epfml/ML_course.git grading_tests/_course
	$(BIN)/pip install gitpython black
	cd grading_tests/_course/projects/project1/grading_tests && \
		$(CURDIR)/$(BIN)/pytest --github_link $(CURDIR) .

hash-data:  ## Regenerate data/MANIFEST.sha256 (repo owner only)
	cd data && shasum -a 256 x_train.csv x_test.csv y_train.csv sample_submission.csv > MANIFEST.sha256

check-data:  ## Verify your local dataset matches the committed manifest
	cd data && shasum -a 256 -c MANIFEST.sha256

clean:  ## Remove caches and cloned grading tests
	rm -rf .pytest_cache .ruff_cache grading_tests
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
