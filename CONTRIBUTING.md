# Contributing to Sumplicity

Thanks for your interest in improving Sumplicity.

## Reporting bugs and asking questions

Open an issue at https://github.com/YOUR-USERNAME/sumplicity/issues. For bugs,
include your Python, NumPy and NLTK versions, a minimal input that reproduces the
problem, and the output you expected.

## Proposing changes

1. Fork the repository and create a branch from `main`.
2. Install in editable mode: `pip install -e ".[test]"`.
3. Make your change and add or update tests in `tests/`.
4. Run `pytest` and make sure everything passes.
5. Add a line under **Unreleased** in `CHANGELOG.md`.
6. Open a pull request describing what changed and why.

New algorithms should subclass `BaseSummarizer`, implement `rank(TF, top_k)`
using the shared TF matrix, cite the original paper in the class docstring, and
be registered in `get_all_summarizer()`.

## Support and maintenance

Sumplicity is maintained by Osiris Terry. Issues are typically triaged within
two weeks. Releases follow semantic versioning and are listed in `CHANGELOG.md`.
