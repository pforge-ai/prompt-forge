# Contributing to Prompt Forge

First off, thank you for considering contributing to Prompt Forge! We welcome contributions in various forms, whether it's reporting bugs, suggesting new features, improving documentation, or directly contributing code.

## How to Contribute

We encourage participation through the following channels:

* **Reporting Bugs**: If you find an error, please report it via GitHub Issues.
* **Suggesting Features**: If you have new ideas or suggestions for improvement, feel free to propose them in GitHub Issues or Discussions.
* **Submitting Pull Requests (PRs)**: If you want to fix a bug or implement a new feature directly, please follow the PR process outlined below.
* **Improving Documentation**: If you find any part of the documentation unclear, incorrect, or missing, you can submit a PR directly to improve it.

## Reporting Bugs

Before reporting a bug, please search the existing [GitHub Issues](https://github.com/pforge-ai/prompt-forge/issues) to ensure the bug hasn't already been reported.

When reporting a bug, please include as much detail as possible:

* Your operating system and Python version.
* The version of `prompt-forge` you installed (and relevant dependency versions, if possible).
* Detailed steps to reproduce the bug.
* What you expected to happen.
* What actually happened (including full error messages and tracebacks).

## Suggesting Features

Before suggesting a new feature, please also check existing [GitHub Issues](https://github.com/pforge-ai/prompt-forge/issues) and [Discussions](https://github.com/pforge-ai/prompt-forge/discussions) to see if a similar idea has already been proposed.

When making a suggestion, please clearly describe:

* What feature you would like to see.
* What problem this feature solves or what value it brings.
* (Optional) Any initial thoughts you have on how it could be implemented.

## Pull Request (PR) Process

We greatly appreciate code contributions! Please follow these steps:

1.  **Fork the Repository**: Fork the main repository to your own GitHub account.
2.  **Clone Your Fork**: `git clone https://github.com/pforge-ai/prompt-forge.git` (replace with your username).
3.  **Create a Branch**: Create a new feature branch from `main` (or the current development branch): `git checkout -b feature/your-feature-name` or `bugfix/issue-number`.
4.  **Set Up Development Environment**:
    * It's recommended to create a virtual environment: `python -m venv .venv` and activate it (`source .venv/bin/activate` or `.venv\Scripts\activate`).
    * Install project dependencies and development dependencies: `pip install -e .[dev]` (assuming you define `dev` dependencies like `pytest`, `black`, `flake8`, `mypy` in your `setup.py`'s `extras_require`).
5.  **Make Changes**: Write your code or fix the bug.
6.  **Follow Coding Standards**:
    * Please format your code using [Black](https://github.com/psf/black).
    * Please check your code style and potential errors using [Flake8](https://flake8.pycqa.org/en/latest/).
    * Please sort your import statements using [isort](https://pycqa.github.io/isort/).
    * (Recommended) Use Type Hinting and check with [MyPy](http://mypy-lang.org/).
7.  **Add Tests**: If you add a new feature or fix a bug, please add corresponding unit tests (using `pytest`). Ensure all tests pass (`pytest`).
8.  **Update Documentation**: If your changes affect the user interface or add new functionality, please update the relevant documentation (e.g., README or Docstrings).
9.  **Commit Changes**: `git commit -m "feat: Add feature X"` or `fix: Fix bug Y related to #issue_number`. Following the [Conventional Commits](https://www.conventionalcommits.org/) specification is a good practice.
10. **Push Branch**: `git push origin feature/your-feature-name`.
11. **Create Pull Request**: Go to your fork on GitHub, click "New pull request", select your branch, and compare it against the main repository's `main` branch. Fill in a clear PR title and description explaining what changes you made and why. If your PR addresses an issue, link it in the description (e.g., `Closes #123`).
12. **Code Review**: Wait for project maintainers to review your code and address any feedback.

## Coding Standards

* Please follow PEP 8 guidelines.
* Use Black for code formatting.
* Use Flake8 for code linting.
* Use isort for import sorting.
* Type hinting checked with MyPy is encouraged.
* Write clear and meaningful commit messages.

## Code of Conduct

We are committed to providing a friendly, safe, and welcoming environment for all participants. Please ensure you read and adhere to our [Code of Conduct](CODE_OF_CONDUCT.md).

## Questions

If you have any questions during the contribution process or about the project, feel free to reach out via [GitHub Discussions](https://github.com/pforge-ai/prompt-forge/discussions) or [GitHub Issues](https://github.com/pforge-ai/prompt-forge/issues).

Thank you for your contributions!
