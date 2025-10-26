# Contributing to LMMs Engine

Thank you for your interest in contributing to LMMs Engine! We appreciate your efforts to improve this project.

## Getting Started

### Development Setup

#### Prerequisite: Install `uv` (optional)

The following instructions use the [`uv`](https://github.com/astral-sh/uv) package manager for faster and more reliable installs. If you don't have `uv` installed, you can install it with:

```bash
pip install uv
# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### Code Formatting and Linting

We use automated tools to maintain code quality and consistency:

#### Pre-commit Hooks

The project uses [pre-commit](https://pre-commit.com/) hooks that automatically run on every commit:

- **Black**: Code formatter (line-length=120)
- **isort**: Import statement organizer (black profile)

```bash
# Install pre-commit hooks (one-time setup)
pip install pre-commit
pre-commit install

# Run hooks manually on all files
pre-commit run --all-files

# Run hooks on specific files
pre-commit run --files src/lmms_engine/file.py
```

#### Format Your Code Before Committing

```bash
# Format with black
black --line-length=120 .

# Sort imports
isort --profile black .

# Or run pre-commit to do both
pre-commit run --all-files
```

## Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification to maintain a clean and meaningful commit history. This helps us:

- Automatically generate changelogs
- Quickly understand the nature of changes
- Trigger appropriate semantic versioning

### Commit Message Format

```
<type>: <description>

[optional body]

[optional footer(s)]
```

### Types

- **feat:** A new feature or functionality
- **fix:** A bug fix
- **docs:** Documentation changes only
- **style:** Code style changes (formatting, missing semicolons, etc.) that don't affect functionality
- **refactor:** Code changes that neither fix bugs nor add features
- **perf:** Performance improvements
- **test:** Adding or updating tests
- **chore:** Maintenance tasks, dependency updates, build configuration
- **ci:** Changes to CI/CD configuration files and scripts

### Examples

#### Feature Addition
```
feat: add qwen omni iterable dataset support

Implements new iterable dataset for Qwen Omni model with custom processor
and data loading pipeline.
```

#### Bug Fix
```
fix: resolve bagel model configuration error

Corrects the model initialization parameters in the Bagel configuration
to prevent runtime errors during training.
```

#### Documentation Update
```
docs: update training guide with YAML examples

Adds comprehensive YAML configuration examples to the training documentation
for better clarity.
```

#### Refactoring
```
refactor: simplify dataset processor initialization

Reduces code complexity by consolidating processor factory methods.
```

### Guidelines

1. **Keep the subject line concise** (50 characters or less)
2. **Use the imperative mood** ("add" not "added", "fix" not "fixed")
3. **Don't capitalize the first letter** after the type prefix
4. **No period at the end** of the subject line
5. **Use the body** to explain what and why, not how
6. **Reference issues** in the footer (e.g., `Fixes #123`)

## Code Style

Please follow the coding best practices outlined in [CLAUDE.md](../CLAUDE.md):

- Write simple, straightforward code
- Use early returns to avoid nested conditions
- Use descriptive variable and function names
- Prefer functional and immutable approaches when appropriate
- Keep files organized and appropriately sized
- Add TODO comments for issues in existing code

## Pull Request Process

1. **Fork the repository** and create your branch from `main`
2. **Follow commit message conventions** for all commits
3. **Format your code** according to the project standards
4. **Add tests** for new functionality
5. **Update documentation** as needed (docstrings, user guides, examples)
6. **Ensure all tests pass** before submitting
7. **Run pre-commit checks**: `pre-commit run --all-files`
8. **Fill out the PR template** completely
9. **Request review** from maintainers

## CI/CD Pipeline

All pull requests automatically trigger the following checks:

### Linting Workflow

**Trigger**: Runs on every push and pull request
**Platform**: Ubuntu Latest
**Python Version**: 3.9

**Checks performed**:
1. **Black** - Code formatting (line-length=120)
2. **isort** - Import organization (black profile)

**How to pass CI checks**:

```bash
# Run locally before pushing
pre-commit run --all-files

# Fix any formatting issues
black --line-length=120 .
isort --profile black .
```

**Common CI failures**:
- Code not formatted with Black
- Imports not sorted correctly
- Line length exceeds 120 characters

If CI checks fail, review the GitHub Actions logs and run the formatting tools locally to fix issues.

## Development Philosophy

Our development philosophy emphasizes:

- **Simplicity**: Prioritize simple, readable solutions
- **Less Code = Less Debt**: Minimize code footprint
- **Build Iteratively**: Start minimal, verify, then expand
- **Test Frequently**: Validate with realistic inputs
- **Clean Logic**: Keep core logic clean, push details to edges

## Testing

```bash
# Run tests
pytest tests/

# Run specific test
pytest tests/test_specific.py
```

## Questions or Issues?

If you have questions or encounter issues:

- Check existing [GitHub Issues](https://github.com/lmms-lab/lmms-engine/issues)
- Create a new issue with a clear description
- Join our community discussions

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
