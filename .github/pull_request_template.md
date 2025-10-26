<!-- Thank you for your contribution! We appreciate it. The following guidelines will help improve your pull request and facilitate feedback. If anything is unclear, don't hesitate to submit your pull request and ask the maintainers for assistance. -->

## Motivation

<!-- Explain the purpose of this PR and the goals it aims to achieve. -->

## Modifications

<!-- Describe the changes made in this PR. -->

## Commit Message Convention

Please follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for your commit messages:

- `feat:` - New features or functionality
- `fix:` - Bug fixes
- `docs:` - Documentation changes only
- `style:` - Code style changes (formatting, missing semicolons, etc.)
- `refactor:` - Code refactoring without changing functionality
- `perf:` - Performance improvements
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks, dependency updates, etc.
- `ci:` - CI/CD configuration changes

**Examples:**
- `feat: add qwen omni iterable dataset support`
- `fix: resolve bagel model configuration error`
- `docs: update training guide with YAML examples`

See [CONTRIBUTING.md](../CONTRIBUTING.md) for more details.

## CI/CD Checks

Your PR will automatically run the following checks:

- **Linting**: Code formatting with `black` (line-length=120) and import sorting with `isort`
- Run `pre-commit run --all-files` locally to verify before pushing

## Checklist

- [ ] Follow commit message convention (see above)
- [ ] Run `pre-commit run --all-files` and ensure all checks pass
- [ ] Format your code with `black` (line-length=120) and `isort`
- [ ] Add unit tests for new functionality
- [ ] Update documentation as needed, including docstrings or example tutorials
- [ ] Ensure all CI/CD checks pass