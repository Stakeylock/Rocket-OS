# Contributing to Autonomous Rocket AI OS

Thank you for interest in contributing! This document outlines how to contribute to this research project.

## Code of Conduct

We are committed to keeping this a welcoming, inclusive, and professional project. Please treat all contributors with respect.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion, please open a GitHub issue with:
- **Clear title** describing the problem
- **Steps to reproduce** (for bugs) or detailed description (for features)
- **Expected vs. actual behavior**
- **Your environment** (Python version, OS, key deps)

### Submitting Changes

1. **Fork the repository** and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** with clear commit messages:
   ```bash
   git commit -m "brief description of change"
   ```

3. **Run tests** to verify nothing breaks:
   ```bash
   pytest test_mission_smoke.py
   ```

4. **Keep code style consistent**:
   - Use Python 3.10+ syntax
   - Follow PEP 8 naming and formatting
   - Add docstrings to functions and classes
   - Keep lines ≤100 characters where practical

5. **Open a pull request** with:
   - Clear description of what and why
   - Reference to any related issues
   - Confirmation that tests pass

### Adding New Subsystems

To add a new subsystem module:

1. Create a new folder under the relevant category (e.g., `rocket_ai_os/gnc/`, `rocket_ai_os/comms/`)
2. Implement your module with clear class/function interfaces
3. Add a docstring explaining the subsystem's role in the architecture
4. Create a test file `test_<subsystem>.py` in the module folder
5. Update the README subsystem table and project structure if adding a major component
6. Update `CHANGELOG.md` with your changes

### Adding Tests

New features should include tests:

```python
# test_my_feature.py
def test_feature_basic():
    """Test basic functionality."""
    result = my_function(input_data)
    assert result == expected_output

def test_feature_edge_case():
    """Test edge case handling."""
    # Example: empty input, boundary conditions, etc.
    pass
```

Run tests with:
```bash
pytest -v
```

## Development Workflow

```bash
# Install development dependencies
pip install -r requirements.txt pytest pytest-cov

# Make changes in a feature branch
git checkout -b feature/something

# Run linting & tests
pytest

# Commit with clear messages
git commit -m "Add feature: clear description"

# Push and open PR
git push origin feature/something
```

## Release Process

Maintainers will:
1. Merge approved PRs to `main`
2. Update `CHANGELOG.md`
3. Create a GitHub Release with tag `vX.Y.Z`
4. Update the version badge in README

## Questions?

- Check existing issues and documentation
- Open a discussion or issue with your question
- For research-specific questions, see the research paper

## License

By contributing, you agree your contributions will be licensed under the MIT License (same as the project).

---

Thank you for helping improve this project! 🚀
