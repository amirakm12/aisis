# Al-artworks Test Suite

This folder contains all tests for Al-artworks, covering agents, core modules, UI, and integration.

## Test Structure

- `test_agents.py` - Agent system tests
- `test_core.py` - Core functionality tests  
- `test_integration.py` - Integration tests
- `test_performance.py` - Performance tests
- `test_ui.py` - UI component tests

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_ui.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=src

# Run performance tests only
pytest tests/test_performance.py
```

## Test Categories

### Unit Tests
- Individual component functionality
- Isolated module testing
- Mock dependencies

### Integration Tests  
- Component interaction
- End-to-end workflows
- System integration

### Performance Tests
- Benchmarking
- Resource usage
- Scalability testing

### UI Tests
- User interface components
- User interaction flows
- Responsiveness testing

## Test Configuration

Tests use pytest fixtures and configuration files in `tests/fixtures/` for consistent test data and setup.

## Continuous Integration

Tests are automatically run on:
- Pull requests
- Main branch commits
- Release tags

## Coverage Requirements

- Minimum 80% code coverage
- 100% coverage for critical paths
- UI component testing
- Performance regression testing 