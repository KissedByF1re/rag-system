# Tests Directory

This directory contains tests for the RAG System project. All tests in this directory are passing and can be run to verify the system's functionality.

## Directory Structure

- **`/tests/core/`**: Tests for core functionality
  - `test_llm.py`: Tests for language model integrations
  - `test_settings.py`: Tests for configuration settings

- **`/tests/client/`**: Tests for client functionality
  - `test_client.py`: Tests for the Agent client

- **`/tests/service/`**: Tests for service functionality
  - `test_utils.py`: Tests for service utility functions
  - `test_service_e2e.py`: Basic end-to-end tests for service functionality

## Running Tests

To run all tests:

```bash
python -m pytest tests
```

To run tests for a specific component:

```bash
python -m pytest tests/core/
python -m pytest tests/client/
python -m pytest tests/service/
```

To run a specific test file:

```bash
python -m pytest tests/core/test_llm.py
```

To run tests with verbose output:

```bash
python -m pytest tests -v
```

## Test Coverage

The current test suite covers:
- Core functionality (settings, LLM models)
- Client API
- Service utilities
- Basic service functionality

## Requirements

Tests require the following dependencies installed:
- pytest
- langchain and its related packages
- httpx for client tests

These dependencies are specified in the project's `requirements.txt` file. 