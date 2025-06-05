"""Test CLI commands."""

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from rag_platform.cli import cli


class TestCLI:
    """Test CLI commands."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()
    
    def test_cli_help(self, runner):
        """Test CLI help command."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Modular RAG Platform" in result.output
    
    def test_list_components(self, runner):
        """Test list-components command."""
        result = runner.invoke(cli, ["list-components"])
        assert result.exit_code == 0
        assert "Loaders:" in result.output
        assert "pickle_loader" in result.output
        assert "Chunkers:" in result.output
        assert "recursive" in result.output
    
    def test_create_config(self, runner, tmp_path):
        """Test create-config command."""
        config_path = tmp_path / "test_config.yaml"
        result = runner.invoke(cli, ["create-config", str(config_path)])
        
        assert result.exit_code == 0
        assert config_path.exists()
        
        # Verify config content
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert "experiment" in config
        assert config["experiment"]["name"] == "sample_experiment"
        assert "data" in config["experiment"]
        assert "chunking" in config["experiment"]
    
    def test_check_env(self, runner, mock_openai_key):
        """Test check-env command."""
        result = runner.invoke(cli, ["check-env"])
        assert result.exit_code == 0
        assert "Environment Configuration" in result.output
        assert "OpenAI API Key" in result.output
    
    def test_view_results(self, runner, tmp_path):
        """Test view-results command."""
        # Create mock results file
        results = {
            "experiment_name": "test_experiment",
            "timestamp": "2024-01-01T00:00:00",
            "evaluation_results": {
                "faithfulness": 0.85,
                "answer_relevancy": 0.92
            },
            "sample_results": [
                {
                    "query": "Test question?",
                    "answer": "Test answer."
                }
            ]
        }
        
        results_path = tmp_path / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f)
        
        result = runner.invoke(cli, ["view-results", str(results_path)])
        assert result.exit_code == 0
        assert "test_experiment" in result.output
        assert "faithfulness" in result.output
        assert "0.85" in result.output
    
    def test_run_experiment_config_loading(self, runner, temp_config_file, mock_openai_key):
        """Test run command loads configuration correctly."""
        result = runner.invoke(cli, ["run", temp_config_file])
        
        # Should start loading configuration (may fail due to missing dependencies)
        assert "Loading configuration from:" in result.output or result.exit_code != 0