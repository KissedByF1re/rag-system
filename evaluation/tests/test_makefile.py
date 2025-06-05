"""Tests for Makefile commands and functionality."""

import subprocess
import pytest
from pathlib import Path


class TestMakefile:
    """Test Makefile commands."""
    
    @pytest.fixture
    def makefile_path(self):
        """Get path to Makefile."""
        return Path(__file__).parent.parent / "Makefile"
    
    def test_makefile_exists(self, makefile_path):
        """Test that Makefile exists."""
        assert makefile_path.exists(), "Makefile should exist in project root"
    
    def test_makefile_help_command(self, makefile_path):
        """Test that make help command works."""
        result = subprocess.run(
            ["make", "-f", str(makefile_path), "help"],
            capture_output=True,
            text=True,
            cwd=makefile_path.parent
        )
        assert result.returncode == 0, "make help should succeed"
        assert "Available commands:" in result.stdout
        assert "make setup" in result.stdout
        assert "make build" in result.stdout
        assert "make test" in result.stdout
    
    def test_makefile_phony_targets(self, makefile_path):
        """Test that .PHONY targets are properly declared."""
        with open(makefile_path, 'r') as f:
            content = f.read()
        
        # Check for .PHONY declaration
        assert ".PHONY:" in content, "Makefile should have .PHONY declaration"
        
        # Check for common targets
        phony_targets = ["help", "build", "test", "run", "clean", "setup"]
        for target in phony_targets:
            assert target in content, f"Target '{target}' should be defined in Makefile"
    
    def test_makefile_docker_commands(self, makefile_path):
        """Test that Docker-related commands are properly defined."""
        with open(makefile_path, 'r') as f:
            content = f.read()
        
        # Check for Docker commands
        assert "docker-compose" in content, "Makefile should use docker-compose"
        assert "build:" in content, "build target should be defined"
        assert "test:" in content, "test target should be defined"
        assert "clean:" in content, "clean target should be defined"
    
    def test_makefile_local_commands(self, makefile_path):
        """Test that local development commands are defined."""
        with open(makefile_path, 'r') as f:
            content = f.read()
        
        # Check for local development commands
        assert "test-local:" in content, "test-local target should be defined"
        assert "poetry run" in content, "Should use poetry for local commands"
    
    def test_makefile_test_targets(self, makefile_path):
        """Test that test-related targets are properly configured."""
        with open(makefile_path, 'r') as f:
            content = f.read()
        
        # Check for test targets
        test_targets = ["test:", "test-local:", "test-config:", "test-components:"]
        for target in test_targets:
            assert target in content, f"Test target '{target}' should be defined"
    
    def test_makefile_syntax_check(self, makefile_path):
        """Test that Makefile syntax is valid."""
        # Run make with dry-run to check syntax
        result = subprocess.run(
            ["make", "-f", str(makefile_path), "-n", "help"],
            capture_output=True,
            text=True,
            cwd=makefile_path.parent
        )
        assert result.returncode == 0, f"Makefile syntax error: {result.stderr}"
    
    @pytest.mark.integration
    def test_makefile_setup_command(self, makefile_path, tmp_path):
        """Test that setup command works (integration test)."""
        # Change to temporary directory to avoid modifying real .env
        original_cwd = Path.cwd()
        try:
            # Copy Makefile to temp directory
            temp_makefile = tmp_path / "Makefile"
            temp_makefile.write_text(makefile_path.read_text())
            
            # Create .env.example for setup command
            env_example = tmp_path / ".env.example"
            env_example.write_text("OPENAI_API_KEY=your_key_here\\n")
            
            # Run setup command
            result = subprocess.run(
                ["make", "-f", str(temp_makefile), "setup"],
                capture_output=True,
                text=True,
                cwd=tmp_path
            )
            
            assert result.returncode == 0, f"Setup command failed: {result.stderr}"
            assert (tmp_path / ".env").exists(), ".env file should be created"
            
        finally:
            # Restore working directory
            import os
            os.chdir(original_cwd)
    
    def test_makefile_target_dependencies(self, makefile_path):
        """Test that target dependencies are reasonable."""
        with open(makefile_path, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        targets = {}
        
        for line in lines:
            if ':' in line and not line.startswith('\\t') and not line.startswith('#'):
                if line.strip().startswith('.PHONY'):
                    continue
                parts = line.split(':')
                if len(parts) >= 2:
                    target = parts[0].strip()
                    deps = parts[1].strip() if len(parts) > 1 else ""
                    targets[target] = deps
        
        # Basic checks
        assert len(targets) > 0, "Should have defined targets"
        assert "help" in targets, "help target should be defined"
    
    def test_makefile_echo_commands(self, makefile_path):
        """Test that commands have proper echo statements for user feedback."""
        with open(makefile_path, 'r') as f:
            content = f.read()
        
        # Check that important commands have echo statements
        important_targets = ["setup", "build", "test", "run", "clean"]
        for target in important_targets:
            # Look for echo statements after target definition
            target_pattern = f"{target}:"
            if target_pattern in content:
                # Find the section for this target
                lines = content.split('\n')
                in_target = False
                has_echo = False
                
                for line in lines:
                    if line.startswith(target + ':'):
                        in_target = True
                        continue
                    elif in_target and line.startswith('\t@echo'):
                        has_echo = True
                        break
                    elif in_target and line.strip() and not line.startswith('\t'):
                        # Next target started
                        break
                
                assert has_echo, f"Target '{target}' should have echo statement for user feedback"