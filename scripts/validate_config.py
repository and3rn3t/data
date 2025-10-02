#!/usr/bin/env python3
"""
Project Configuration Validation Script
Validates that all configuration files are properly set up for the Data Science Sandbox project.
"""

import configparser
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import toml
except ImportError:
    print("Warning: toml package not available. Install with: pip install toml")
    toml = None

try:
    import yaml
except ImportError:
    print("Warning: yaml package not available. Install with: pip install pyyaml")
    yaml = None


class ConfigValidator:
    """Validates project configuration files."""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(f"❌ {message}")

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(f"⚠️  {message}")

    def add_success(self, message: str) -> None:
        """Add a success message."""
        print(f"✅ {message}")

    def validate_pyproject_toml(self) -> None:
        """Validate pyproject.toml configuration."""
        pyproject_path = self.project_root / "pyproject.toml"

        if not pyproject_path.exists():
            self.add_error("pyproject.toml not found")
            return

        if toml is None:
            self.add_warning(
                "Cannot validate pyproject.toml - toml package not available"
            )
            return

        try:
            with open(pyproject_path, "r", encoding="utf-8") as f:
                config = toml.load(f)

            # Check for required sections
            required_sections = [
                "build-system",
                "tool.black",
                "tool.isort",
                "tool.ruff",
            ]
            for section in required_sections:
                if self._get_nested_key(config, section.split(".")) is None:
                    self.add_error(f"Missing section [{section}] in pyproject.toml")
                else:
                    self.add_success(f"Found [{section}] configuration")

            # Check black configuration
            black_config = self._get_nested_key(config, ["tool", "black"])
            if black_config and black_config.get("line-length") == 88:
                self.add_success("Black line length correctly set to 88")

            # Check ruff configuration
            ruff_config = self._get_nested_key(config, ["tool", "ruff"])
            if ruff_config and ruff_config.get("line-length") == 88:
                self.add_success("Ruff line length correctly set to 88")

        except Exception as e:
            self.add_error(f"Failed to parse pyproject.toml: {e}")

    def validate_flake8_config(self) -> None:
        """Validate .flake8 configuration."""
        flake8_path = self.project_root / ".flake8"

        if not flake8_path.exists():
            self.add_error(".flake8 configuration file not found")
            return

        try:
            config = configparser.ConfigParser()
            config.read(flake8_path)

            if "flake8" not in config:
                self.add_error("Missing [flake8] section in .flake8")
                return

            flake8_section = config["flake8"]

            # Check line length
            if flake8_section.get("max-line-length") == "88":
                self.add_success("Flake8 line length correctly set to 88")
            else:
                self.add_warning("Flake8 line length not set to 88")

            # Check if extend-ignore is set
            if "extend-ignore" in flake8_section:
                self.add_success("Flake8 extend-ignore configuration found")

            # Check if exclude is set
            if "exclude" in flake8_section:
                self.add_success("Flake8 exclude configuration found")

        except Exception as e:
            self.add_error(f"Failed to parse .flake8: {e}")

    def validate_pytest_config(self) -> None:
        """Validate pytest.ini configuration."""
        pytest_path = self.project_root / "pytest.ini"

        if not pytest_path.exists():
            self.add_error("pytest.ini configuration file not found")
            return

        try:
            config = configparser.ConfigParser()
            config.read(pytest_path)

            if "tool:pytest" not in config:
                self.add_error("Missing [tool:pytest] section in pytest.ini")
                return

            pytest_section = config["tool:pytest"]

            # Check for important settings
            important_settings = ["testpaths", "python_files", "addopts", "markers"]
            for setting in important_settings:
                if setting in pytest_section:
                    self.add_success(f"Pytest {setting} configuration found")
                else:
                    self.add_warning(f"Pytest {setting} configuration missing")

        except Exception as e:
            self.add_error(f"Failed to parse pytest.ini: {e}")

    def validate_pre_commit_config(self) -> None:
        """Validate .pre-commit-config.yaml configuration."""
        precommit_path = self.project_root / ".pre-commit-config.yaml"

        if not precommit_path.exists():
            self.add_error(".pre-commit-config.yaml not found")
            return

        if yaml is None:
            self.add_warning(
                "Cannot validate .pre-commit-config.yaml - yaml package not available"
            )
            return

        try:
            with open(precommit_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            if "repos" not in config:
                self.add_error("Missing 'repos' section in .pre-commit-config.yaml")
                return

            repo_hooks = []
            for repo in config["repos"]:
                if "hooks" in repo:
                    for hook in repo["hooks"]:
                        repo_hooks.append(hook.get("id", "unknown"))

            # Check for important hooks
            important_hooks = ["black", "isort", "ruff", "trailing-whitespace"]
            for hook in important_hooks:
                if any(hook in repo_hook for repo_hook in repo_hooks):
                    self.add_success(f"Pre-commit hook '{hook}' found")
                else:
                    self.add_warning(f"Pre-commit hook '{hook}' not found")

        except Exception as e:
            self.add_error(f"Failed to parse .pre-commit-config.yaml: {e}")

    def validate_gitignore(self) -> None:
        """Validate .gitignore file."""
        gitignore_path = self.project_root / ".gitignore"

        if not gitignore_path.exists():
            self.add_error(".gitignore file not found")
            return

        try:
            with open(gitignore_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for important patterns
            important_patterns = [
                "__pycache__",
                "*.pyc",
                ".venv",
                "venv/",
                ".env",
                "*.log",
                "mlruns/",
                "wandb/",
                "*.pkl",
                "*.pickle",
            ]

            for pattern in important_patterns:
                if pattern in content:
                    self.add_success(f"Gitignore pattern '{pattern}' found")
                else:
                    self.add_warning(f"Gitignore pattern '{pattern}' not found")

        except Exception as e:
            self.add_error(f"Failed to read .gitignore: {e}")

    def validate_editorconfig(self) -> None:
        """Validate .editorconfig file."""
        editorconfig_path = self.project_root / ".editorconfig"

        if not editorconfig_path.exists():
            self.add_warning(".editorconfig file not found (recommended)")
            return

        try:
            with open(editorconfig_path, "r", encoding="utf-8") as f:
                content = f.read()

            if "root = true" in content:
                self.add_success("EditorConfig root setting found")

            if "[*.py]" in content:
                self.add_success("EditorConfig Python file settings found")
            else:
                self.add_warning("EditorConfig Python file settings not found")

        except Exception as e:
            self.add_error(f"Failed to read .editorconfig: {e}")

    def validate_makefile(self) -> None:
        """Validate Makefile."""
        makefile_path = self.project_root / "Makefile"

        if not makefile_path.exists():
            self.add_warning("Makefile not found (recommended for development)")
            return

        try:
            with open(makefile_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for important targets
            important_targets = ["help", "test", "lint", "format", "clean"]
            for target in important_targets:
                if f"{target}:" in content:
                    self.add_success(f"Makefile target '{target}' found")
                else:
                    self.add_warning(f"Makefile target '{target}' not found")

        except Exception as e:
            self.add_error(f"Failed to read Makefile: {e}")

    def _get_nested_key(self, data: Dict, keys: List[str]) -> Optional[any]:
        """Get a nested key from a dictionary."""
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    def validate_all(self) -> bool:
        """Run all validations."""
        print("🔍 Validating Data Science Sandbox Configuration...\n")

        print("📋 Checking core configuration files:")
        self.validate_pyproject_toml()
        self.validate_flake8_config()
        self.validate_pytest_config()

        print("\n🔨 Checking development tools:")
        self.validate_pre_commit_config()
        self.validate_gitignore()
        self.validate_editorconfig()
        self.validate_makefile()

        print(f"\n📊 Validation Summary:")
        if self.warnings:
            print("\nWarnings:")
            for warning in self.warnings:
                print(f"  {warning}")

        if self.errors:
            print("\nErrors:")
            for error in self.errors:
                print(f"  {error}")
            return False
        else:
            print(
                "\n🎉 All validations passed! Your project configuration looks great."
            )
            if self.warnings:
                print(f"   Note: {len(self.warnings)} warning(s) found (see above)")
            return True


def main():
    """Main validation function."""
    validator = ConfigValidator()
    success = validator.validate_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
