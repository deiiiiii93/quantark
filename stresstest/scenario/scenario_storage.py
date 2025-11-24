"""
Storage and serialization for stress test scenarios.

Supports loading and saving scenarios from/to YAML and JSON files.
"""

import json
import yaml
from pathlib import Path
from typing import List, Union
from stresstest.scenario.scenario import Scenario
from util.exceptions import ValidationError


class ScenarioStorage:
    """
    Storage manager for stress test scenarios.
    
    Handles serialization and deserialization of scenarios to/from
    YAML and JSON files for reusability and version control.
    
    Example:
        >>> # Save scenarios
        >>> ScenarioStorage.save_scenarios(scenarios, "my_scenarios.yaml")
        
        >>> # Load scenarios
        >>> scenarios = ScenarioStorage.load_scenarios("my_scenarios.yaml")
    """
    
    @staticmethod
    def save_scenarios(
        scenarios: List[Scenario],
        filepath: Union[str, Path],
        format: str = "auto"
    ) -> None:
        """
        Save scenarios to a file.
        
        Args:
            scenarios: List of scenarios to save
            filepath: Output file path
            format: File format ('yaml', 'json', or 'auto' to detect from extension)
            
        Raises:
            ValidationError: If format is invalid or file cannot be written
        """
        filepath = Path(filepath)
        
        # Detect format from extension if auto
        if format == "auto":
            ext = filepath.suffix.lower()
            if ext in ['.yaml', '.yml']:
                format = 'yaml'
            elif ext == '.json':
                format = 'json'
            else:
                raise ValidationError(
                    f"Cannot auto-detect format from extension '{ext}'. "
                    "Use .yaml, .yml, or .json, or specify format explicitly."
                )
        
        # Convert scenarios to dictionaries
        data = {
            'version': '1.0',
            'scenarios': [s.to_dict() for s in scenarios]
        }
        
        # Write file
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'yaml':
            with open(filepath, 'w') as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        elif format == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValidationError(f"Invalid format: {format}. Must be 'yaml' or 'json'")
    
    @staticmethod
    def load_scenarios(
        filepath: Union[str, Path],
        format: str = "auto"
    ) -> List[Scenario]:
        """
        Load scenarios from a file.
        
        Args:
            filepath: Input file path
            format: File format ('yaml', 'json', or 'auto' to detect from extension)
            
        Returns:
            List of loaded scenarios
            
        Raises:
            ValidationError: If file cannot be read or format is invalid
            FileNotFoundError: If file does not exist
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Scenario file not found: {filepath}")
        
        # Detect format from extension if auto
        if format == "auto":
            ext = filepath.suffix.lower()
            if ext in ['.yaml', '.yml']:
                format = 'yaml'
            elif ext == '.json':
                format = 'json'
            else:
                raise ValidationError(
                    f"Cannot auto-detect format from extension '{ext}'. "
                    "Use .yaml, .yml, or .json, or specify format explicitly."
                )
        
        # Load file
        if format == 'yaml':
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
        elif format == 'json':
            with open(filepath, 'r') as f:
                data = json.load(f)
        else:
            raise ValidationError(f"Invalid format: {format}. Must be 'yaml' or 'json'")
        
        # Validate structure
        if not isinstance(data, dict):
            raise ValidationError("Invalid scenario file: root must be a dictionary")
        
        if 'scenarios' not in data:
            raise ValidationError("Invalid scenario file: missing 'scenarios' key")
        
        # Convert to Scenario objects
        scenarios = []
        for scenario_data in data['scenarios']:
            try:
                scenario = Scenario.from_dict(scenario_data)
                scenarios.append(scenario)
            except Exception as e:
                raise ValidationError(f"Error loading scenario: {e}")
        
        return scenarios
    
    @staticmethod
    def save_scenario(
        scenario: Scenario,
        filepath: Union[str, Path],
        format: str = "auto"
    ) -> None:
        """
        Save a single scenario to a file.
        
        Args:
            scenario: Scenario to save
            filepath: Output file path
            format: File format ('yaml', 'json', or 'auto')
        """
        ScenarioStorage.save_scenarios([scenario], filepath, format)
    
    @staticmethod
    def load_scenario(
        filepath: Union[str, Path],
        format: str = "auto",
        scenario_index: int = 0
    ) -> Scenario:
        """
        Load a single scenario from a file.
        
        Args:
            filepath: Input file path
            format: File format ('yaml', 'json', or 'auto')
            scenario_index: Index of scenario to load if file contains multiple
            
        Returns:
            Loaded scenario
            
        Raises:
            IndexError: If scenario_index is out of range
        """
        scenarios = ScenarioStorage.load_scenarios(filepath, format)
        
        if not scenarios:
            raise ValidationError("No scenarios found in file")
        
        if scenario_index < 0 or scenario_index >= len(scenarios):
            raise IndexError(
                f"Scenario index {scenario_index} out of range. "
                f"File contains {len(scenarios)} scenario(s)"
            )
        
        return scenarios[scenario_index]
    
    @staticmethod
    def create_template(
        filepath: Union[str, Path],
        format: str = "yaml"
    ) -> None:
        """
        Create a template scenario file with examples.
        
        Args:
            filepath: Output file path
            format: File format ('yaml' or 'json')
        """
        from stresstest.scenario.scenario_library import ScenarioLibrary
        
        # Create a few example scenarios
        scenarios = [
            ScenarioLibrary.market_crash(),
            ScenarioLibrary.vol_spike(),
        ]
        
        ScenarioStorage.save_scenarios(scenarios, filepath, format)


# Convenience functions for common use cases

def save_scenarios_yaml(scenarios: List[Scenario], filepath: Union[str, Path]) -> None:
    """Save scenarios to YAML file."""
    ScenarioStorage.save_scenarios(scenarios, filepath, format='yaml')


def save_scenarios_json(scenarios: List[Scenario], filepath: Union[str, Path]) -> None:
    """Save scenarios to JSON file."""
    ScenarioStorage.save_scenarios(scenarios, filepath, format='json')


def load_scenarios_yaml(filepath: Union[str, Path]) -> List[Scenario]:
    """Load scenarios from YAML file."""
    return ScenarioStorage.load_scenarios(filepath, format='yaml')


def load_scenarios_json(filepath: Union[str, Path]) -> List[Scenario]:
    """Load scenarios from JSON file."""
    return ScenarioStorage.load_scenarios(filepath, format='json')

