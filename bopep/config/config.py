"""
Configuration management for BoGA.

Provides a Config class that:
1. Loads default parameters from YAML
2. Allows user overrides via dict or another YAML file
3. Saves the final configuration to output directory for reproducibility
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from copy import deepcopy


class Config:
    """
    Configuration manager for BoGA that handles loading, merging, and saving configs.
    """
    
    def __init__(
        self,
        script: str,
        user_overrides: Optional[Union[Dict[str, Any], str, Path]] = None
    ):
        """
        Initialize configuration with defaults and optional user overrides.
        """
        # Load defaults
        if script == "BoGA":
            # Use built-in defaults from bopep/config/boga_defaults.yaml
            defaults_path = Path(__file__).parent / "boga_defaults.yaml"
        elif script == "BoPep":
            # Use built-in defaults from bopep/config/bopep_defaults.yaml
            defaults_path = Path(__file__).parent / "bopep_defaults.yaml"
        else:
            raise ValueError(f"Unknown script type: {script}. Must be 'BoGA' or 'BoPep'")
        
        self.defaults_path = Path(defaults_path)
        self.config = self._load_yaml(self.defaults_path)
        
        # Apply user overrides
        if user_overrides is not None:
            if isinstance(user_overrides, (str, Path)):
                # Load from YAML file
                override_dict = self._load_yaml(user_overrides)
            elif isinstance(user_overrides, dict):
                override_dict = user_overrides
            else:
                raise ValueError(
                    f"user_overrides must be dict or path to YAML file, got {type(user_overrides)}"
                )
            
            self._deep_merge(self.config, override_dict)
    
    def _load_yaml(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load YAML file and return as dict."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """
        Deep merge override dict into base dict (in-place).
        
        For nested dicts, recursively merge. For other types, override replaces base.
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                self._deep_merge(base[key], value)
            else:
                # Override value
                base[key] = deepcopy(value)
    
    def get(self, key: str) -> Any:
        """
        Get config value by key. Supports nested keys with dot notation.
        All defaults should be defined in the YAML file.
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set config value by key. Supports nested keys with dot notation.
        """
        keys = key.split('.')
        target = self.config
        
        # Navigate to the parent dict
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        # Set the final value
        target[keys[-1]] = value
    
    def update_from_used_values(self, **kwargs) -> None:
        """
        Update config with actually-used parameter values.
        Useful for updating config after BoGA initialization to capture user overrides.
        """
        for key, value in kwargs.items():
            if value is not None:  # Only update if value was actually set
                self.set(key, value)
    
    def save(self, output_dir: Union[str, Path], filename: str = "boga_config_used.yaml") -> Path:
        """
        Save the current configuration to a YAML file in the output directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / filename
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        
        return output_path
    
    def to_dict(self) -> Dict[str, Any]:
        """Return a deep copy of the config as a dict."""
        return deepcopy(self.config)
    
    def flatten(self, prefix: str = '') -> Dict[str, Any]:
        """
        Flatten nested config into a single-level dict with dot-notation keys.
        """
        def _flatten_dict(d: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(_flatten_dict(v, new_key).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        return _flatten_dict(self.config, prefix)
    
    def __repr__(self) -> str:
        return f"Config({self.config})"
    
    def __str__(self) -> str:
        return yaml.dump(self.config, default_flow_style=False, sort_keys=False)
