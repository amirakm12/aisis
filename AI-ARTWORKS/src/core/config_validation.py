"""
Configuration validation for Al-artworks

This module provides configuration validation using Pydantic models
to ensure proper configuration of the Al-artworks system.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
import logging

logger = logging.getLogger(__name__)

class AlArtworksConfig(BaseModel):
    """
    Configuration model for Al-artworks system
    
    Validates and provides type-safe access to configuration settings
    """
    
    # Application settings
    app_name: str = Field(default="Al-artworks", description="Application name")
    version: str = Field(default="3.0.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # AI/ML settings
    model_path: Optional[str] = Field(default=None, description="Path to AI models")
    device: str = Field(default="auto", description="Processing device (cpu/gpu/auto)")
    max_workers: int = Field(default=4, description="Maximum worker threads")
    
    # UI settings
    theme: str = Field(default="dark", description="UI theme (light/dark)")
    language: str = Field(default="en", description="Interface language")
    
    # Processing settings
    quality: str = Field(default="high", description="Processing quality (low/medium/high)")
    timeout: float = Field(default=30.0, description="Operation timeout in seconds")
    
    # Logging settings
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    
    # Voice settings
    voice_enabled: bool = Field(default=True, description="Enable voice input")
    voice_device: Optional[str] = Field(default=None, description="Voice input device")
    
    # Integration settings
    integrations: Dict[str, Any] = Field(default_factory=dict, description="External integrations")
    
    @validator('device')
    def validate_device(cls, v):
        """Validate device setting"""
        valid_devices = ['cpu', 'gpu', 'auto']
        if v.lower() not in valid_devices:
            raise ValueError(f"Device must be one of: {valid_devices}")
        return v.lower()
    
    @validator('theme')
    def validate_theme(cls, v):
        """Validate theme setting"""
        valid_themes = ['light', 'dark']
        if v.lower() not in valid_themes:
            raise ValueError(f"Theme must be one of: {valid_themes}")
        return v.lower()
    
    @validator('quality')
    def validate_quality(cls, v):
        """Validate quality setting"""
        valid_qualities = ['low', 'medium', 'high']
        if v.lower() not in valid_qualities:
            raise ValueError(f"Quality must be one of: {valid_qualities}")
        return v.lower()
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level setting"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    class Config:
        """Pydantic configuration"""
        extra = "forbid"  # Reject unknown fields
        validate_assignment = True  # Validate on assignment

class ValidationError(Exception):
    """Custom validation error for Al-artworks configuration"""
    pass

def validate_config(config_dict: Dict[str, Any]) -> AlArtworksConfig:
    """
    Validate configuration dictionary
    
    Args:
        config_dict: Configuration dictionary to validate
        
    Returns:
        Validated AlArtworksConfig object
        
    Raises:
        ValidationError: If configuration is invalid
    """
    try:
        return AlArtworksConfig(**config_dict)
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise ValidationError(f"Invalid configuration: {e}")

def load_config_from_file(file_path: str) -> AlArtworksConfig:
    """
    Load and validate configuration from file
    
    Args:
        file_path: Path to configuration file
        
    Returns:
        Validated AlArtworksConfig object
        
    Raises:
        ValidationError: If configuration file is invalid
    """
    import json
    
    try:
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        return validate_config(config_dict)
        
    except FileNotFoundError:
        logger.warning(f"Configuration file not found: {file_path}")
        return AlArtworksConfig()  # Return default config
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        raise ValidationError(f"Invalid JSON in configuration file: {e}")
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise ValidationError(f"Failed to load configuration: {e}")

def save_config_to_file(config: AlArtworksConfig, file_path: str) -> None:
    """
    Save configuration to file
    
    Args:
        config: AlArtworksConfig object to save
        file_path: Path to save configuration file
    """
    import json
    
    try:
        config_dict = config.dict()
        
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to: {file_path}")
        
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        raise

def get_default_config() -> AlArtworksConfig:
    """
    Get default configuration
    
    Returns:
        Default AlArtworksConfig object
    """
    return AlArtworksConfig()

# Example usage:
#     config = AlArtworksConfig(**your_dict) 