"""
Advanced Configuration Validation System
Provides comprehensive validation, security checks, and environment verification
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Type
from dataclasses import dataclass, field
from enum import Enum
import re
import ipaddress
from urllib.parse import urlparse
from loguru import logger

try:
    from pydantic import BaseModel, validator, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    logger.warning("Pydantic not available - advanced validation disabled")


class ValidationLevel(Enum):
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


class ConfigType(Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    PATH = "path"
    URL = "url"
    EMAIL = "email"
    IP_ADDRESS = "ip_address"
    PORT = "port"


@dataclass
class ValidationRule:
    """Configuration validation rule"""
    key: str
    config_type: ConfigType
    required: bool = True
    default: Any = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    validator_func: Optional[callable] = None
    description: str = ""
    security_sensitive: bool = False


@dataclass
class ValidationResult:
    """Validation result container"""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized_config: Dict[str, Any] = field(default_factory=dict)
    security_issues: List[str] = field(default_factory=list)


class ConfigValidator:
    """
    Comprehensive configuration validation system
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        """
        Initialize configuration validator
        
        Args:
            validation_level: Level of validation strictness
        """
        self.validation_level = validation_level
        self.rules: Dict[str, ValidationRule] = {}
        self.custom_validators: Dict[str, callable] = {}
        
        # Security patterns
        self.security_patterns = {
            'potential_injection': re.compile(r'[<>"\';\\]|script|exec|eval|system', re.IGNORECASE),
            'path_traversal': re.compile(r'\.\./|\.\.\\'),
            'suspicious_urls': re.compile(r'file://|ftp://|data:', re.IGNORECASE)
        }
        
        # Register default validators
        self._register_default_validators()
        
        logger.info(f"Config Validator initialized with {validation_level.value} validation level")
    
    def add_rule(self, rule: ValidationRule):
        """Add a validation rule"""
        self.rules[rule.key] = rule
        logger.debug(f"Added validation rule for {rule.key}")
    
    def add_rules(self, rules: List[ValidationRule]):
        """Add multiple validation rules"""
        for rule in rules:
            self.add_rule(rule)
    
    def register_validator(self, name: str, validator_func: callable):
        """Register a custom validator function"""
        self.custom_validators[name] = validator_func
        logger.debug(f"Registered custom validator: {name}")
    
    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """
        Validate configuration against defined rules
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Validation result
        """
        result = ValidationResult(valid=True)
        result.sanitized_config = config.copy()
        
        logger.info("Starting configuration validation")
        
        # Check required fields
        self._validate_required_fields(config, result)
        
        # Validate each configured value
        for key, value in config.items():
            if key in self.rules:
                self._validate_field(key, value, result)
            else:
                if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                    result.warnings.append(f"Unknown configuration key: {key}")
        
        # Add defaults for missing optional fields
        self._add_default_values(result)
        
        # Security validation
        if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
            self._security_validation(result)
        
        # Environment validation
        self._validate_environment(result)
        
        # Final validation status
        result.valid = len(result.errors) == 0
        
        if result.valid:
            logger.info("Configuration validation passed")
        else:
            logger.error(f"Configuration validation failed with {len(result.errors)} errors")
        
        return result
    
    def _validate_required_fields(self, config: Dict[str, Any], result: ValidationResult):
        """Validate that all required fields are present"""
        for key, rule in self.rules.items():
            if rule.required and key not in config:
                result.errors.append(f"Required field missing: {key}")
    
    def _validate_field(self, key: str, value: Any, result: ValidationResult):
        """Validate a single field"""
        rule = self.rules[key]
        
        try:
            # Type validation
            validated_value = self._validate_type(key, value, rule, result)
            if validated_value is None and key in result.errors:
                return
            
            # Range/length validation
            self._validate_constraints(key, validated_value, rule, result)
            
            # Pattern validation
            if rule.pattern and isinstance(validated_value, str):
                if not re.match(rule.pattern, validated_value):
                    result.errors.append(f"{key}: Value does not match required pattern")
            
            # Allowed values validation
            if rule.allowed_values and validated_value not in rule.allowed_values:
                result.errors.append(f"{key}: Value must be one of {rule.allowed_values}")
            
            # Custom validator
            if rule.validator_func:
                try:
                    if not rule.validator_func(validated_value):
                        result.errors.append(f"{key}: Failed custom validation")
                except Exception as e:
                    result.errors.append(f"{key}: Custom validator error: {e}")
            
            # Security validation for sensitive fields
            if rule.security_sensitive:
                self._validate_security_sensitive_field(key, validated_value, result)
            
            # Update sanitized config
            result.sanitized_config[key] = validated_value
            
        except Exception as e:
            result.errors.append(f"{key}: Validation error: {e}")
    
    def _validate_type(self, key: str, value: Any, rule: ValidationRule, result: ValidationResult) -> Any:
        """Validate and convert field type"""
        try:
            if rule.config_type == ConfigType.STRING:
                return str(value)
            
            elif rule.config_type == ConfigType.INTEGER:
                return int(value)
            
            elif rule.config_type == ConfigType.FLOAT:
                return float(value)
            
            elif rule.config_type == ConfigType.BOOLEAN:
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    return value.lower() in ['true', '1', 'yes', 'on']
                return bool(value)
            
            elif rule.config_type == ConfigType.LIST:
                if isinstance(value, list):
                    return value
                if isinstance(value, str):
                    # Try to parse as JSON array or comma-separated
                    try:
                        return json.loads(value)
                    except:
                        return [item.strip() for item in value.split(',')]
                return list(value)
            
            elif rule.config_type == ConfigType.DICT:
                if isinstance(value, dict):
                    return value
                if isinstance(value, str):
                    return json.loads(value)
                return dict(value)
            
            elif rule.config_type == ConfigType.PATH:
                path_str = str(value)
                path = Path(path_str)
                if not path.exists() and self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                    result.warnings.append(f"{key}: Path does not exist: {path_str}")
                return path_str
            
            elif rule.config_type == ConfigType.URL:
                url_str = str(value)
                parsed = urlparse(url_str)
                if not parsed.scheme or not parsed.netloc:
                    result.errors.append(f"{key}: Invalid URL format")
                    return None
                return url_str
            
            elif rule.config_type == ConfigType.EMAIL:
                email_str = str(value)
                email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
                if not email_pattern.match(email_str):
                    result.errors.append(f"{key}: Invalid email format")
                    return None
                return email_str
            
            elif rule.config_type == ConfigType.IP_ADDRESS:
                ip_str = str(value)
                try:
                    ipaddress.ip_address(ip_str)
                    return ip_str
                except ValueError:
                    result.errors.append(f"{key}: Invalid IP address")
                    return None
            
            elif rule.config_type == ConfigType.PORT:
                port_num = int(value)
                if not (1 <= port_num <= 65535):
                    result.errors.append(f"{key}: Port must be between 1 and 65535")
                    return None
                return port_num
            
            else:
                return value
                
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            result.errors.append(f"{key}: Type conversion error: {e}")
            return None
    
    def _validate_constraints(self, key: str, value: Any, rule: ValidationRule, result: ValidationResult):
        """Validate value constraints"""
        # Range validation for numeric types
        if isinstance(value, (int, float)):
            if rule.min_value is not None and value < rule.min_value:
                result.errors.append(f"{key}: Value {value} is below minimum {rule.min_value}")
            if rule.max_value is not None and value > rule.max_value:
                result.errors.append(f"{key}: Value {value} is above maximum {rule.max_value}")
        
        # Length validation for strings and collections
        if hasattr(value, '__len__'):
            length = len(value)
            if rule.min_length is not None and length < rule.min_length:
                result.errors.append(f"{key}: Length {length} is below minimum {rule.min_length}")
            if rule.max_length is not None and length > rule.max_length:
                result.errors.append(f"{key}: Length {length} is above maximum {rule.max_length}")
    
    def _validate_security_sensitive_field(self, key: str, value: Any, result: ValidationResult):
        """Validate security-sensitive fields"""
        if isinstance(value, str):
            # Check for potential injection attacks
            for pattern_name, pattern in self.security_patterns.items():
                if pattern.search(value):
                    result.security_issues.append(f"{key}: Potential {pattern_name} detected")
            
            # Check for suspicious content
            if len(value) > 10000:  # Unusually long values
                result.security_issues.append(f"{key}: Unusually long value (potential DoS)")
    
    def _add_default_values(self, result: ValidationResult):
        """Add default values for missing optional fields"""
        for key, rule in self.rules.items():
            if not rule.required and key not in result.sanitized_config and rule.default is not None:
                result.sanitized_config[key] = rule.default
                logger.debug(f"Added default value for {key}: {rule.default}")
    
    def _security_validation(self, result: ValidationResult):
        """Perform additional security validation"""
        # Check for common security misconfigurations
        config = result.sanitized_config
        
        # Debug mode in production
        if config.get('debug', False) and config.get('environment') == 'production':
            result.security_issues.append("Debug mode enabled in production environment")
        
        # Weak passwords
        password_fields = ['password', 'secret', 'key', 'token']
        for field in password_fields:
            if field in config:
                value = str(config[field])
                if len(value) < 8:
                    result.security_issues.append(f"{field}: Password/secret is too short")
                if value.lower() in ['password', 'admin', '123456', 'secret']:
                    result.security_issues.append(f"{field}: Using common/weak password")
        
        # Insecure protocols
        url_fields = [k for k, v in config.items() if isinstance(v, str) and v.startswith(('http://', 'ftp://'))]
        for field in url_fields:
            result.security_issues.append(f"{field}: Using insecure protocol")
    
    def _validate_environment(self, result: ValidationResult):
        """Validate environment-specific requirements"""
        config = result.sanitized_config
        
        # Check environment variables
        required_env_vars = config.get('required_env_vars', [])
        for env_var in required_env_vars:
            if env_var not in os.environ:
                result.errors.append(f"Required environment variable not set: {env_var}")
        
        # Check file permissions for sensitive files
        sensitive_files = config.get('sensitive_files', [])
        for file_path in sensitive_files:
            if os.path.exists(file_path):
                stat = os.stat(file_path)
                if stat.st_mode & 0o077:  # World or group readable
                    result.security_issues.append(f"Sensitive file has permissive permissions: {file_path}")
    
    def _register_default_validators(self):
        """Register default validation functions"""
        
        def validate_positive_number(value):
            return isinstance(value, (int, float)) and value > 0
        
        def validate_non_empty_string(value):
            return isinstance(value, str) and len(value.strip()) > 0
        
        def validate_directory_exists(value):
            return Path(value).is_dir()
        
        def validate_file_exists(value):
            return Path(value).is_file()
        
        self.custom_validators.update({
            'positive_number': validate_positive_number,
            'non_empty_string': validate_non_empty_string,
            'directory_exists': validate_directory_exists,
            'file_exists': validate_file_exists
        })
    
    def load_rules_from_file(self, rules_file: Union[str, Path]):
        """Load validation rules from a file"""
        rules_path = Path(rules_file)
        
        if not rules_path.exists():
            raise FileNotFoundError(f"Rules file not found: {rules_file}")
        
        try:
            if rules_path.suffix.lower() == '.json':
                with open(rules_path, 'r') as f:
                    rules_data = json.load(f)
            elif rules_path.suffix.lower() in ['.yml', '.yaml']:
                with open(rules_path, 'r') as f:
                    rules_data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported rules file format: {rules_path.suffix}")
            
            # Convert to ValidationRule objects
            rules = []
            for rule_data in rules_data.get('rules', []):
                rule = ValidationRule(
                    key=rule_data['key'],
                    config_type=ConfigType(rule_data['type']),
                    required=rule_data.get('required', True),
                    default=rule_data.get('default'),
                    min_value=rule_data.get('min_value'),
                    max_value=rule_data.get('max_value'),
                    min_length=rule_data.get('min_length'),
                    max_length=rule_data.get('max_length'),
                    pattern=rule_data.get('pattern'),
                    allowed_values=rule_data.get('allowed_values'),
                    description=rule_data.get('description', ''),
                    security_sensitive=rule_data.get('security_sensitive', False)
                )
                rules.append(rule)
            
            self.add_rules(rules)
            logger.info(f"Loaded {len(rules)} validation rules from {rules_file}")
            
        except Exception as e:
            logger.error(f"Failed to load rules from {rules_file}: {e}")
            raise
    
    def generate_config_template(self, output_file: Union[str, Path]):
        """Generate a configuration template based on validation rules"""
        template = {
            "_description": "Configuration template generated from validation rules",
            "_validation_level": self.validation_level.value
        }
        
        for key, rule in self.rules.items():
            section = template
            
            # Create nested structure for dotted keys
            parts = key.split('.')
            for part in parts[:-1]:
                if part not in section:
                    section[part] = {}
                section = section[part]
            
            final_key = parts[-1]
            
            # Add field with description and default
            field_info = {
                "_description": rule.description or f"{rule.config_type.value} field",
                "_required": rule.required,
                "_type": rule.config_type.value
            }
            
            if rule.default is not None:
                field_info["_default"] = rule.default
                section[final_key] = rule.default
            else:
                section[final_key] = field_info
        
        # Write template
        output_path = Path(output_file)
        if output_path.suffix.lower() == '.json':
            with open(output_path, 'w') as f:
                json.dump(template, f, indent=2)
        else:
            with open(output_path, 'w') as f:
                yaml.dump(template, f, default_flow_style=False, indent=2)
        
        logger.info(f"Generated configuration template: {output_file}")


def create_default_rules() -> List[ValidationRule]:
    """Create default validation rules for common configuration"""
    return [
        ValidationRule(
            key="app.name",
            config_type=ConfigType.STRING,
            required=True,
            min_length=1,
            max_length=100,
            description="Application name"
        ),
        ValidationRule(
            key="app.version",
            config_type=ConfigType.STRING,
            required=True,
            pattern=r'^\d+\.\d+\.\d+$',
            description="Application version (semantic versioning)"
        ),
        ValidationRule(
            key="app.debug",
            config_type=ConfigType.BOOLEAN,
            required=False,
            default=False,
            description="Enable debug mode"
        ),
        ValidationRule(
            key="server.host",
            config_type=ConfigType.IP_ADDRESS,
            required=False,
            default="127.0.0.1",
            description="Server host address"
        ),
        ValidationRule(
            key="server.port",
            config_type=ConfigType.PORT,
            required=False,
            default=8000,
            description="Server port number"
        ),
        ValidationRule(
            key="database.url",
            config_type=ConfigType.URL,
            required=True,
            description="Database connection URL"
        ),
        ValidationRule(
            key="models.cache_dir",
            config_type=ConfigType.PATH,
            required=False,
            default="./models",
            description="Directory for model cache"
        ),
        ValidationRule(
            key="memory.max_usage_gb",
            config_type=ConfigType.FLOAT,
            required=False,
            default=8.0,
            min_value=1.0,
            max_value=64.0,
            description="Maximum memory usage in GB"
        ),
        ValidationRule(
            key="security.secret_key",
            config_type=ConfigType.STRING,
            required=True,
            min_length=32,
            security_sensitive=True,
            description="Secret key for encryption"
        )
    ]


# Global validator instance
config_validator = ConfigValidator()


def setup_config_validation(validation_level: ValidationLevel = ValidationLevel.STANDARD) -> ConfigValidator:
    """
    Setup global configuration validation
    
    Args:
        validation_level: Validation strictness level
        
    Returns:
        Configured validator
    """
    global config_validator
    config_validator = ConfigValidator(validation_level)
    
    # Add default rules
    default_rules = create_default_rules()
    config_validator.add_rules(default_rules)
    
    return config_validator