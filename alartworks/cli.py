"""
Al-artworks Command Line Interface
Comprehensive CLI for all Al-artworks functionality
"""

import click
import sys
import json
from pathlib import Path
from typing import Optional, List
import asyncio
from loguru import logger

from . import aisis, __version__
from src.core.config import config

@click.group()
@click.version_option(version=__version__, prog_name="AISIS")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config-file', '-c', type=click.Path(), help='Path to config file')
def cli(verbose: bool, config_file: Optional[str]):
    """AISIS - AI Creative Studio Command Line Interface"""
    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    if config_file:
        config.load_from_file(config_file)

@cli.command()
def gui():
    """Launch the AISIS GUI application"""
    try:
        aisis.initialize()
        window = aisis.create_gui()
        
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        window.show()
        app.exec()
        
    except Exception as e:
        logger.error(f"Failed to launch GUI: {e}")
        sys.exit(1)

@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--operations', '-op', multiple=True, help='Operations to perform')
@click.option('--agent', '-a', help='Specific agent to use')
@click.option('--quality', '-q', type=click.Choice(['low', 'medium', 'high']), default='medium')
def process(image_path: str, output: Optional[str], operations: List[str], agent: Optional[str], quality: str):
    """Process an image using AISIS agents"""
    try:
        aisis.initialize()
        
        # Set quality parameters
        config.set_quality_preset(quality)
        
        # Process the image
        result = aisis.process_image(
            image_path=image_path,
            operations=list(operations) if operations else None,
            agent=agent
        )
        
        # Save result
        if output:
            output_path = Path(output)
        else:
            input_path = Path(image_path)
            output_path = input_path.parent / f"{input_path.stem}_processed{input_path.suffix}"
        
        result.save(output_path)
        click.echo(f"Processed image saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)

@cli.command()
def agents():
    """List available agents and their capabilities"""
    try:
        aisis.initialize()
        available_agents = aisis.get_available_agents()
        
        click.echo("Available AISIS Agents:")
        click.echo("=" * 50)
        
        for agent_name, agent_info in available_agents.items():
            click.echo(f"\n{agent_name}:")
            click.echo(f"  Description: {agent_info.get('description', 'No description')}")
            click.echo(f"  Capabilities: {', '.join(agent_info.get('capabilities', []))}")
            click.echo(f"  Model: {agent_info.get('model', 'Unknown')}")
            
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        sys.exit(1)

@cli.command()
@click.option('--download', '-d', is_flag=True, help='Download missing models')
@click.option('--validate', '-v', is_flag=True, help='Validate existing models')
def models(download: bool, validate: bool):
    """Manage AISIS models"""
    try:
        aisis.initialize()
        model_manager = aisis.model_manager
        
        if download:
            click.echo("Downloading missing models...")
            model_manager.download_missing_models()
            click.echo("Model download complete")
        
        if validate:
            click.echo("Validating models...")
            results = model_manager.validate_models()
            
            for model_name, is_valid in results.items():
                status = "✓" if is_valid else "✗"
                click.echo(f"  {status} {model_name}")
        
        # List available models
        models = model_manager.list_models()
        click.echo("\nAvailable Models:")
        click.echo("=" * 30)
        
        for model in models:
            click.echo(f"  {model['name']} ({model['size']})")
            
    except Exception as e:
        logger.error(f"Model management failed: {e}")
        sys.exit(1)

@cli.command()
@click.option('--list', '-l', is_flag=True, help='List installed plugins')
@click.option('--install', '-i', help='Install plugin from path or URL')
@click.option('--uninstall', '-u', help='Uninstall plugin by name')
@click.option('--enable', '-e', help='Enable plugin by name')
@click.option('--disable', '-d', help='Disable plugin by name')
def plugins(list: bool, install: Optional[str], uninstall: Optional[str], 
           enable: Optional[str], disable: Optional[str]):
    """Manage AISIS plugins"""
    try:
        aisis.initialize()
        plugin_manager = aisis.plugin_manager
        
        if list:
            plugins = plugin_manager.list_plugins()
            click.echo("Installed Plugins:")
            click.echo("=" * 30)
            
            for plugin in plugins:
                status = "enabled" if plugin['enabled'] else "disabled"
                click.echo(f"  {plugin['name']} ({plugin['version']}) - {status}")
                click.echo(f"    {plugin['description']}")
        
        if install:
            click.echo(f"Installing plugin from: {install}")
            plugin_manager.install_plugin(install)
            click.echo("Plugin installed successfully")
        
        if uninstall:
            click.echo(f"Uninstalling plugin: {uninstall}")
            plugin_manager.uninstall_plugin(uninstall)
            click.echo("Plugin uninstalled successfully")
        
        if enable:
            plugin_manager.enable_plugin(enable)
            click.echo(f"Plugin '{enable}' enabled")
        
        if disable:
            plugin_manager.disable_plugin(disable)
            click.echo(f"Plugin '{disable}' disabled")
            
    except Exception as e:
        logger.error(f"Plugin management failed: {e}")
        sys.exit(1)

@cli.command()
@click.option('--key', '-k', help='Configuration key to get/set')
@click.option('--value', '-v', help='Value to set')
@click.option('--list', '-l', is_flag=True, help='List all configuration')
@click.option('--reset', '-r', is_flag=True, help='Reset to defaults')
def config_cmd(key: Optional[str], value: Optional[str], list: bool, reset: bool):
    """Manage AISIS configuration"""
    try:
        if reset:
            config.reset_to_defaults()
            click.echo("Configuration reset to defaults")
            return
        
        if list:
            config_data = config.to_dict()
            click.echo("Current Configuration:")
            click.echo("=" * 30)
            click.echo(json.dumps(config_data, indent=2))
            return
        
        if key and value:
            config.set(key, value)
            config.save()
            click.echo(f"Set {key} = {value}")
        elif key:
            current_value = config.get(key)
            click.echo(f"{key} = {current_value}")
        else:
            click.echo("Please specify --key or use --list to see all configuration")
            
    except Exception as e:
        logger.error(f"Configuration management failed: {e}")
        sys.exit(1)

@cli.command()
def benchmark():
    """Run performance benchmarks"""
    try:
        aisis.initialize()
        
        click.echo("Running AISIS benchmarks...")
        
        # Device benchmark
        device_manager = aisis.device_manager
        device_info = device_manager.get_device_info()
        
        click.echo("\nDevice Information:")
        click.echo("=" * 30)
        click.echo(f"GPU: {device_info.get('gpu', 'None')}")
        click.echo(f"Memory: {device_info.get('memory', 'Unknown')}")
        click.echo(f"Compute Capability: {device_info.get('compute_capability', 'Unknown')}")
        
        # Model benchmarks
        from src.core.model_benchmarking import ModelBenchmarker
        benchmarker = ModelBenchmarker()
        results = benchmarker.run_benchmarks()
        
        click.echo("\nModel Performance:")
        click.echo("=" * 30)
        for model_name, metrics in results.items():
            click.echo(f"{model_name}:")
            click.echo(f"  Inference Time: {metrics.get('inference_time', 'N/A')}ms")
            click.echo(f"  Memory Usage: {metrics.get('memory_usage', 'N/A')}MB")
            click.echo(f"  Quality Score: {metrics.get('quality_score', 'N/A')}/100")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)

@cli.command()
@click.option('--output', '-o', type=click.Path(), help='Output file for report')
def health():
    """Run health check and generate report"""
    try:
        from health_check import AISISHealthChecker
        
        checker = AISISHealthChecker()
        report = checker.run_full_check()
        
        if output:
            with open(output, 'w') as f:
                json.dump(report, f, indent=2)
            click.echo(f"Health report saved to: {output}")
        else:
            click.echo("AISIS Health Check Report:")
            click.echo("=" * 40)
            
            if report['issues']:
                click.echo("\nIssues Found:")
                for issue in report['issues']:
                    click.echo(f"  ❌ {issue['message']}")
            
            if report['warnings']:
                click.echo("\nWarnings:")
                for warning in report['warnings']:
                    click.echo(f"  ⚠️  {warning['message']}")
            
            if not report['issues'] and not report['warnings']:
                click.echo("✅ All systems operational!")
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    cli()