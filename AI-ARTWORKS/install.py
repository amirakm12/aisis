#!/usr/bin/env python3
"""
AI-ARTWORK Installer Launcher
Simple launcher for the enhanced installer with command-line options
"""

import sys
import argparse
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

try:
    from enhanced_installer import EnhancedInstaller, InstallationConfig, Colors
except ImportError as e:
    print(f"Error importing enhanced installer: {e}")
    print("Please ensure you're running this from the AI-ARTWORK root directory")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="AI-ARTWORK Enhanced Installer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python install.py                           # Interactive installation
  python install.py --path ~/AI-ARTWORK      # Custom installation path
  python install.py --no-shortcuts           # Skip shortcut creation
  python install.py --no-gpu                 # Disable GPU support
  python install.py --models whisper-tiny    # Install specific models
  python install.py --dev                    # Development mode
        """
    )
    
    parser.add_argument(
        "--path", 
        type=Path,
        default=Path.home() / "AI-ARTWORK",
        help="Installation directory (default: ~/AI-ARTWORK)"
    )
    
    parser.add_argument(
        "--no-shortcuts",
        action="store_true",
        help="Skip creating desktop and start menu shortcuts"
    )
    
    parser.add_argument(
        "--no-path",
        action="store_true", 
        help="Don't add installation to system PATH"
    )
    
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU support and CUDA packages"
    )
    
    parser.add_argument(
        "--models",
        nargs="*",
        default=["whisper-base", "llama-2-7b-chat"],
        help="AI models to install (default: whisper-base llama-2-7b-chat)"
    )
    
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable development mode"
    )
    
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Automatically start AI-ARTWORK after installation"
    )
    
    parser.add_argument(
        "--optional-deps",
        action="store_true",
        help="Install optional dependencies"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimize output (quiet mode)"
    )
    
    return parser.parse_args()

def main():
    """Main installer launcher"""
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                   AI-ARTWORK Installer                       ║")
    print("║                                                              ║")
    print("║  🎨 Enhanced Installation with Smart Features                ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}\n")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Display installation configuration
    if not args.quiet:
        print(f"{Colors.BLUE}📋 Installation Configuration:{Colors.END}")
        print(f"  Path: {args.path}")
        print(f"  Shortcuts: {'No' if args.no_shortcuts else 'Yes'}")
        print(f"  Add to PATH: {'No' if args.no_path else 'Yes'}")
        print(f"  GPU Support: {'No' if args.no_gpu else 'Yes'}")
        print(f"  Models: {', '.join(args.models) if args.models else 'None'}")
        print(f"  Development Mode: {'Yes' if args.dev else 'No'}")
        print(f"  Optional Dependencies: {'Yes' if args.optional_deps else 'No'}")
        print()
    
    # Create installation configuration
    config = InstallationConfig(
        install_path=args.path,
        create_shortcuts=not args.no_shortcuts,
        add_to_path=not args.no_path,
        install_models=args.models,
        gpu_support=not args.no_gpu,
        development_mode=args.dev,
        auto_start=args.auto_start,
        install_optional_dependencies=args.optional_deps
    )
    
    # Confirm installation unless in quiet mode
    if not args.quiet:
        response = input(f"Proceed with installation? {Colors.GREEN}(Y/n){Colors.END}: ").strip().lower()
        if response and response not in ['y', 'yes']:
            print("Installation cancelled.")
            return 0
    
    # Run installation
    installer = EnhancedInstaller()
    success = installer.run_installation(config)
    
    if success:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 AI-ARTWORK installed successfully!{Colors.END}")
        
        if args.auto_start:
            print(f"\n{Colors.CYAN}🚀 Starting AI-ARTWORK...{Colors.END}")
            try:
                import subprocess
                subprocess.Popen([
                    sys.executable, 
                    str(args.path / "launch.py")
                ])
            except Exception as e:
                print(f"{Colors.YELLOW}⚠️  Failed to auto-start: {e}{Colors.END}")
        
        return 0
    else:
        print(f"\n{Colors.RED}❌ Installation failed!{Colors.END}")
        return 1

if __name__ == "__main__":
    sys.exit(main())