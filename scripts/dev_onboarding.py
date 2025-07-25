import sys
import subprocess

def check_python():
    if sys.version_info < (3, 10):
        print("Python 3.10+ required.")
        sys.exit(1)
    print(f"Python version: {sys.version}")

def install_dev_deps():
    print("Installing dev dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"])
    print("Dev dependencies installed.")

def main():
    check_python()
    install_dev_deps()
    print("Onboarding complete! Run 'pytest' to test your setup.")

if __name__ == "__main__":
    main() 