from setuptools import setup, find_packages

setup(
    name="aisis",
    version="0.1.0",
    description="AI Creative Studio with GPU acceleration and voice interaction",
    author="AISIS Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "loguru>=0.7.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "psutil>=5.9.0",
        "PySide6>=6.5.0",
        "QtPy>=2.3.0",
        "whisper>=1.0.0",
        "soundfile>=0.12.0",
        "bark>=0.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-qt>=4.2.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pylint>=2.17.0",
            "mypy>=1.0.0",
        ]
    },
)
