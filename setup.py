from setuptools import setup, find_packages

setup(
    name="neuromorphic",
    version="0.2.0",
    description="An autonomous neuromorphic computing primitive with ML-inspired infrastructure",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.0",
        "scipy>=1.18",
    ],
    extras_require={
        "dev": [
            "numba>=0.60",
            "pytest>=9.0",
            "hypothesis>=6.100",
            "pytest-benchmark>=5.0",
            "pyyaml>=6.0",
            "h5py>=3.10",
            "matplotlib>=3.0",
            "tensorboardX>=2.0",
            "wandb>=0.15",
        ]
    },
)
