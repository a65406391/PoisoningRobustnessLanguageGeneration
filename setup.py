"""Setup for pip package."""

from setuptools import find_packages, setup

setup(
    name="certifiable_learning_stability",
    version="0.1.0",
    description="A library for certifying learning stability using Renyi divergences",
    url="<anonymized>",
    author="Anonymous",
    author_email="anonymous@example.com",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch >= 2.4.1",
        "torchvision >= 0.19",
        "pydantic >= 2.9",
        "gurobipy >= 10.0",
        "numpy",
        "scipy >= 1.14",
        "opacus >= 1.5",
        "seaborn >= 0.13",
        "matplotlib >= 3.9",
    ],
    extras_require={
        "experiments": [],
    },
    platforms=["any"],
)
