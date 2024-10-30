from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="jtransformer",
    version="0.1.0",
    author="Jackson Kunde",
    author_email="jkunde@wisc.edu",
    description="A lightweight GPT-2 style transformer implemented from scratch in PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/jtransformer",
    packages=find_packages(include=["jtransformer", "jtransformer.*"]),
    install_requires=[
        "einops==0.8.0",
        "jaxtyping==0.2.34",
        "torch==2.5.0",
        "numpy==2.1.2",
        "pytest==8.2.2",
        "wandb==0.18.5",
        "transformers==4.45.2",
        "datasets==3.0.2",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
