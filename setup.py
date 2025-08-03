from setuptools import find_packages, setup

__version__ = "0.8.0"
setup(
    name="sgcharts-ml",
    version=__version__,
    python_requires=">=3.9,<3.13",
    install_requires=[
        "numpy>=1.22.0,<2.0.0",
        "numba>=0.59.0",  # official support for python 3.12 https://github.com/numba/numba/issues/9197
        "pandas>=1.0.0",
        "scikit-learn>=1.3.0",
        "networkx>=3.2.1",  # last version supporting python 3.9
        "torch>=2.2.1",
    ],
    extras_require={
        "lint": [
            "black==25.1.0",
            "isort==6.0.1",
            "pre-commit==4.2.0",
            "flake8==7.3.0",
            "mypy==1.16.1",
        ],
        "tests": [
            "pytest==8.3.3",
            "pytest-cov==6.0.0",
        ],
        "notebook": ["jupyterlab>=1.2.10", "tqdm>=4.45.0"],
        "nlp": ["transformers>=4.20.1", "spacy>=3.4.2,<3.8"],
    },
    packages=find_packages("src", exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    package_dir={"": "src"},
    include_package_data=True,
    description="sgcharts machine learning library",
    license="MIT",
    author="seahrh",
    author_email="seahrh@gmail.com",
    url="https://github.com/seahrh/sgcharts-ml",
)
