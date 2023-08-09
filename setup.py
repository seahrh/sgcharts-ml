from setuptools import find_packages, setup

__version__ = "0.6.0"
setup(
    name="sgcharts-ml",
    version=__version__,
    python_requires=">=3.9,<3.12",
    install_requires=[
        "numba>=0.52.0",
        "pandas>=1.0.0",
        "scikit-learn>=1.0.2",
        "networkx>=2.6.3",
        "torch>=1.11.0",
    ],
    extras_require={
        "lint": [
            "black==23.7.0",
            "isort==5.12.0",
            "pre-commit==3.3.3",
            "flake8==6.1.0",
            "mypy==1.4.1",
        ],
        "tests": [
            "pytest==7.4.0",
            "pytest-cov==4.1.0",
        ],
        "notebook": ["jupyterlab>=1.2.10", "tqdm>=4.45.0"],
        "transformers": ["transformers>=4.20.1"],
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
