from setuptools import find_packages, setup

__version__ = "0.5.0"
setup(
    name="sgcharts-ml",
    version=__version__,
    python_requires="~=3.7",
    install_requires=["numba>=0.52.0", "pandas>=1.0.0", "scikit-learn>=0.23.0"],
    extras_require={
        "lint": ["pre-commit==2.17.0", "black==22.1.0"],
        "tests": [
            "mypy>=0.780",
            "pytest>=5.4.2",
            "pytest-cov>=2.9.0",
            "tensorflow>=2.1.0",
        ],
        "notebook": ["jupyterlab>=1.2.10", "tqdm>=4.45.0"],
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
