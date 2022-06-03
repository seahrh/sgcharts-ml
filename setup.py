from setuptools import find_packages, setup

__version__ = "0.5.0"
setup(
    name="sgcharts-ml",
    version=__version__,
    python_requires=">=3.7",
    install_requires=["numba>=0.52.0", "pandas>=1.0.0", "scikit-learn>=0.23.0"],
    extras_require={
        "lint": [
            "black==22.3.0",
            "isort==5.10.1",
            "pre-commit==2.19.0",
            "flake8==4.0.1",
        ],
        "tests": [
            "mypy==0.960",
            "pytest==7.1.2",
            "pytest-cov==3.0.0",
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
