from setuptools import setup, find_packages

__version__ = "0.2.0"
setup(
    name="kaggle-rig",
    version=__version__,
    python_requires="~=3.7",
    install_requires=["pandas~=1.1.1", "scikit-learn~=0.23.2"],
    extras_require={
        "tests": ["black~=19.10b0", "mypy>=0.780", "pytest>=5.4.2", "pytest-cov>=2.9.0"]
    },
    packages=find_packages("src", exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    package_dir={"": "src"},
    include_package_data=True,
    description="kaggle rig",
    license="MIT",
    author="seahrh",
    author_email="seahrh@gmail.com",
    url="https://github.com/seahrh/kaggle-rig",
)
