from setuptools import setup, find_packages

__version__ = '0.1.0'
setup(
    name='kaggle-rig',
    version=__version__,
    python_requires='>=3.7.0',
    install_requires=[
        'pandas>=0.25.0',
    ],
    extras_require={
        'tests': [
            'mypy>=0.761',
            'pytest>=5.3.4',
        ]
    },
    packages=find_packages(exclude=['*.tests', '*.tests.*', 'tests.*', 'tests']),
    include_package_data=True,
    description='kaggle rig',
    license='MIT',
    author='seahrh',
    author_email='seahrh@gmail.com',
    url='https://github.com/seahrh/kaggle-rig'
)
