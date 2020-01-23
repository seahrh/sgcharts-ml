from setuptools import setup, find_packages
__version__ = '1.0.0'
setup(
    name='kaggle-rig',
    version=__version__,
    python_requires='>=3.7.0',
    install_requires=[],
    packages=find_packages(exclude=['*.tests', '*.tests.*', 'tests.*', 'tests']),
    include_package_data=True,
    description='kaggle rig',
    license='MIT',
    author='seahrh',
    author_email='seahrh@gmail.com',
    url='https://github.com/seahrh/kaggle-rig'
)
