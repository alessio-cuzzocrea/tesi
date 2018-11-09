from setuptools import setup, find_packages



setup(
    name='bioinformatics_helpers',
    version='0.0.0',
    author='Alessio Cuzzocrea',
    packages=find_packages(exclude=('tests', 'docs'))
)