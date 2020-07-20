import os
from setuptools import setup, find_packages

with open('./requirements.txt') as f:
    required_packages = f.read().splitlines()
dependency_links = [required_packages.pop()[4:]]

setup(name='ssds',
      version='1.5',
      description='Single Shot Detector and its variants',
      install_requires=required_packages,
      dependency_links=dependency_links,
      python_requires='>=3.6',
      packages=find_packages()
)