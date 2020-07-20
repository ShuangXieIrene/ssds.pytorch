import os
from setuptools import setup, find_packages

with open('./requirements.txt') as f:
    required_packages = f.read().splitlines()
with open('./extra_requirements.txt') as f:
    dependency_links = f.read().splitlines()

setup(name='ssds',
      version='1.5',
      description='Single Shot Detector and its variants',
      install_requires=required_packages,
      dependency_links=dependency_links,
      python_requires='>=3.6',
      packages=find_packages()
)