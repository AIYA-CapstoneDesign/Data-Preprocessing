from setuptools import setup, find_packages

setup(
    name="data_preprocessing",
    version="0.1",
    description="Data preprocessing pipeline for fall detection",
    author="Mingyu Kim",
    packages=find_packages(),
    install_requires=open("requirements.txt").read().splitlines(),
    python_requires=">=3.8,<3.11",
)
