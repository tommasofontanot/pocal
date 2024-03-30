import setuptools
from setuptools import find_packages, setup

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "pocal",
    version = "2.236",
    description = "POCAL (Python Optical Coating Analysis Library)",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    packages=find_packages(),
    classifiers = [
        "Programming Language :: Python",
	"License :: OSI Approved :: MIT License",
	"Topic :: Scientific/Engineering :: Physics"
    ]
)