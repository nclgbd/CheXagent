"""Python `setup.py` for `rtk` package. Adapted from https://github.com/rochacbruno/python-project-template/blob/main/setup.py."""
import os
from setuptools import setup, find_packages

setup(
    version="0.1.0",
    author="",
    description="",
    name="chexagent",
    entry_points={
        "console_scripts": [
            f"example=demos.run_examples:main",
        ],
    },
    long_description="",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    scripts=["./demos/run_examples.py"]
)
