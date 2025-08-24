import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from distutils import log
from typing import List, Dict, Any

# Constants
PROJECT_NAME = "enhanced_cs"
VERSION = "1.0.0"
DESCRIPTION = "Enhanced AI project based on cs.RO_2508.15501v1_LLM-Driven-Self-Refinement-for-Embodied-Drone-Task"
AUTHOR = "Your Name"
AUTHOR_EMAIL = "your@email.com"
URL = "https://your-website.com"
LICENSE = "MIT"
REQUIRES_PYTHON = ">=3.8.0"
REQUIRED_PACKAGES = [
    "torch",
    "numpy",
    "pandas",
]

# Setup configuration
class CustomInstallCommand(install):
    """Custom install command to handle additional setup tasks."""
    def run(self):
        install.run(self)
        log.info("Running custom install tasks...")

class CustomDevelopCommand(develop):
    """Custom develop command to handle additional development setup tasks."""
    def run(self):
        develop.run(self)
        log.info("Running custom develop tasks...")

class CustomEggInfoCommand(egg_info):
    """Custom egg info command to handle additional egg info tasks."""
    def run(self):
        egg_info.run(self)
        log.info("Running custom egg info tasks...")

def get_package_data() -> Dict[str, List[str]]:
    """Get package data."""
    package_data = {}
    for root, dirs, files in os.walk("enhanced_cs"):
        package_data[root] = [f for f in files if not f.endswith(".py")]
    return package_data

def get_long_description() -> str:
    """Get long description from README file."""
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
    return long_description

def main():
    """Main setup function."""
    setup(
        name=PROJECT_NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        license=LICENSE,
        python_requires=REQUIRES_PYTHON,
        packages=find_packages(exclude=["tests", "tests.*"]),
        package_data=get_package_data(),
        install_requires=REQUIRED_PACKAGES,
        include_package_data=True,
        zip_safe=False,
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
        ],
        cmdclass={
            "install": CustomInstallCommand,
            "develop": CustomDevelopCommand,
            "egg_info": CustomEggInfoCommand,
        },
    )

if __name__ == "__main__":
    main()