"""
README.md Generator for Agent Project
=====================================

This module generates a README.md file for the agent project, providing an overview of the project's purpose, dependencies, and usage.

Classes:
    READMEGenerator
"""

import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class READMEGenerator:
    """
    Generates a README.md file for the agent project.

    Attributes:
        project_name (str): The name of the project.
        project_description (str): A brief description of the project.
        dependencies (List[str]): A list of dependencies required by the project.
        usage (str): Instructions on how to use the project.
    """

    def __init__(self, project_name: str, project_description: str, dependencies: List[str], usage: str):
        """
        Initializes the READMEGenerator instance.

        Args:
            project_name (str): The name of the project.
            project_description (str): A brief description of the project.
            dependencies (List[str]): A list of dependencies required by the project.
            usage (str): Instructions on how to use the project.
        """
        self.project_name = project_name
        self.project_description = project_description
        self.dependencies = dependencies
        self.usage = usage

    def generate_readme(self) -> str:
        """
        Generates the README.md content.

        Returns:
            str: The generated README.md content.
        """
        readme_content = f"# {self.project_name}\n\n"
        readme_content += f"{self.project_description}\n\n"
        readme_content += "## Dependencies\n\n"
        for dependency in self.dependencies:
            readme_content += f"* {dependency}\n"
        readme_content += "\n"
        readme_content += "## Usage\n\n"
        readme_content += self.usage
        return readme_content

    def write_to_file(self, filename: str) -> None:
        """
        Writes the generated README.md content to a file.

        Args:
            filename (str): The name of the file to write to.
        """
        try:
            with open(filename, 'w') as file:
                file.write(self.generate_readme())
            logging.info(f"README.md generated and written to {filename}")
        except Exception as e:
            logging.error(f"Error writing to file: {e}")

class ProjectConfig:
    """
    Represents the project configuration.

    Attributes:
        project_name (str): The name of the project.
        project_description (str): A brief description of the project.
        dependencies (List[str]): A list of dependencies required by the project.
        usage (str): Instructions on how to use the project.
    """

    def __init__(self, project_name: str, project_description: str, dependencies: List[str], usage: str):
        """
        Initializes the ProjectConfig instance.

        Args:
            project_name (str): The name of the project.
            project_description (str): A brief description of the project.
            dependencies (List[str]): A list of dependencies required by the project.
            usage (str): Instructions on how to use the project.
        """
        self.project_name = project_name
        self.project_description = project_description
        self.dependencies = dependencies
        self.usage = usage

def main() -> None:
    """
    The main entry point of the program.
    """
    project_config = ProjectConfig(
        project_name="Agent Project",
        project_description="This is an agent project.",
        dependencies=["torch", "numpy", "pandas"],
        usage="To use this project, simply run the main.py file."
    )
    readme_generator = READMEGenerator(
        project_name=project_config.project_name,
        project_description=project_config.project_description,
        dependencies=project_config.dependencies,
        usage=project_config.usage
    )
    readme_generator.write_to_file("README.md")

if __name__ == "__main__":
    main()