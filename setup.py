from setuptools import find_packages, setup

setup(
    name="archmind",
    version="0.1.0",
    description="Repository ingestion and parsing utilities",
    packages=find_packages(exclude=("env", "env.*")),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "tree-sitter>=0.21,<0.22",
        "tree-sitter-languages>=1.10.2,<2.0.0",
    ],
)
