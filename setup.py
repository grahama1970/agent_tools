from setuptools import setup, find_packages

setup(
    name="agent_tools",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "tree-sitter>=0.24.0",
        "tree-sitter-javascript>=0.20.0",
        "tree-sitter-typescript>=0.20.0",
        "loguru>=0.7.0",
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0"
    ],
    python_requires=">=3.8",
    description="Agent tools for code analysis and manipulation",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/agent_tools",
) 