from setuptools import setup, find_packages

setup(
    name="gems",
    version="0.1.0",
    description="Geometric Constraints Enable Multi-Semantic Superposition in LLMs",
    author="Yu Deng",
    author_email="lulu663939@pm.me",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch==2.5.1",
        "transformers==5.9.0",
    ],
)
