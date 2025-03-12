from setuptools import setup, find_packages

setup(
    name="xai-lens",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A universal explainability analysis package for ML models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/XAI-Lens",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
