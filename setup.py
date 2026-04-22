from setuptools import setup, find_packages

setup(
    name="xai-lens",
    version="0.1.0",
    author="nazabu",
    description="A universal explainability analysis package for ML models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nazabu/xai-lens",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "shap",
        "lime",
        "joblib",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
