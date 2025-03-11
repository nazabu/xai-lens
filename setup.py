from setuptools import setup, find_packages

setup(
    name="xai-lens",
    version="0.1.0",
    description="A universal explainability analysis package for ML models",
    author="Abu",
    url="https://github.com/nazabu/xai-lens",
    packages=find_packages(),
    install_requires=[
        "shap",
        "lime",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "ipython",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
