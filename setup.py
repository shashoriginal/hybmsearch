from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="hybmsearch",
    version="1.0.0",
    author="Shashank Raj, Kalyanmoy Deb",
    description="Hybrid Bayesian Multi-Level Search: fast parallel search on billion-scale sorted arrays",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="CC BY-NC 4.0",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.21.0",
        "numba>=0.56.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "deap>=1.3.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
    ],
)
