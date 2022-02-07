import pathlib

# The directory containing this file
PARENT_DIR = pathlib.Path(__file__).parent

# The text of the README file
README = (PARENT_DIR / "README.rst").read_text()

SETUP_ARGS = dict(
    name="GeneticAlgos",
    version="1.0.2",
    description="Simple and powerful Python library for creating genetic algorithms.",
    long_description=README,
    long_description_content_type="text/x-rst",
    url="https://geneticalgos.readthedocs.io/en/latest/",
    author="Lukas Kozelnicky",
    author_email="python@kozelnicky.com",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    py_modules=[
        "geneticalgos",
    ],
    install_requires=[
        "numpy",
    ],
    project_urls={
        "Documentation": "https://geneticalgos.readthedocs.io/en/latest/",
        "Source": "https://github.com/lkozelnicky/GeneticAlgos",
    },
)

if __name__ == "__main__":
    from setuptools import setup, find_packages

    SETUP_ARGS["packages"] = find_packages()
    setup(**SETUP_ARGS)
