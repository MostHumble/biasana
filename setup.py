import pathlib
from setuptools import find_packages, setup


def get_version() -> str:
    rel_path = "src/biasana/__init__.py"
    with open(rel_path, "r") as fp:
        for line in fp.read().splitlines():
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


setup(
    name="biasana",
    version=get_version(),
    description="A package for analyzing bias in textual data.",
    long_description=pathlib.Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    Homepage="https://github.com/MostHumble/biasana/",
    url="https://github.com/MostHumble/biasana/",
    Issues="https://github.com/MostHumble/biasana/issues",
    authors=[{"name": "sifal klioui", "email": "sifalklioui@yahoo.com"}],
    author_email="sifalklioui@yahoo.com",
    license="Apache 2.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    classifiers=[
        "Topic :: Utilities",
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=[
        "setuptools",
        "wheel",
        "typing",
        "scikit-learn",
        "numpy",
        "spacy",
    ],
)