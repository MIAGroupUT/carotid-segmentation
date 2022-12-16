from setuptools import setup, find_packages

VERSION = "0.1.0"

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="carotid",
    version=VERSION,
    packages=find_packages(exclude=["tests"]),
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "carotid = carotid.cli:cli",
        ],
    },
    # metadata for upload to PyPI
    author="Elina Thibeau-Sutre",
    author_email="elina.ts@free.fr",
    description="Utilitaries for carotid segmentation explainbility project",
    keywords="carotid segmentation deep learning explainability",
    license="MIT",
)
