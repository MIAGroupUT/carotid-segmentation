from setuptools import setup, find_packages, dist

VERSION = "0.1.0"

dist.Distribution().fetch_build_eggs(['numpy==1.22.4'])

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
    description="Utilitaries for carotid segmentation project",
    keywords="carotid segmentation deep learning",
    license="MIT",
)
