# Carotid segmentation

This repo contains the source code of the winning algorithm 
of the MICCAI Grand Challenge [Carotid Artery Vessel Wall Segmentation Challenge](https://vessel-wall-segmentation.grand-challenge.org/).
This algorithm computes the contours of the lumens and walls of the internal and external carotids 
on both sides of the neck from 3D black-blood MRI.

Documentation can be found on [ReadTheDocs](https://carotidsegmentation.readthedocs.io/en/latest/).

![Illustration of the pipeline](docs/images/global_illustration.png)


## Installation

As the package relies on multiple dependencies, we advise to install it in a separate environment:

```console
conda create -n carotid-segmentation python=3.9
conda activate carotid-segmentation
```

The package was not published on PyPi (yet), hence the repo must be cloned to install it:

```console
git clone git@github.com:MIAGroupUT/carotid-segmentation.git
cd carotid-segmentation
pip install [-e] ./src
```

## Usage

This package relies on pre-trained deep learning models that are currently stored on SurfDrive.

### Getting the models

- On Mac / Linux:

You can use the following command to fetch all the models:
```
make get-models
```

- On Windows:

Models can be downloaded with [this link](https://surfdrive.surf.nl/files/index.php/s/ywbP34RlUyuMIZU/download)
Unzip the tar file and copy the different folders in the `models` folder at the root of the repo.

The final architecture of your repo should be the following:
```
carotid-segmentation
├── models
│       ├── contour_transform
│       ├── contour_transform_dropout
│       └── heatmap_transform
│               ├── <filename1>.pt
│               ├── ....
│               └── <filenameN>.pt
...
```

### Organize your data

The algorithms were trained to segment carotids from 3D black-blood MRI volumes. 
The raw data can be provided as DICOM, MHD or MHA files.

Structure of the raw directory for DICOM files:

```console
raw_dir
├── <participant1>
│       ├── <filename1>.dcm
│       ...
│       └── <filenameN>.dcm
...
└── <participantN>
        ├── <filename1>.dcm
        ...
        └── <filenameN>.dcm
```

Structure of the raw directory for MHA/MHD files:

```console
raw_dir
├── <participant1>.mha
...
└── <participantN>.mha
```

### Run the pipeline

You can run the full pipeline using the following command:
```console
carotid pipeline_transform \
    /path/to/raw_dir \
    <repo-path>/models/heatmap_transform \
    <repo-path>/models/contour_transform \
    /path/to/output
```

## Test

The package is tested with the CI of Gitlab. You can also run yourself the tests with `pytest`.
First install the requirements for the tests in your conda environment:
```
conda activate carotid-segmentation
cd <repo-path>
pip install -r tests/requirements.txt
```

Then get the data necessary to run the tests.
- On Mac / Linux:

```
make prepare-test
```
- On Windows:

Download test data using this [link](https://surfdrive.surf.nl/files/index.php/s/CoazEglbyGXS23G/download).
Unzip the downloaded tar file and move the directories in the `tests` folder.
Create a `models` directory in the `tests` folder and copy one of each of the model contained in your root
`models` folder (you can also put all the models, but the tests take a longer time).

The final architecture should be the following:
```console
tests
├── centerline_transform
│       ├── input
│       ├── reference
│       ├── test_args.toml
│       └── test_centerline_transform.py
...
├── models
│       ├── contour_transform
│       ├── contour_transform_dropout
│       └── heatmap_transform
...
├── raw_dir
│       ├── 0_P125_U.mha
│       └── parameters.json
...
```

You can now run the tests with pytest:
```
pytest tests/
```

## Build docker container

The pipeline can be run on the [grand-challenge platform](https://grand-challenge.org/).
To update the current docker container and generate a new one from the current version of the code please run:
```
make build-grand-challenge
```
To build a lighter docker container (but not adapted to the grand-challenge platform):
```
make build
```

## Authors and acknowledgment
This algorithm was originally developed by researchers of the team [Mathematics of Imaging & AI](https://www.utwente.nl/en/eemcs/sacs/people/sort-chair/?category=mia) of the University of Twente
and presented in the [SPIE Medical Imaging conference in 2022](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12032/120320Y/Deep-learning-based-carotid-artery-vessel-wall-segmentation-in-black/10.1117/12.2611112.short?SSO=1).
