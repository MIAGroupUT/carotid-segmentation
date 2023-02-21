# Documentation of the XXX method

This documentation describes how to install, use and test the source code of XXX, the winning algorithm 
of the MICCAI Grand Challenge [Carotid Artery Vessel Wall Segmentation Challenge](https://vessel-wall-segmentation.grand-challenge.org/).
This algorithm computes the contours of the lumens and walls of the internal and external carotids 
on both sides of the neck from 3D black-blood MRI.

<figure>
<img src="images/anatomical_description.png" alt="Illustration of the wall and lumen contours on an axial slice" style="width:100%">
<figcaption align = "center"><b>Figure originally created by the Grand Challenge organisers</b></figcaption>
</figure>

The method was originally developed in the team of [Mathematics of Imaging & AI](https://www.utwente.nl/en/eemcs/sacs/people/sort-chair/?category=mia)
presented in SPIE Medical Imaging ((Alblas et al., 2022))[https://ris.utwente.nl/ws/portalfiles/portal/283040086/120320Y_alblas_brune_wolterink.pdf]

It mainly consists of two steps:
1. A centerline is estimated for the external and internal carotids on both sides,
2. The lumen and wall is locally estimated on each axial slice using a patch centered on the previously found centerline.

![Illustration of the method](images/global_illustration.png)

## Installation

Depending on the way you want to execute XXX, you may have to install Python or Docker.
You don't need to install anything to use the [Grand Challenge platform](#use-grand-challenge-platform).


### Execute with Python

You will need a Python environment to run XXX. We advise you to use Miniconda. 
Miniconda allows you to install, run, and update Python packages and their dependencies. 
It can also create environments to isolate your libraries. 
To install Miniconda, open a new terminal and type the following commands:

If you are on Linux:
```
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda-installer.sh
bash /tmp/miniconda-installer.sh
```
If you are on Mac:
```
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o /tmp/miniconda-installer.sh
bash /tmp/miniconda-installer.sh
```

The method is distributed as a PyPi package and can be installed with the following commands:

```
conda create -n carotid-segmentation python=3.9
conda activate carotid-segmentation
pip install XXX
```

### Run a docker

You can also run XXX using its docker. To do so please make sure that [Docker](https://docs.docker.com/engine/install/)
is correctly installed on your machine.

## Application to your data set

XXX was trained to segment carotids from 3D black-blood MRI volumes. 
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

!!! warning
    The orientation of your volume is crucial for the algorithm.
    Please make sure that your tensor and affine allows to correctly orientate your image.

### Command line

### API

### Run docker

### Use Grand Challenge platform

## Test



### Source code

### Build docker
