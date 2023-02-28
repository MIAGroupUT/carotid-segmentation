# Input data

The models provided by this package were trained to segment carotids from 3D black-blood MRI volumes.

!!! note "Original training data"
    Training data is distributed by the Grand Challenge platform.
    You can request access via this [link](https://vessel-wall-segmentation.grand-challenge.org/).

The raw data can be provided as DICOM, MHD or MHA files. Structure of the raw directory for DICOM files:

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

!!! warning "Orientation"
    The orientation of your volume is crucial for the algorithm.
    Please make sure that your tensor and affine allows to correctly orientate your image.
    The algorithm also assumed that your image has an isotropic resolution in an axial slice.

!!! note "Identification of patients"
    Participants will be associated with a `participant_id`. For DICOM files it will correspond
    to the names of the directories, and for the MHD/MHA files to the filename without the extension.
