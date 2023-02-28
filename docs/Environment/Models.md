# Deep learning models

## U-Nets for heatmap estimation

## CNNs for lumen and wall distances regression

## Getting pre-trained models

- On Mac / Linux:

You can execute the following command at the root of the directory to fetch all the models:
```
make get-models
```

- On Windows:

Models can be downloaded with [this link](https://surfdrive.surf.nl/files/index.php/s/DanUvHpx6BXM7dY/download)
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