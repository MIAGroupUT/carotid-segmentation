# Convert MICCAI 2020 grand challenge data set

This converter outputs two folders:

- a folder containing the original raw data (DICOM files),
- a folder containing the contour annotations.

4 vessels can be annotated, the left and right internal and external carotids.
However, they may not be annotated for all participants (most of the time 
one of the two external carotids is missing).
Note that the common carotid is labelled as internal.

## Prerequisites
You can download the data on the [grand challenge website](https://vessel-wall-segmentation.grand-challenge.org/Data). 
Detailed explanations on the acquisition protocols are given on [Zenodo](https://zenodo.org/record/4575301).

## Running the task

The task can be run with the following command line:
```
carotid convert miccai2020 ORIGINAL_DIR RAW_DIR ANNOTATION_DIR
```
where:

- `ORIGINAL_DIR` (str) is the data as downloaded from the website (after decompression).
- `RAW_DIR` (str) is the folder containing formatted raw data.
- `ANNOTATION_DIR` (str) is the folder containing formatted contour annotations.

## Outputs

`RAW_DIR` contains one folder per participant in which the DICOM files of
the raw MR are copied. The JSON file `parameters.json` gives the parameters
used to rescale the intensities in this data set.

`ANNOTATION_DIR` contains one folder per participant, formatted like the output
of [carotid transform contour](../Transforms/Contour.md#outputs).