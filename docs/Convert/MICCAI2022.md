# Convert MICCAI 2022 grand challenge data set

This converter outputs two folders:

- a folder containing cropped raw data (MHA files),
- a folder containing the contour annotations.

2 vessels can be annotated, the left and right internal carotids.
However, they may not be annotated for all participants (only one of the side
might be present).
Note that the common carotid is labelled as internal.

## Prerequisites

You can download the data on the [grand challenge website](https://vessel-wall-segmentation-2022.grand-challenge.org/data/). 

## Running the task

The task can be run with the following command line:
```
carotid convert miccai2022 ORIGINAL_DIR RAW_DIR ANNOTATION_DIR
```
where:

- `ORIGINAL_DIR` (str) is the data as downloaded from the website (after decompression).
- `RAW_DIR` (str) is the folder containing formatted raw data.
- `ANNOTATION_DIR` (str) is the folder containing formatted contour annotations.

## Outputs

`RAW_DIR` contains one MHA file per participant. The images are cropped to remove anterior and posterior
background. The JSON file `parameters.json` gives the parameters used to rescale the intensities in this data set.

`ANNOTATION_DIR` contains one folder per participant, formatted like the output
of [carotid transform contour](../Transforms/Contour.md#outputs).