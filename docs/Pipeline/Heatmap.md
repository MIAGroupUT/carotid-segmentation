# `heatmap_transform` - Compute heatmaps with pre-trained U-Nets

The first step of the pipeline consists in estimating the position of the centerlines of each carotid.
Each side of the image (left or right) is treated independently. One heatmap is extracted for each side,
and have two channels corresponding to each of the carotid (internal or external).

## Prerequisites

Make sure that your data set is correctly structured (see the [data section](../Data.md) for more information).

You also need to provide one (or several models) stored in the same directory.

## Running the task

The task can be run with the following command line:
```
carotid heatmap_transform RAW_DIR HEATMAP_MODEL_DIR OUTPUT_DIR
```
where:

- `RAW_DIR` (str) is the path to the directory containing the raw data.
- `HEATMAP_MODEL_DIR` (str) is the path to the directory containing the weights of the pre-trained U-Nets.
- `OUTPUT_DIR` (str) is the path to the directory containing the outputs.

Options:

- `--config_path` (str) is the path to a config file defining the values of the parameters.
For more information on the config file, refer to [this section](../Configuration.md).
- `--participant` (List[str]) restricts the application of the transform to this list of participant IDs. 
Default will perform the pipeline on all participants with a raw image.
- `--device` (`cuda`|`cpu`) is the device that will be used to perform the forward pass of the U-Net.
Default will try to find `cuda`, and use `cpu` if it is not available.

## Parameters

Two values can be changed in the config file:

- `roi_size` (Tuple[int, int, int]) is the size of the window used to infer the whole heatmap. The order of the coordinates is (X, Y, Z). 
Default: `(64, 64, 160)`. 
- `spacing` (bool) indicates if the image should be resamples to the resolution used by the network [0.5, 0.5, 0.5]. If the resolution of your image
is close to this resolution, you can choose to disable resampling to gain computational time and avoid image distortion. Default: `True`.


## Outputs

Output structure for participant `participant_id`:
```console
<output_dir>
├── heatmap_parameters.json
└── <participant_id>
        ├── left_heatmap.mha
        └── right_heatmap.mha
```

where:

- `heatmap_parameters.json` is a JSON file summarizing the parameters used to perform the transform
- `<side>_heatmap.mha` is a volume with the same spatial size than the corresponding raw input and two channels.
The first channel corresponds to the heatmap for the internal carotid, the second one corresponds to the external carotid.