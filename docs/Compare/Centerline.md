# `centerline` - Evaluate the distances between centers in each axial slices

This pipeline compares two centerlines by computing the euclidean distance of centers
found in each axial slice.

## Prerequisites

This step relies on the outputs of `transform centerline`.

## Running the task

The task can be run with the following command line:
```
carotid compare centerline TRANSFORM1_DIR TRANSFORM2_DIR OUTPUT_PATH
```
where:

- `TRANSFORM1_DIR` (str) is the path to the first directory containing the centerlines.
- `TRANSFORM2_DIR` (str) is the path to the second directory containing the centerlines.
- `OUTPUT_PATH` (str) is the path to computed TSV file. If this path is a directory, it will create
the file `compare_centerline.tsv` there.


## Outputs

The TSV file
participant_id", "side", "label", "z", "euclidean_distance

| participant_id | side  | label    | z   | euclidean distance |
|----------------|-------|----------|-----|--------------------|
| Anon450        | left  | internal | 340 | 3.5                | 
| Anon450        | left  | internal | 341 | 3.4                | 
| ...            | ...   | ...      | ... | ...                | 
| Anon062        | right | external | 413 | 0.0                | 