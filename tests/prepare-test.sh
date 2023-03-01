if [ -d "tests/transform/tmp" ]; then
    rm -r "tests/transform/tmp"
fi

if [ ! -d "tests/raw_dir" ]; then
    curl https://surfdrive.surf.nl/files/index.php/s/e13O5s7PPTsJNli/download -o test.tar -L
    unzip test.tar
    rm test.tar

    # Copy raw data
    cp -r v1/raw_dir tests
    # Copy transforms data
    for transform in centerline contour heatmap pipeline polar segmentation
    do
      if [ -d "tests/transform/$transform/reference" ]; then
          rm -r tests/transform/$transform/reference
      fi

      if [ -d "tests/transform/$transform/input" ]; then
          rm -r tests/transform/$transform/input
      fi
      cp -r v1/transform/$transform/* tests/transform/$transform
    done

    # rm -r v1
else
    echo "Data was already downloaded. To force a new download remove tests/raw_dir"
fi

MODELPATH="tests/models"

if [ ! -d $MODELPATH ]; then
    mkdir $MODELPATH
    mkdir $MODELPATH/contour_transform
    cp models/contour_transform/model_0* $MODELPATH/contour_transform
    cp models/contour_transform/model_1* $MODELPATH/contour_transform
    mkdir $MODELPATH/heatmap_transform
    cp models/heatmap_transform/model_0* $MODELPATH/heatmap_transform
    cp -r models/contour_transform_dropout $MODELPATH
fi