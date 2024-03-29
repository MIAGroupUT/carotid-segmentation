if [ -d "tests/transform/tmp" ]; then
    rm -r "tests/transform/tmp"
fi

if [ ! -d "tests/raw_dir" ]; then
    curl https://surfdrive.surf.nl/files/index.php/s/CoazEglbyGXS23G/download -o test.tar -L
    unzip test.tar
    rm test.tar

    # Copy raw data
    cp -r v4/raw_dir tests
    # Copy transforms data
    for transform in centerline contour heatmap pipeline polar segmentation
    do
      if [ -d "tests/transform/$transform/reference" ]; then
          rm -r tests/transform/$transform/reference
      fi

      if [ -d "tests/transform/$transform/input" ]; then
          rm -r tests/transform/$transform/input
      fi
      cp -r v4/transform/$transform/* tests/transform/$transform
    done

    # Copy comparison data
    for transform in centerline contour
    do
      if [ -d "tests/compare/$transform/reference.tsv" ]; then
          rm tests/compare/$transform/reference.tsv
      fi

      if [ -d "tests/compare/$transform/input" ]; then
          rm -r tests/compare/$transform/input
      fi
      cp -r v4/compare/$transform/* tests/compare/$transform
    done

    # Copy comparison data
    for transform in contour
    do
      if [ -d "tests/train/$transform/input" ]; then
          rm -r tests/train/$transform/input
      fi
      cp -r v4/train/$transform/* tests/train/$transform
    done

    # Copy convert data
    for transform in miccai2020 miccai2022
    do
      if [ -d "tests/convert/$transform/reference" ]; then
          rm -r tests/convert/$transform/reference
      fi

      if [ -d "tests/convert/$transform/input" ]; then
          rm -r tests/convert/$transform/input
      fi
      cp -r v4/convert/$transform/* tests/convert/$transform
    done

    rm -r v4
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
