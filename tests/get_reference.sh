curl https://surfdrive.surf.nl/files/index.php/s/e13O5s7PPTsJNli/download -o test.tar -L
tar -xvf test.tar
rm test.tar

# Copy raw data
cp -r v1/raw_dir tests
# Copy transforms data
for transform in centerline_transform contour_transform heatmap_transform pipeline_transform polar_transform segmentation_transform
do
  echo $transform
  cp -r v1/$transform/* tests/$transform
done

rm -r v1

MODELPATH="tests/models"

if [ ! -d $MODELPATH ]; then
    mkdir $MODELPATH
    mkdir $MODELPATH/contour_transform
    cp models/contour_transform/model_0* $MODELPATH/contour_transform
    mkdir $MODELPATH/heatmap_transform
    cp models/heatmap_transform/model_0* $MODELPATH/heatmap_transform
    cp -r models/contour_transform_dropout $MODELPATH