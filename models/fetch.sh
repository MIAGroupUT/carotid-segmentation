cd models/

if [ ! -d "contour_transform" ]; then
    # Retrieve contour models
    curl https://surfdrive.surf.nl/files/index.php/s/evopHTobixCuf3t/download -o contour_transform.tar
    unzip contour_transform.tar
    rm contour_transform.tar
else
    echo "models for contour_transform already found."
fi

if [ ! -d "heatmap_transform" ]; then
    # Retrieve heatmap models
    curl https://surfdrive.surf.nl/files/index.php/s/1wG3WbCuEy34NsU/download -o heatmap_transform.tar
    unzip heatmap_transform.tar
    rm heatmap_transform.tar
else
    echo "models for heatmap_transform already found."
fi

if [ ! -d "contour_transform_dropout" ]; then
    # Retrieve heatmap models
    curl https://surfdrive.surf.nl/files/index.php/s/P9uMnfKWSbenfDx/download -o contour_transform_dropout.tar
    unzip contour_transform_dropout.tar
    rm contour_transform_dropout.tar
else
    echo "models for contour_transform_dropout already found."
fi