cd models/

if [ ! -d "contour_transform" ]; then
    # Retrieve contour models
    curl https://surfdrive.surf.nl/files/index.php/s/evopHTobixCuf3t/download -o contour_transform.tar
    tar -xvf contour_transform.tar
    rm contour_transform.tar
else
    echo "models for contour_transform already found."
fi

if [ ! -d "heatmap_transform" ]; then
    # Retrieve heatmap models
    curl https://surfdrive.surf.nl/files/index.php/s/1wG3WbCuEy34NsU/download -o heatmap_transform.tar
    tar -xvf heatmap_transform.tar
    rm heatmap_transform.tar
else
    echo "models for heatmap_transform already found."
fi