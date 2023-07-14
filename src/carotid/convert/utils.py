import numpy as np
from typing import List, Union
from xml.etree.ElementTree import Element


def find_annotated_slices(qvs_root: Element) -> List[int]:
    """
    As all slices are not annotated, this function allows to know which slices were annotated.

    Args:
        qvs_root: reader of QVS file containing annotations.

    Returns:
        list of indices corresponding to annotated slices
    """
    avail_slices = []
    qvasimg = qvs_root.findall("QVAS_Image")
    for i in range(len(qvasimg)):
        contours = qvasimg[i - 1].findall("QVAS_Contour")
        if len(contours):  # Contours were found for slice i
            avail_slices.append(i)
    return avail_slices


def get_contour(
    qvs_root: Element,
    slice_idx: int,
    contour_type: str,
    image_size: int = 720,
    check_integrity: bool = True,
) -> Union[None, np.ndarray]:
    """
    Computes the list of the cartesian coordinates of a contour corresponding to a particular slice.

    Args:
        qvs_root: reader of QVS file containing annotations.
        slice_idx: index corresponding to the slice whose contour is extracted.
        contour_type: type of contour. Must be chosen in ["Lumen", "Outer Wall"].
        image_size: last dimension of the image. Used to rescale the coordinates.
        check_integrity: check if slice number corresponds to contour in QVS file.

    Returns:
        Array of size (N, 2) corresponding to the list of N coordinates.
    """

    possible_types = ["Lumen", "Outer Wall"]

    if contour_type not in possible_types:
        raise ValueError(
            f"Type should be in {possible_types}.\n" f"Current value is {contour_type}."
        )

    qvasimg = qvs_root.findall("QVAS_Image")

    # Check that slice_index corresponds to slice_index - 1 in QVS
    if check_integrity:
        assert int(qvasimg[slice_idx - 1].get("ImageName").split("I")[-1]) == slice_idx

    qvascontour_list = qvasimg[slice_idx - 1].findall("QVAS_Contour")
    for qvascontour in qvascontour_list:
        if qvascontour.find("ContourType").text == contour_type:
            point_list = qvascontour.find("Contour_Point").findall("Point")
            contours = []
            for point in point_list:
                # Annotations were rescaled to 512 x 512 by challenge organizers
                contx = float(point.get("x")) / 512 * image_size
                conty = float(point.get("y")) / 512 * image_size
                # if current point is different from last point, add to contours
                if (
                    len(contours) == 0
                    or contours[-1][0] != contx
                    or contours[-1][1] != conty
                ):
                    contours.append([contx, conty])

            return np.array(contours)
