from typing import List, Union, Tuple, Callable
import numpy as np
from acvl_utils.morphology.morphology_helper import remove_all_but_largest_component
from skimage.measure import label


def remove_all_but_largest_component_from_segmentation(segmentation: np.ndarray,
                                                       labels_or_regions: Union[int, Tuple[int, ...],
                                                                                List[Union[int, Tuple[int, ...]]]],
                                                       background_label: int = 0) -> np.ndarray:
    mask = np.zeros_like(segmentation, dtype=bool)
    if not isinstance(labels_or_regions, list):
        labels_or_regions = [labels_or_regions]
    for l_or_r in labels_or_regions:
        mask |= region_or_label_to_mask(segmentation, l_or_r)  # or operation

    mask_keep = remove_all_but_largest_component(mask)

    ret = np.copy(segmentation)  # do not modify the input!
    ret[mask & ~mask_keep] = background_label
    return ret


def region_or_label_to_mask(segmentation: np.ndarray, region_or_label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros_like(segmentation, dtype=bool)
        for r in region_or_label:
            mask[segmentation == r] = True
    return mask


def remove_all_but_largest_component(binary_image: np.ndarray, connectivity: int = None) -> np.ndarray:
    """
    Removes all but the largest component in binary_image. Replaces pixels that don't belong to it with background_label
    """
    filter_fn = lambda x, y: [i for i, j in zip(x, y) if j == max(y)]
    return generic_filter_components(binary_image, filter_fn, connectivity)


def generic_filter_components(binary_image: np.ndarray, filter_fn: Callable[[List[int], List[int]], List[int]],
                              connectivity: int = None):
    """
    filter_fn MUST return the component ids that should be KEPT!
    filter_fn will be called as: filter_fn(component_ids, component_sizes) and is expected to return a List of int

    returns a binary array that is True where the filtered components are
    """
    labeled_image, component_sizes = label_with_component_sizes(binary_image, connectivity)
    component_ids = list(component_sizes.keys())
    component_sizes = list(component_sizes.values())
    # print(component_ids, component_sizes)
    keep = filter_fn(component_ids, component_sizes)
    # print(keep)
    return np.in1d(labeled_image.ravel(), keep).reshape(labeled_image.shape)


def label_with_component_sizes(binary_image: np.ndarray, connectivity: int = None) -> Tuple[np.ndarray, dict]:
    if not binary_image.dtype == bool:
        print('Warning: it would be way faster if your binary image had dtype bool')
    labeled_image, num_components = label(binary_image, return_num=True, connectivity=connectivity)
    component_sizes = {i + 1: j for i, j in enumerate(np.bincount(labeled_image.ravel())[1:])}
    return labeled_image, component_sizes


def apply_postprocessing(segmentation: np.ndarray, pp_fn_kwargs: List[dict]):
    for kwargs in pp_fn_kwargs:
        segmentation = remove_all_but_largest_component_from_segmentation(segmentation, **kwargs)
    return segmentation


def get_center_crop_coords(height: int, width: int, crop_height: int, crop_width: int):
    y1 = (height - crop_height) // 2
    y2 = y1 + crop_height
    x1 = (width - crop_width) // 2
    x2 = x1 + crop_width
    return x1, y1, x2, y2

