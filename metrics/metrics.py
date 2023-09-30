from typing import Union, Optional
import torch
import numpy as np
from monai.metrics import get_mask_edges, compute_percent_hausdorff_distance, get_surface_distance


def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


def hausdorff_distance(
    y_pred: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    distance_metric: str = "euclidean",
    percentile: Optional[float] = 95,
    directed: bool = False,
    spacing = None
):
    """
    Compute the Hausdorff distance.

    Args:
        y_pred: input data to compute, typical segmentation model output.
            It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
            should be binarized.
        y: ground truth to compute mean the distance. It must be one-hot format and first dim is batch.
            The values should be binarized.
        distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
            the metric used to compute surface distance. Defaults to ``"euclidean"``.
        percentile: an optional float number between 0 and 100. If specified, the corresponding
            percentile of the Hausdorff Distance rather than the maximum result will be achieved.
            Defaults to ``None``.
        directed: whether to calculate directed Hausdorff distance. Defaults to ``False``.
    """
    if y.shape != y_pred.shape:
        raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")

    (edges_pred, edges_gt) = get_mask_edges(y_pred, y)

    distance_1 = compute_percent_hausdorff_distance(edges_pred, edges_gt, distance_metric, percentile, spacing)
    if directed:
        return distance_1
    else:
        distance_2 = compute_percent_hausdorff_distance(edges_gt, edges_pred, distance_metric, percentile, spacing)
        return max(distance_1, distance_2)


def average_surface_distance(
    y_pred: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    symmetric: bool = False,
    distance_metric: str = "euclidean",
    spacing = None,
) -> torch.Tensor:
    """
    This function is used to compute the Average Surface Distance from `y_pred` to `y`
    under the default setting.
    In addition, if sets ``symmetric = True``, the average symmetric surface distance between
    these two inputs will be returned.
    The implementation refers to `DeepMind's implementation <https://github.com/deepmind/surface-distance>`_.

    Args:
        y_pred: input data to compute, typical segmentation model output.
            It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
            should be binarized.
        y: ground truth to compute mean the distance. It must be one-hot format and first dim is batch.
            The values should be binarized.
        include_background: whether to include distance computation on the first channel of
            the predicted output. Defaults to ``False``.
        symmetric: whether to calculate the symmetric average surface distance between
            `seg_pred` and `seg_gt`. Defaults to ``False``.
        distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
            the metric used to compute surface distance. Defaults to ``"euclidean"``.
        spacing: spacing of pixel (or voxel). This parameter is relevant only if ``distance_metric`` is set to ``"euclidean"``.
            If a single number, isotropic spacing with that value is used for all images in the batch. If a sequence of numbers,
            the length of the sequence must be equal to the image dimensions. This spacing will be used for all images in the batch.
            If a sequence of sequences, the length of the outer sequence must be equal to the batch size.
            If inner sequence has length 1, isotropic spacing with that value is used for all images in the batch,
            else the inner sequence length must be equal to the image dimensions. If ``None``, spacing of unity is used
            for all images in batch. Defaults to ``None``.
    """

    if y.shape != y_pred.shape:
        raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")

    (edges_pred, edges_gt) = get_mask_edges(y_pred, y)

    surface_distance = get_surface_distance(
        edges_pred, edges_gt, distance_metric=distance_metric, spacing=spacing
    )
    if symmetric:
        surface_distance_2 = get_surface_distance(
            edges_gt, edges_pred, distance_metric=distance_metric, spacing=spacing
        )
        surface_distance = np.concatenate([surface_distance, surface_distance_2])

    return np.nan if surface_distance.shape == (0,) else surface_distance.mean()
