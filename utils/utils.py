from typing import List, Dict, Optional
import os
import re
import glob
import json
import torch
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from albumentations.core.transforms_interface import BasicTransform, ImageOnlyTransform, NoOp
from monai.networks import one_hot
from monai.transforms.transform import Transform, MapTransform


class ToTensor(BasicTransform):
    """Convert image and mask to `torch.Tensor`. The numpy `HWC` image is converted to pytorch `CHW` tensor.
    If the image is in `HW` format (grayscale image), it will be converted to pytorch `HW` tensor.
    This is a simplified and improved version of the old `ToTensor`
    transform (`ToTensor` was deprecated, and now it is not present in Albumentations. You should use `ToTensorV2`
    instead).
    Args:
        transpose_mask (bool): If True and an input mask has three dimensions, this transform will transpose dimensions
            so the shape `[height, width, num_channels]` becomes `[num_channels, height, width]`. The latter format is a
            standard format for PyTorch Tensors. Default: False.
        always_apply (bool): Indicates whether this transformation should be always applied. Default: True.
        p (float): Probability of applying the transform. Default: 1.0.
    """

    def __init__(self, p=1):
        super(ToTensor, self).__init__(p)

    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask, "masks": self.apply_to_masks}

    def apply(self, img, **params):  # skipcq: PYL-W0613
        if len(img.shape) not in [3, 4]:
            raise ValueError("Albumentations only supports images in HW or HWC format")
        return torch.from_numpy(img)

    def apply_to_mask(self, mask, **params):  # skipcq: PYL-W0613
        return torch.from_numpy(mask)

    def apply_to_masks(self, masks, **params):
        return [self.apply_to_mask(mask, **params) for mask in masks]

    def get_transform_init_args_names(self):
        return ("transpose_mask",)

    def get_params_dependent_on_targets(self, params):
        return {}


class EnsureChannelFirst(ImageOnlyTransform):
    def __init__(self, p=1):
        super(EnsureChannelFirst, self).__init__(p)

    def apply(self, img, **params):
        return np.expand_dims(img, axis=0)


class SampleWiseNormalize(ImageOnlyTransform):
    def __init__(self, p=1):
        super(SampleWiseNormalize, self).__init__(p)

    def apply(self, img, **params):
        return (img - img.mean()) / img.std()


class ConvertToMultiChannel(NoOp):
    """
    Convert labels to multi channels based on classes:
    (0 = background, 1 = RA wall, 2 = LA wall, 3 = RA endo, 4 = LA endo, 5 = septum)
    to
    (0 = background, 1 = RA+LA wall, 2 = RA endo, 3 = LA endo)
    """
    def __init__(self, binary=False, p=1):
        super(ConvertToMultiChannel, self).__init__(p)

        self.binary = binary

    def apply_to_mask(self, mask, **params):
        result = []

        if self.binary:
            result.append(mask == 0)
            result.append(mask > 0)
        else:
            # remove label 5
            result.append(np.logical_or(mask == 0, mask == 5))
            # merge label 1 and label 2 to RA+LA wall
            result.append(np.logical_or(mask == 1, mask == 2))
            # label 3 is RA endo
            result.append(mask == 3)
            # label 4 is LA endo
            result.append(mask == 4)

        return np.stack(result, axis=0).astype(np.float32)


class ConvertToMultiChannelBasedOnClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    (0 = background, 1 = RA wall, 2 = LA wall, 3 = RA endo, 4 = LA endo, 5 = septum)
    to
    (0 = background, 1 = RA+LA wall, 2 = RA endo, 3 = LA endo)

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # background and label 5
            result.append(torch.logical_or(d[key] == 0, d[key] == 5))
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 1, d[key] == 2))
            # label 3 is RA endo
            result.append(d[key] == 3)
            # label 4 is LA endo
            result.append(d[key] == 4)
            d[key] = torch.stack(result, axis=0).float()
        return d


class AsDiscrete(Transform):
    def __init__(
        self,
        argmax: bool = False,
        to_onehot: Optional[int] = None,
        threshold: Optional[float] = None,
        **kwargs,
    ) -> None:
        self.argmax = argmax
        if isinstance(to_onehot, bool):  # for backward compatibility
            raise ValueError("`to_onehot=True/False` is deprecated, please use `to_onehot=num_classes` instead.")
        self.to_onehot = to_onehot
        self.threshold = threshold
        self.kwargs = kwargs

    def __call__(
        self,
        img,
        argmax: Optional[bool] = None,
        to_onehot: Optional[int] = None,
        threshold: Optional[float] = None,
    ):
        if argmax or self.argmax:
            img = torch.argmax(img, dim=self.kwargs.get("dim", 0), keepdim=self.kwargs.get("keepdim", True))

        to_onehot = self.to_onehot if to_onehot is None else to_onehot
        if to_onehot is not None:
            if not isinstance(to_onehot, int):
                raise ValueError("the number of classes for One-Hot must be an integer.")
            img = one_hot(
                img, num_classes=to_onehot, dim=self.kwargs.get("dim", 0), dtype=self.kwargs.get("dtype", torch.float)
            )

        threshold = self.threshold if threshold is None else threshold
        if threshold is not None:
            img = img >= threshold

        return img


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def load_nrrd(path):
    # this function loads .nrrd files into a 3D matrix and outputs it
    # the input is the specified file path
    # the output is the N x A x B for N slices of sized A x B
    # after rolling, the output is the A x B x N
    data = sitk.ReadImage(path)  # read in image
    # print(data.GetSpacing())
    data = sitk.Cast(sitk.RescaleIntensity(data),sitk.sitkUInt8)  # sacle to (0-255)
    data = sitk.GetArrayFromImage(data)  # convert to numpy array
    data = np.moveaxis(data, 0, -1)

    return data


def sort(l):
    """ Sort the given list in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    l.sort(key=alphanum_key)
    return l


def tif_to_nii(img_dir):
    """Convert multiple tiff images in a directory to a single 3D nifti file"""
    # get names for the file
    names = glob.glob(os.path.join(img_dir, '*.tif'))

    names = sort(names)

    assert len(names) > 0, f"No label in folder {img_dir}"

    image_stacks = []
    # # create an empty 3d numpy array with the desired shape: number of slices, dimensions of each slice
    # loop through all the files in the list and open them
    for filename in names:
        img = Image.open(filename)
        # add the slice to the 3D volume
        image_stacks.append(np.array(img))
    data = np.array(image_stacks)
    data = np.moveaxis(data, 0, -1)

    return data


def load_nii(path):
    # this function loads .nii files into a 3D matrix and outputs it
    # the input is the specified file path
    # the output is the N x A x B for N slices of sized A x B
    # after rolling, the output is the A x B x N
    data = sitk.ReadImage(path)  # read in image
    data = sitk.GetArrayFromImage(data)  # convert to numpy array
    return data.astype(np.float32)


def _compute_path(base_dir, element, check_path=False):
    """
    Args:
        base_dir: the base directory of the dataset.
        element: file path(s) to append to directory.
        check_path: if `True`, only compute when the result is an existing path.

    Raises:
        TypeError: When ``element`` contains a non ``str``.
        TypeError: When ``element`` type is not in ``Union[list, str]``.

    """

    def _join_path(base_dir: str, item: str):
        result = os.path.normpath(os.path.join(base_dir, item))
        if check_path and not os.path.exists(result):
            # if not an existing path, don't join with base dir
            return f"{item}"
        return f"{result}"

    if isinstance(element, (str, os.PathLike)):
        return _join_path(base_dir, element)
    if isinstance(element, list):
        for e in element:
            if not isinstance(e, (str, os.PathLike)):
                return element
        return [_join_path(base_dir, e) for e in element]
    return element


def _append_paths(base_dir: str, is_segmentation: bool, items: List[Dict]) -> List[Dict]:
    """
    Args:
        base_dir: the base directory of the dataset.
        is_segmentation: whether the datalist is for segmentation task.
        items: list of data items, each of which is a dict keyed by element names.

    Raises:
        TypeError: When ``items`` contains a non ``dict``.

    """
    for item in items:
        if not isinstance(item, dict):
            raise TypeError(f"Every item in items must be a dict but got {type(item).__name__}.")
        for k, v in item.items():
            if k == "image" or is_segmentation and k == "label":
                item[k] = _compute_path(base_dir, v, check_path=False)
            else:
                item[k] = v
    return items


def load_datalist(
    data_list_file_path: str,
    is_segmentation: bool = True,
    data_list_key: str = "training",
    base_dir: Optional[str] = None,
) -> List[Dict]:
    """Load image/label paths of decathlon challenge from JSON file

    Json file is similar to what you get from http://medicaldecathlon.com/
    Those dataset.json files

    Args:
        data_list_file_path: the path to the json file of datalist.
        is_segmentation: whether the datalist is for segmentation task, default is True.
        data_list_key: the key to get a list of dictionary to be used, default is "training".
        base_dir: the base directory of the dataset, if None, use the datalist directory.

    Raises:
        ValueError: When ``data_list_file_path`` does not point to a file.
        ValueError: When ``data_list_key`` is not specified in the data list file.

    Returns a list of data items, each of which is a dict keyed by element names, for example:

    .. code-block::

        [
            {'image': '/workspace/data/chest_19.nii.gz',  'label': 0},
            {'image': '/workspace/data/chest_31.nii.gz',  'label': 1}
        ]

    """
    data_list_file_path = Path(data_list_file_path)
    if not data_list_file_path.is_file():
        raise ValueError(f"Data list file {data_list_file_path} does not exist.")
    with open(data_list_file_path) as json_file:
        json_data = json.load(json_file)
    if data_list_key not in json_data:
        raise ValueError(f'Data list {data_list_key} not specified in "{data_list_file_path}".')
    expected_data = json_data[data_list_key]
    if data_list_key == "test" and not isinstance(expected_data[0], dict):
        # decathlon datalist may save the test images in a list directly instead of dict
        expected_data = [{"image": i} for i in expected_data]

    if base_dir is None:
        base_dir = data_list_file_path.parent

    return _append_paths(base_dir, is_segmentation, expected_data)


def save_weight_update_plot(model_params, ud, save_path):
    # Plot to visualize weight update magnitude
    plt.figure(figsize=(20, 4))
    legends = []
    for i, p in enumerate(model_params):
        if p.ndim == 4:
            plt.plot([ud[j][i] for j in range(len(ud))])
            legends.append('param %d' % i)
    plt.plot([0, len(ud)], [-3, -3], 'k')  # these ratios should be ~1e-3, indicate on plot
    plt.legend(legends)
    plt.savefig(save_path)

    return plt


def get_image_files(data_dir, patient_list, has_section=False):
    image_files = []

    for patient in patient_list:
        for root, subdirs, files in os.walk(os.path.join(data_dir, patient)):
            if len(files) > 0:
                image_path = root.replace(data_dir, "")
                print(image_path)
                if image_path.startswith('/'):
                    image_path = image_path[1:]

                image = files[0] if 'lgemri' in files[0] else files[1]
                label = files[0] if 'label' in files[0] else files[1]
                data = {"image": os.path.join(image_path, image),
                        "label": os.path.join(image_path, label),
                        "patient": patient}

                if has_section:
                    data["session"] = root.split('/')[-1]

                image_files.append(data)

    return image_files


def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data in tqdm(loader):
        channels_sum += torch.mean(data["image"])
        channels_sqrd_sum += torch.mean(data["image"] ** 2)
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def distributed_all_gather(
    tensor_list, valid_batch_size=None, out_numpy=False, world_size=None, no_barrier=False, is_valid=None
):

    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        torch.distributed.barrier()
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params (MB): {trainable_params/1e6} || all params (MB): {all_param/1e6} || trainable%: {100 * trainable_params / all_param}"
    )


def freeze_model_weights(model, freeze_keys):
    """
    freeze the model weights based on the layer names
    """
    print('Going to apply weight frozen')
    # print('before frozen, require grad parameter names:')
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    print('freeze_keys', freeze_keys)
    for name, para in model.named_parameters():
        if para.requires_grad and any(key in name for key in freeze_keys):
            para.requires_grad = False

    # print('after frozen, require grad parameter names:')
    # for name, para in model.named_parameters():
    #     if para.requires_grad:
    #         print(name)

    return model


def resample_sitk_image(sitk_image, spacing=None, interpolator=None, fill_value=0):
    # https://github.com/SimpleITK/SlicerSimpleFilters/blob/master/SimpleFilters/SimpleFilters.py
    _SITK_INTERPOLATOR_DICT = {
        'nearest': sitk.sitkNearestNeighbor,
        'linear': sitk.sitkLinear,
        'gaussian': sitk.sitkGaussian,
        'label_gaussian': sitk.sitkLabelGaussian,
        'bspline': sitk.sitkBSpline,
        'hamming_sinc': sitk.sitkHammingWindowedSinc,
        'cosine_windowed_sinc': sitk.sitkCosineWindowedSinc,
        'welch_windowed_sinc': sitk.sitkWelchWindowedSinc,
        'lanczos_windowed_sinc': sitk.sitkLanczosWindowedSinc
    }
    """Resamples an ITK image to a new grid. If no spacing is given,
    the resampling is done isotropically to the smallest value in the current
    spacing. This is usually the in-plane resolution. If not given, the
    interpolation is derived from the input data type. Binary input
    (e.g., masks) are resampled with nearest neighbors, otherwise linear
    interpolation is chosen.
    Parameters
    ----------
    sitk_image : SimpleITK image or str
      Either a SimpleITK image or a path to a SimpleITK readable file.
    spacing : tuple
      Tuple of integers
    interpolator : str
      Either `nearest`, `linear` or None.
    fill_value : int
    Returns
    -------
    SimpleITK image.
    """

    if isinstance(sitk_image, str):
        sitk_image = sitk.ReadImage(sitk_image)
    num_dim = sitk_image.GetDimension()

    if not interpolator:
        interpolator = 'linear'
        pixelid = sitk_image.GetPixelIDValue()

        if pixelid not in [1, 2, 4]:
            raise NotImplementedError(
                'Set `interpolator` manually, '
                'can only infer for 8-bit unsigned or 16, 32-bit signed integers')
        if pixelid == 1:  # 8-bit unsigned int
            interpolator = 'nearest'

    orig_origin = sitk_image.GetOrigin()
    orig_spacing = np.array(sitk_image.GetSpacing())
    orig_size = np.array(sitk_image.GetSize())

    if not spacing:
        min_spacing = orig_spacing.min()
        new_spacing = [min_spacing] * num_dim
    else:
        new_spacing = [float(s) for s in spacing]

    assert interpolator in _SITK_INTERPOLATOR_DICT.keys(), \
        '`interpolator` should be one of {}'.format(_SITK_INTERPOLATOR_DICT.keys())

    sitk_interpolator = _SITK_INTERPOLATOR_DICT[interpolator]

    # print(orig_size, orig_spacing, new_spacing)

    # calculate output size in voxels
    new_size = [
        int(np.round(
            size * (spacing_in / spacing_out)
        ))
        for size, spacing_in, spacing_out in zip(orig_size, orig_spacing, new_spacing)
    ]

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetSize(new_size)
    resample_filter.SetTransform(sitk.Transform())
    resample_filter.SetInterpolator(sitk_interpolator)
    resample_filter.SetOutputOrigin(orig_origin)
    resample_filter.SetOutputSpacing(new_spacing)
    resample_filter.SetOutputDirection(sitk_image.GetDirection())
    resample_filter.SetOutputDirection(sitk_image.GetDirection())
    resample_filter.SetDefaultPixelValue(fill_value)
    resample_filter.SetOutputPixelType(sitk_image.GetPixelIDValue())

    resampled_sitk_image = resample_filter.Execute(sitk_image)
    return resampled_sitk_image