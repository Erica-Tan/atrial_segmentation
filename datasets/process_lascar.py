import glob
import os
import argparse
import numpy as np
import SimpleITK as sitk
from skimage import exposure

from utils.utils import resample_sitk_image

parser = argparse.ArgumentParser(description="Process LAScar dataset")
parser.add_argument("--data_dir", default="./data/dataset0/", type=str, help="dataset directory")
parser.add_argument("--output_dir", default="./data/dataset0_processed/", type=str, help="output dataset directory")

resolution = (0.625, 0.625, 2.5)


def load_nii(path, is_label=False):
    # this function loads .nrrd files into a 3D matrix and outputs it
    # the input is the specified file path
    # the output is the N x A x B for N slices of sized A x B
    # after rolling, the output is the A x B x N
    data = sitk.ReadImage(path)  # read in image
    print(f"Original size: {data.GetSize()}, {data.GetSpacing()}")

    if is_label:
        sampled_data = resample_sitk_image(data, spacing=resolution, interpolator='nearest')
    else:
        sampled_data = resample_sitk_image(data, spacing=resolution, interpolator='linear')

    print(f"Resample size: {sampled_data.GetSize()}, {sampled_data.GetSpacing()}", np.array_equal(sitk.GetArrayFromImage(data), sitk.GetArrayFromImage(sampled_data)))

    if not is_label:
        sampled_data = sitk.Cast(sitk.RescaleIntensity(sampled_data), sitk.sitkUInt8)  # sacle to (0-255)
    sampled_data = sitk.GetArrayFromImage(sampled_data)  # convert to numpy array

    return sampled_data


def main():
    args = parser.parse_args()

    # folder checking
    assert os.path.isdir(args.data_dir), f"Data directory not exist {args.data_dir}"

    os.makedirs(args.output_dir, exist_ok=True)

    patients = os.listdir(args.data_dir)
    for i, patient in enumerate(patients):
        images = glob.glob(os.path.join(args.data_dir, patient, "*enhanced.nii.gz"))
        labels = glob.glob(os.path.join(args.data_dir, patient, "*atriumSegImgMO.nii.gz"))

        for image_path, label_path in zip(images, labels):
            save_dir = os.path.join(args.output_dir, patient)
            os.makedirs(save_dir, exist_ok=True)
            print(save_dir, image_path, label_path)

            # load image file
            image = load_nii(image_path)
            # process image file
            processed_image = exposure.equalize_adapthist(image, clip_limit=0.03)
            print(processed_image.shape, processed_image.max(), processed_image.dtype)
            # create nifti file
            ni_img = sitk.GetImageFromArray(processed_image)
            ni_img.SetSpacing(resolution)
            print(f"Save size: {ni_img.GetSize()}, {ni_img.GetSpacing()}")
            sitk.WriteImage(ni_img, os.path.join(save_dir, 'lgemri_equalized.nii.gz'))
            # sitk.WriteImage(ni_img, os.path.join(args.output_dir, f'LASCAR_{patient}_000{i}.nii.gz'))

            # load label
            label = load_nii(label_path, is_label=True)
            label = label//255
            label = label.astype(np.uint8)
            print(label.shape, label.min(), label.max(), label.dtype)
            # create nifti file
            ni_label = sitk.GetImageFromArray(label)
            ni_label.SetSpacing(resolution)
            print(f"Save size: {ni_label.GetSize()}, {ni_label.GetSpacing()}")
            sitk.WriteImage(ni_label, os.path.join(save_dir, 'label.nii.gz'))

            del image, processed_image, ni_img, ni_label


if __name__ == "__main__":
    main()


