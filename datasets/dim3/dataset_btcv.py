import glob
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from torch.utils.data import Dataset
from monai import transforms
from monai.data import DataLoader

from utils.utils import resample_sitk_image


class BTCVDataset(Dataset):
    def __init__(self, datalist, transform=None, cache=False) -> None:
        super().__init__()
        self.transform = transform
        self.datalist = datalist
        self.cache = cache
        if cache:
            self.cache_data = []
            for i in tqdm(range(len(datalist)), total=len(datalist)):
                d = self.read_data(datalist[i])
                self.cache_data.append(d)

    def read_data(self, data_path):
        image_path = data_path[0]
        label_path = data_path[1]

        image_data = sitk.GetArrayFromImage(resample_sitk_image(sitk.ReadImage(image_path), spacing=[1.5, 1.5, 2.0], interpolator='bspline'))
        # print(sitk.ReadImage(image_path).GetSpacing(), sitk.ReadImage(image_path).GetSize())

        raw_label_data = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        raw_label_data = np.expand_dims(raw_label_data, axis=0).astype(np.float32)

        seg_data = sitk.GetArrayFromImage(
            resample_sitk_image(sitk.ReadImage(label_path), spacing=[1.5, 1.5, 2.0], interpolator='nearest'))

        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        seg_data = np.expand_dims(seg_data, axis=0).astype(np.float32)

        return {
            "image": image_data,
            "label": seg_data,
            "raw_label": raw_label_data
        }

    def __getitem__(self, i):
        if self.cache:
            image = self.cache_data[i]
        else:
            try:
                image = self.read_data(self.datalist[i])
            except:
                if i != len(self.datalist) - 1:
                    return self.__getitem__(i + 1)
                else:
                    return self.__getitem__(i - 1)
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.datalist)


if __name__ == "__main__":
    data_dir = '/media/erica/Data/Projects/medical_imaging/datasets/BTCV/Training/'

    all_images = sorted(glob.glob(f"{data_dir}imagesTr/*.nii.gz"))
    all_labels = sorted(glob.glob(f"{data_dir}labelsTr/*.nii.gz"))

    all_paths = [[all_images[i], all_labels[i]] for i in range(len(all_images))]

    train_files = []
    val_files = []
    for p in all_paths:
        if "0008" in p[0] or "0022" in p[0] or \
                "0038" in p[0] or "0036" in p[0] or \
                "0032" in p[0] or "0002" in p[0] or \
                "0029" in p[0] or \
                "0003" in p[0] or \
                "0001" in p[0] or "0004" in p[0] or \
                "0025" in p[0] or "0035" in p[0]:

            val_files.append(p)
        else:
            train_files.append(p)

    train_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),

            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),

            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
            transforms.AsDiscreted(keys="label", to_onehot=14),
            transforms.ToTensord(keys=["image", "label"], ),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.AsDiscreted(keys="label", to_onehot=14),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    print(train_files)

    for image_path, label_path in train_files:
        print(image_path)

        data = sitk.ReadImage(image_path)  # read in image
        print(f"Original size: {data.GetSize()}, {data.GetSpacing()}")

    # train_ds = BTCVDataset(train_files[:5], transform=train_transform, cache=True)
    #
    # train_loader = DataLoader(train_ds,
    #            batch_size=1,
    #            shuffle=True,
    #            num_workers=12)
    #
    # for idx, batch_data in enumerate(train_loader):
    #     print(batch_data['image'].shape, batch_data['label'].shape)
    #
    # # val_ds = BTCVDataset(val_files, transform=val_transform, cache=True)