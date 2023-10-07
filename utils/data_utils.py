import glob
import math
import cv2
import albumentations as A
import numpy as np
import torch
from torch.utils.data import DataLoader
from monai import transforms, data

from utils.utils import (
    ConvertToMultiChannel,
    ToTensor,
    EnsureChannelFirst,
    load_datalist,
    ConvertToMultiChannelBasedOnClassesd
)
from datasets.dim3.dataset_btcv import BTCVDataset
from datasets.dim2.dataset_lascar import LAScarDataset


def get_loader(args):
    if args.spatial_dims == 2:
        if args.dataset == 'lascar':
            return get_lascar_loader(args)
        else:
            raise ValueError(f"Unsupported Dataset: 2d {args.dataset}")
    elif args.spatial_dims == 3:
        if args.dataset == 'btcv':
            return get_btcv_3d_loader(args)
        else:
            raise ValueError(f"Unsupported Dataset: 3d {args.dataset}")
    else:
        raise ValueError('Invalid dimension, should be \'2d\' or \'3d\'')


def get_lascar_loader(args):
    data_dir = args.data_dir
    datalist_json = args.json_list

    val_transform = []

    if args.transform == 'centercrop':
        val_transform.append(A.CenterCrop(args.roi_x, args.roi_y))
    elif args.transform == 'resize':
        val_transform.append(A.Resize(width=args.roi_x, height=args.roi_y))

    val_transform.extend([
            A.Normalize(mean=0.22, std=0.24, max_pixel_value=1),
            EnsureChannelFirst(),
            ConvertToMultiChannel(binary=True),
            ToTensor()
        ])

    val_transform = A.Compose(val_transform)

    if args.test_mode:
        test_files = load_datalist(datalist_json, True, "testing", base_dir=data_dir)
        test_ds = LAScarDataset(data=test_files, transform=val_transform, mode='test')
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=test_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = test_loader
    else:
        raise ValueError(f"Unsupported Dataset Mode: {args.dataset} {args.test_mode}")

    return loader



def get_btcv_3d_loader(args):
    data_dir = args.data_dir
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
    if args.test_mode:
        test_ds = BTCVDataset(val_files, transform=val_transform, cache=True)
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=test_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = test_loader
    else:
        train_ds = BTCVDataset(train_files, transform=train_transform, cache=True)
        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        val_ds = BTCVDataset(val_files, transform=val_transform, cache=True)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=val_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = [train_loader, val_loader]

    return loader


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank: self.total_size: self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank: self.total_size: self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from types import SimpleNamespace

    args = SimpleNamespace(
        data_dir='/media/erica/Data/Projects/medical_imaging/datasets/BTCV/Training/',
        json_list='./data/dataset_utah.json',
        roi_x=20,
        roi_y=320,
        roi_z=320,
        use_normal_dataset=False,
        transform='centercrop',
        test_mode=False,
        distributed=False,
        workers=1,
        batch_size=2,
    )

    train_loader, val_loader = get_btcv_3d_loader(args)
    print(len(train_loader), len(val_loader))

    for idx, batch_data in enumerate(train_loader):
        print(batch_data['image'].shape, batch_data['label'].shape, batch_data['image'].min(),
              batch_data['image'].max(), batch_data['label'].dtype, batch_data['image'].dtype)

        # plt.imsave(f'test.png', batch_data['image'][0, 0, :, :, 12], cmap=plt.cm.gray)
        break
