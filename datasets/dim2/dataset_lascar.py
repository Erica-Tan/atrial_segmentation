import cv2
import copy

from tqdm import tqdm
import albumentations as A
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils.utils import (
    load_nii,
    ConvertToMultiChannel,
    ToTensor,
    EnsureChannelFirst,
    SampleWiseNormalize,
    load_datalist
)


class LAScarDataset(Dataset):
    def __init__(
            self,
            data,
            mode="train",
            transform=None,
    ):
        self.transform = transform
        self.mode = mode

        assert mode in ['train', 'test']

        img_list = []
        lab_list = []
        meta_list = []

        for data_i in tqdm(data, desc="Loading dataset"):
            img = load_nii(data_i['image'])
            lab = load_nii(data_i['label'])

            if img.shape != lab.shape:
                continue

            assert img.shape == lab.shape

            img_list.append(img)
            lab_list.append(lab)
            meta_list.append({"patient": data_i["patient"]})

        self.img_slice_list = []
        self.lab_slice_list = []
        self.meta_slice_list = []
        if self.mode == 'train':
            for i in range(len(img_list)):
                C, W, D = img_list[i].shape

                for j in range(D):
                    self.img_slice_list.append(copy.deepcopy(img_list[i][:, :, j]))
                    self.lab_slice_list.append(copy.deepcopy(lab_list[i][:, :, j]))
                    self.meta_slice_list.append(meta_list[i])

        else:
            self.img_slice_list = img_list
            self.lab_slice_list = lab_list
            self.meta_slice_list = meta_list

        del img_list, lab_list, meta_list

    def __len__(self):
        return len(self.img_slice_list)

    def __getitem__(self, idx):
        image = self.img_slice_list[idx]
        label = self.lab_slice_list[idx]
        meta_data = self.meta_slice_list[idx]

        if self.transform:
            transformed = self.transform(image=image, mask=label)

            image = transformed["image"]
            label = transformed["mask"]

        return {"image": image, "label": label, "meta_data": meta_data}


if __name__ == "__main__":
    datalist = load_datalist('./data/dataset_lascar.json', True, "testing", base_dir='./data/dataset_lascar_processed')

    transform = A.Compose(
        [
            A.Resize(width=272, height=272),


            # standardization
            A.Normalize(mean=0.18, std=0.22, max_pixel_value=1),
            EnsureChannelFirst(),
            ConvertToMultiChannel(),

            ToTensor()
        ]
    )

    dataset = LAScarDataset(datalist[3:5], mode='test', transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=1,
    )

    for idx, batch_data in enumerate(loader):
        print(batch_data['image'].shape, batch_data['label'].shape, batch_data['image'].min(),
              batch_data['image'].max(), batch_data['label'].dtype, batch_data['image'].dtype, batch_data['meta_data'])

