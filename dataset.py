# -*- coding: utf-8 -*-
"""
Writer: WJQpe
Date: 2022 04 02 
"""
import torch
import os
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class COCO2014(Dataset):
    def __init__(self, images_path, transform=None, image_num=None):
        self.images_path = images_path
        self.transform = transform
        self.image_list = os.listdir(images_path)
        if image_num is not None:
            self.image_list = self.image_list[:image_num]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        image_name = os.path.join(self.images_path, self.image_list[item])
        image = read_image(image_name, mode=ImageReadMode.RGB)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def collate_fn(batch):
        images = zip(*batch)
        images = torch.stack(images, dim=0)
        return images


def transform_image(resize=256, gray=False):
    if gray:
        tf_list = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(400),
                transforms.RandomCrop(resize),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()
            ]
        )
    else:
        tf_list = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(400),
                transforms.RandomCrop(resize),
                transforms.ToTensor()
            ]
        )
    return tf_list


from args_fusion import set_args
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    args = set_args()
    transform = transform_image(gray=args.gray)
    coco_dataset = COCO2014(images_path=args.image_path, transform=transform)
    print(coco_dataset.__len__())
    image = coco_dataset.__getitem__(20)
    print(type(image))
    print(image.shape)
    print(image.max())
    print(image.min())

    img_np = image.numpy()
    print(type(img_np))

    plt.axis("off")
    if args.gray:
        plt.imshow(np.transpose(img_np, (1, 2, 0)), cmap='gray')
    else:
        plt.imshow(np.transpose(img_np, (1, 2, 0)))
    plt.show()
