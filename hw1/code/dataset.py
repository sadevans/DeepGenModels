import os
import warnings
import cv2
import jpeg4py
import pandas as pd
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import v2


warnings.filterwarnings('ignore')


def read_opencv(image_file):
    img = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise ValueError(f'Failed to read {image_file}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def read_image(image_file):
    if image_file.split('.')[-1] in {'jpg', 'jpeg'}:
        try:
            img = jpeg4py.JPEG(image_file).decode()
        except Exception as e:
            print(f'It is not jpg in fact -> {image_file}')
            img = read_opencv(image_file)
    else:
        img = read_opencv(image_file)
    return img


class CustomDataset(data.Dataset):
    def __init__(self, root, annotation_file, is_inference=False, targets_column='class'):
        
        self.root = root
        self.targets_column = targets_column if isinstance(targets_column, str) else targets_column[0]
        self.df = pd.read_csv(annotation_file)
        self.imlist = self.df.values.tolist()
        self.is_inference = is_inference
        
        self.imlist = self.df.values.tolist()

        self.transform = transforms.Compose([
            v2.PILToTensor(),
            v2.Resize((48, 48)),
            v2.ToDtype(dtype=torch.float32, scale=True),
            # v2.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __getitem__(self, index):
        data_item = self.df.iloc[index]
        image_name = data_item['image_name']
        # full_imname = os.path.join(self.root, image_name)

        if os.path.exists(image_name):
            img = read_image(image_name)
        else:
            raise FileNotFoundError(f'No such pic! Check the path {image_name}!')

        try:
            img = Image.fromarray(img)
        except Exception:
            print(f'Problems with that img -> {data_item}')

        img = self.transform(img)

        if not self.is_inference:
            return img

        if self.targets_column in list(self.df.columns):
            target = data_item[self.targets_column]
            if isinstance(target, pd.Series):
                target = target.to_list()

            elif isinstance(target, list) == 1:
                target = target[0]

            return img, target
        else:
            return img

    def __len__(self):
        return len(self.imlist)

    def take_df(self):
        return self.df
