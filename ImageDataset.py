import numpy as np
import torch
import torch.nn as nn
import glob
import os
from PIL import Image
from torchvision.transforms import transforms

# https://www.kaggle.com/code/balraj98/cyclegan-translating-apples-oranges-pytorch
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train", debug_mode=False):
        self.transform = transforms_
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, f"{mode}A") + "/*.*"))
        if debug_mode:
            self.files_A = self.files_A[:100]

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = Image.new('RGB', image_A.size).paste(image_A)

        item_A = self.transform(image_A)

        return item_A

    def __len__(self):
        return len(self.files_A)