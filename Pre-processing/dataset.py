import os
import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import json

class CustomDataset(Dataset):

    def __init__(self, img_root, ann_root, target_type='coco', transforms=None):

        self.img_root = img_root
        self.ann_root = ann_root
        self.target_type = target_type
        self.transforms = transforms

        with open(ann_root) as json_file:
            self.data = json.load(json_file)

    def __getitem__(self, idx):

        image_name = self.data['images'][idx]['file_name']
        image_id = self.data['images'][idx]['id']

        img_path = os.path.join(self.img_root,image_name)

        img = Image.open(img_path).convert("RGB")

        # Initialisation du dictionnaire
        target = {}

        boxes = []
        area = []
        labels = []

        for ann in range(len(self.data['annotations'])):

            if self.data['annotations'][ann]['image_id']== image_id:

                if self.target_type == 'coco':
                    boxes.append(self.data['annotations'][ann]['bbox'])

                elif self.target_type == 'pascal':
                    bndbox = self.data['annotations'][ann]['bbox']
                    xmin = bndbox[0]
                    ymin = bndbox[1]
                    xmax = bndbox[0] + bndbox[2]
                    ymax = bndbox[1] + bndbox[3]
                    boxes.append([xmin,ymin,xmax,ymax])

                labels.append(self.data['annotations'][ann]['category_id'])
                area.append(self.data['annotations'][ann]['area'])

        # Image ID
        image_id = idx

        # Remplissage du dictionnaire
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area

        return img, target

    def __len__(self):
        return len(self.data['images'])
