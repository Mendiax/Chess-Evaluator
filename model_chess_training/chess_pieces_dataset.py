import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat
from torchvision import tv_tensors

class ChessPiecesDataset(Dataset):
    """Dataset that reads dataset for chess pieces
    """
    def __init__(self, root_dir, transforms, image_set='train'):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, image_set, 'images')
        self.label_dir = os.path.join(root_dir, image_set, 'labels')
        self.image_files = sorted(os.listdir(self.image_dir))
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image_tensor = read_image(img_name)
        image_tensor = tv_tensors.Image(image_tensor)

        label_name = os.path.join(self.label_dir, self.image_files[idx].replace('.jpg', '.txt'))
        with open(label_name, 'r') as file:
            lines = file.readlines()
            boxes = []
            labels = []
            img_height, img_width = image_tensor.shape[1:]

            for line in lines:
                label = line.strip().split()
                label = [int(label[0])] + [float(x) for x in label[1:]]
                labels.append(label[0] + 1) # offset from 1
                label = np.array(label).astype(np.float32)
                box_center = (label[1], label[2])
                box_w = label[3]
                box_h = label[4]

                x_min = int((box_center[0] - box_w / 2) * img_width)
                y_min = int((box_center[1] - box_h / 2) * img_height)
                x_max = int((box_center[0] + box_w / 2) * img_width)
                y_max = int((box_center[1] + box_h / 2) * img_height)

                boxes.append([x_min, y_min, x_max, y_max])

        boxes = torch.tensor(boxes)
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])


        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target['boxes'] = BoundingBoxes(boxes, format=BoundingBoxFormat.XYXY, canvas_size=(img_height, img_width))
        target['labels'] = torch.tensor(labels, dtype=torch.int64)
        target["iscrowd"] = iscrowd
        target["image_id"] = idx
        target["area"] = areas

        if self.transforms is not None:
            image_tensor, target = self.transforms(image_tensor, target)

        return image_tensor, target

