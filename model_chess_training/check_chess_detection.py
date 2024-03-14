import math
import os
import sys
import numpy as np
import torch
import torch.utils

from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

from PIL import Image
from chess_pieces_dataset import ChessPiecesDataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


import utils
import config
import detection

# Create datasets and data loaders for train, validation, and test sets
train_dataset = ChessPiecesDataset(config.PATH_TO_DATASETS, utils.get_transform(True),  image_set='train')
valid_dataset = ChessPiecesDataset(config.PATH_TO_DATASETS, utils.get_transform(False), image_set='valid')
test_dataset  = ChessPiecesDataset(config.PATH_TO_DATASETS, utils.get_transform(False), image_set='test')

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=detection.collate_fn)
valid_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=detection.collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=detection.collate_fn)


model = torch.load(f'{config.OUTPUT_PATH}/model.pth')
# print(model)
model = model.to('cpu')
model.eval()
import cv2

CLASSES = ['background', 'B', 'K', 'N', 'P', 'Q', 'R', 'b', 'board', 'k', 'n', 'p', 'q', 'r']
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

img_name = 'tests/board_detection/test_images/board_test_2.jpg'
def convert_cv2_to_tensor(image_cv2):
    # Load the image using OpenCV
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', image_cv2)
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_GRAY2RGB)
    cv2.imshow('gray to rgb', image_cv2)

    image_cv2_tensor = torch.from_numpy(image_cv2.transpose(2, 0, 1)).float() / 255.0
    return image_cv2_tensor
image_cv2 = cv2.imread(img_name)
image = image_cv2.copy()
image_cv2_tensor = convert_cv2_to_tensor(image_cv2)


with torch.no_grad():
    predictions = model([image_cv2_tensor, ])
    pred = predictions[0]

pred_labels = [f"{CLASSES[label]}:{score:.1f}" for label, score in zip(pred["labels"], pred["scores"])]


pred_boxes = pred["boxes"].long()

print(pred_boxes)

for box, label, score in zip(pred["boxes"].long(), pred["labels"], pred["scores"]):
    label_text = f"{CLASSES[label]}:{score:.1f}"
    print(box)
    print(label_text)

    if score > 0.2:
        pred = box.cpu().numpy()
        # cv2.rectangle(image, pred[0:2].numpy(), pred[2:4].numpy(), (0,0,255), 1)
        text_pos = [pred[0], pred[3]]
        cv2.putText(image, label_text, text_pos, 2, 0.5, (0,0,255))
    # print(label)
cv2.imshow('boxes', image)
# cv2.waitKey(3000)
cv2.waitKey(0)
cv2.destroyAllWindows()