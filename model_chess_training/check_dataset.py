# Function to check if dataset works correctyl
import os
import torch
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.ops import box_convert
from chess_pieces_dataset import ChessPiecesDataset

import model_chess_training.train_utils as train_utils
import config

train_dataset = ChessPiecesDataset(config.PATH_TO_DATASETS, train_utils.get_transform(True),  image_set='train')

# Set the index of the image you want to display
idx = 123

# Get image and corresponding boxes
image_tensor, target = train_dataset[idx]

no_to_show = 2
boxes = target['boxes'][0:10]
labels = list(map(lambda x : config.DATASET_LABELS[x], target['labels']))

# Convert boxes to [xmin, ymin, xmax, ymax] format for visualization
boxes = torch.tensor(boxes)
boxes = box_convert(boxes, in_fmt='xyxy', out_fmt='xywh')  # Convert to xywh format
boxes = box_convert(boxes, in_fmt='xywh', out_fmt='xyxy')  # Convert back to xyxy format

# Convert image tensor to numpy array
image_np = image_tensor.permute(1, 2, 0).numpy()

# Display image with bounding boxes
plt.figure(figsize=(10, 10))
plt.imshow(image_np)

for box in boxes:
    xmin, ymin, xmax, ymax = box.tolist()
    plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], color='r')

plt.axis('off')
print(f'{labels=}')

plt.show()
