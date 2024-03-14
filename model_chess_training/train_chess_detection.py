import os
import torch
import torch.utils

from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
from chess_pieces_dataset import ChessPiecesDataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import detection

import train_utils
import config

train_dataset = ChessPiecesDataset(config.PATH_TO_DATASETS, train_utils.get_transform(True),  image_set='train')
test_dataset  = ChessPiecesDataset(config.PATH_TO_DATASETS, train_utils.get_transform(False), image_set='test')

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=detection.collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=detection.collate_fn)


model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
num_classes = len(config.DATASET_LABELS)  # pieces + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0005
)
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

# let's train it just for 2 epochs
num_epochs = 10
detection.evaluate(model, test_dataloader, device=device)
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    try:
        detection.train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=10)
        torch.save(model, os.path.join('trained_models',f'model_epoch{epoch}.pth'))


        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        detection.evaluate(model, test_dataloader, device=device)
    except Exception as e:
        print('Error happened', e)
torch.save(model, os.path.join('trained_models','model.pth'))
print('train done')
