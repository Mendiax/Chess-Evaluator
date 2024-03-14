import logging
import cv2
import numpy as np
import torch
import board_detection as bd

def test_detect_simple():
    MODEL_PATH = "model_chess_training/trained_models/model.pth"
    TEST_IMAGE_PATH = 'tests/board_detection/test_images/board_test_2.jpg'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detection = bd.ModelDetect(MODEL_PATH, device)

    test_image = cv2.imread(TEST_IMAGE_PATH)

    pred = detection.detect_pieces(test_image)
    # check if all pieces and board is detected
    assert len(set(filter(lambda x : x > 0.5, pred["scores"]))) == 33