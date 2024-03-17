import logging
import cv2
import numpy as np
import torch
import board_detection as bd

def test_detect_board():
    MODEL_PATH = "model_chess_training/trained_models/model.pth"
    TEST_IMAGE_PATH = 'tests/board_detection/test_images/board_test_2.jpg'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detection = bd.ModelDetect(MODEL_PATH, device)

    test_image = cv2.imread(TEST_IMAGE_PATH)

    elements = detection.detect_pieces(test_image)
    assert len(elements) >= 33, 'No enough objects'


    board = detection.get_board(elements)
    epsilon = 0.45
    elements_in_board = detection.get_elements_inside_board(board, elements, epsilon)

    # check if all pieces and board is detected
    assert len(elements_in_board) == 32, f'{epsilon=}'
