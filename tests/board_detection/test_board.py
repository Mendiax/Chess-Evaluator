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
    assert len(elements) >= 33
    board = detection.get_board(elements, threshold=0.2)
    epsilon = 0
    elements = detection.get_elements_inside_board(board, elements, epsilon)
    logging.info(elements)
    assert len(elements) >= 32
    while len(elements) < 32 and epsilon < 1000:
        elements = detection.get_elements_inside_board(board, elements, epsilon)
        epsilon += 1

    # check if all pieces and board is detected
    assert len(elements) == 32, f'{epsilon=}'