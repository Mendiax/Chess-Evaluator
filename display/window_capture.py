from typing import Iterable
import cv2
import numpy as np
import pyautogui
import torch

from board_detection.board import BoardElement
from board_detection.detect import Element



class WindowCapture:

    @staticmethod
    def get_frame():
        frame_pygui = pyautogui.screenshot()
        frame = np.array(frame_pygui)
        return frame

    @staticmethod
    def prepare_frame_for_detection(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        frame = WindowCapture._adjust_contrast(frame, 1.5)

        return frame


    @staticmethod
    def _adjust_contrast(image : cv2.typing.MatShape, contrast_factor = 1.5):
        # Adjust as needed, higher values increase contrast, lower values decrease contrast
        image_np = np.array(image, dtype=np.float32)
        adjusted_image = cv2.convertScaleAbs(image_np, alpha=contrast_factor, beta=0)
        image = np.clip(adjusted_image, 0, 255).astype(np.uint8)
        return image

    @staticmethod
    def add_boxes(image : cv2.typing.MatLike, elements : Iterable[Element]):
        """Add prediction squares with labels on image

        Args:
            image (cv2.typing.MatLike): image to modify
            prediction (dict): dictionary with predictions
        """

        for elem in elements:
            if elem is None:
                continue

            label_text = f"{elem.label}:{elem.score:.2f}"

            pred : torch.tensor = elem.box
            p1 = (int(pred[0]), int(pred[1]))  # Convert coordinates to integers
            p2 = (int(pred[2]), int(pred[3]))  # Convert coordinates to integers


            cv2.rectangle(image, p1, p2, (0,0,255), 1)
            text_pos = (int(pred[0]), int(pred[3]))
            cv2.putText(image, label_text, text_pos, 2, 0.5, (0,0,255))

    @staticmethod
    def _zoom_to_board(frame : cv2.typing.MatLike, board : BoardElement, chess_board_display_width):
        x_min, y_min, x_max, y_max = board.borders
        roi = frame[y_min:y_max, x_min:x_max]
        frame = cv2.resize(roi,  (chess_board_display_width, chess_board_display_width))
        return frame

    @staticmethod
    def get_display_frame(frame : cv2.typing.MatLike, board : BoardElement, windows_size : tuple[int,int]):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # add boxes on pieces
        WindowCapture.add_boxes(frame, np.array(board.pieces).flatten())

        # move camera to board
        chess_board_display_width = int(min(windows_size) * 0.9)
        frame_board = WindowCapture._zoom_to_board(frame, board, chess_board_display_width)


        # embed chess view on blank image
        image_with_board = np.zeros((*windows_size, 3), dtype=np.uint8)
        image_height, image_width = frame_board.shape[:2]
        top_right_corner = (windows_size[1] - image_width, 0)
        image_with_board[top_right_corner[1]:top_right_corner[1] + image_height,
                    top_right_corner[0]:top_right_corner[0] + image_width] = frame_board

        return image_with_board