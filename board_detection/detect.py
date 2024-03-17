from dataclasses import dataclass
import logging
from typing import Iterable
import cv2
import numpy as np
import torch
import torch.utils

@dataclass
class Element:
    box : torch.tensor
    label : str
    score : float

    def get_center(self) -> torch.Tensor:
        """Returns the center coordinates of the bounding box."""
        x_min, y_min, x_max, y_max = self.box
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        return torch.tensor([center_x, center_y])



class ModelDetect():
    """Detection class used to detect chess pieces on the image

    Returns:
        _type_: _description_
    """
    CLASSES = ['background', 'B', 'K', 'N', 'P', 'Q', 'R', 'b', 'board', 'k', 'n', 'p', 'q', 'r']

    def __init__(self, model_path, device) -> None:
        self.device=device
        model = torch.load(model_path)
        model.to(device)
        model.eval()
        self.model = model

    @staticmethod
    def _convert_cv2_to_tensor(image_cv2 : cv2.typing.MatLike):
        """Convert image to tensor for prediction model

        Args:
            image_cv2 (cv2.typing.MatLike): image

        Returns:
            Tensor: image as a tensor
        """
        # # Load the image using OpenCV
        # image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
        # image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_GRAY2RGB)

        image_cv2_tensor = torch.from_numpy(image_cv2.transpose(2, 0, 1)).float() / 255.0
        return image_cv2_tensor

    def detect_pieces(self, image_cv2 : cv2.typing.MatLike,  threshold=0.6)-> list[Element]:
        """Detect positions of chess pieces in image

        Args:
            image_cv2 (cv2.typing.MatLike): image
            threshold (float, optional): threshold for reducing random noise. Defaults to 0.6.


        Returns:
            dict: dict with "boxes", "labels", "scores"
        """
        image_cv2_tensor = self._convert_cv2_to_tensor(image_cv2)
        image_cv2_tensor = image_cv2_tensor.to(self.device)


        with torch.no_grad():
            predictions = self.model([image_cv2_tensor, ])

        predictions = [{k: v.cpu() for k, v in pred.items()} for pred in predictions]
        pred = predictions[0]
        # print(pred)
        elements = [
            Element(box.long(), self.CLASSES[int(label)], score)
            for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"])
            if score >= threshold
        ]

        return elements

    @staticmethod
    def get_board(elements : Iterable[Element]):
        """Find board in predictions (if there are many use biggest one)

        Args:
            prediction (Iterable[Element]): predictions
        """
        def find_largest_element(elements: Iterable[Element], element_type) -> Element:
            largest_element = None
            largest_size = float('-inf')

            for elem in elements:
                if not elem.label == element_type:
                    continue
                # Calculate the size of the box
                size = (elem.box[2] - elem.box[0]) * (elem.box[3] - elem.box[1])

                # Update the largest element if the current element has a larger size
                if size > largest_size:
                    largest_element = elem
                    largest_size = size

            return largest_element

        return find_largest_element(elements, 'board')

    @staticmethod
    def get_elements_inside_board(board : Element, elements : Iterable[Element], epsilon=0.0) -> list[Element]:
        """Returns list of elements that are inside board box (+- epsilon)

        Args:
            board (Element): Board element
            epsilon (float): how much other elements can be outside of board
            elements (list[Element]): all detected elements

        Returns:
            list[Element]: List of elements inside board
        """
        inside_elements = []

        board_left = board.box[0]
        board_top = board.box[1]
        board_right = board.box[2]
        board_bottom = board.box[3]

        board_width = board_right - board_left
        board_height = board_bottom - board_top

        width_expansion = board_width * epsilon
        height_expansion = board_height * epsilon

        board_left = board_left - width_expansion
        board_right = board_right + width_expansion
        board_top = board_top - height_expansion
        board_bottom = board_bottom + height_expansion



        # Define the boundaries of the board

        # Check if each element is inside the board
        for elem in elements:
            if elem.label == 'board':
                continue
            x_min, y_min, x_max, y_max = elem.box
            if board_left <= x_min <= board_right and board_top <= y_min <= board_bottom \
                    and board_left <= x_max <= board_right and board_top <= y_max <= board_bottom:
                inside_elements.append(elem)

        return inside_elements

