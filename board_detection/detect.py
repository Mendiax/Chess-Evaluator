import cv2
import torch
import torch.utils


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
        # Load the image using OpenCV
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_GRAY2RGB)

        image_cv2_tensor = torch.from_numpy(image_cv2.transpose(2, 0, 1)).float() / 255.0
        return image_cv2_tensor

    def detect_pieces(self, image_cv2 : cv2.typing.MatLike, threshold=0.6):
        """Detect positions of chess pieces in image

        Args:
            image_cv2 (cv2.typing.MatLike): image

        Returns:
            dict: dict with "boxes", "labels", "scores"
        """
        image_cv2_tensor = self._convert_cv2_to_tensor(image_cv2)
        image_cv2_tensor = image_cv2_tensor.to(self.device)


        with torch.no_grad():
            predictions = self.model([image_cv2_tensor, ])

        predictions = [{k: v.cpu() for k, v in pred.items()} for pred in predictions]
        pred = predictions[0]

        return pred

    @staticmethod
    def add_boxes(image : cv2.typing.MatLike, prediction : dict):
        """Add prediction squares with labels on image

        Args:
            image (cv2.typing.MatLike): image to modify
            prediction (dict): dictionary with predictions
        """

        for box, label, score in zip(prediction["boxes"].long(), prediction["labels"], prediction["scores"]):
            label_text = f"{ModelDetect.CLASSES[label]}:{score:.1f}"

            if score > 0.6:
                pred = box.cpu().numpy()
                cv2.rectangle(image, pred[0:2], pred[2:4], (0,0,255), 1)
                text_pos = [pred[0], pred[3]]
                cv2.putText(image, label_text, text_pos, 2, 0.5, (0,0,255))