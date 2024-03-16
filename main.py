
import cv2
import torch
import board_detection as bd

MODEL_PATH= "model_chess_training/trained_models/model.pth"
bd.BoardElement(bd.Element(torch.tensor([100, 100, 200, 200]), 0, 0.9))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detection = bd.ModelDetect(MODEL_PATH, device)

cap = cv2.VideoCapture(3)



while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        elements = detection.detect_pieces(frame, threshold=0.6)
        board_element = detection.get_board(elements)
        if board_element:
            elements = detection.get_elements_inside_board(board_element, elements, 5)
            detection.add_boxes(frame, elements)

            board = bd.BoardElement(board_element)
            board.add_pieces(elements)
            print(board.get_fen_notation())

            detection
        # Display the resulting frame
        cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()