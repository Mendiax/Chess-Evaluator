
import cv2
import torch
import board_detection as bd

MODEL_PATH= "model_chess_training/trained_models/model.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detection = bd.ModelDetect(MODEL_PATH, device)

cap = cv2.VideoCapture(3)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        pred = detection.detect_pieces(frame)
        detection.add_boxes(frame, pred)
        # Display the resulting frame
        cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()