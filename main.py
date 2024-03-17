
import json
import cv2
import torch
import board_detection as bd
from stockfish import Stockfish


with open('config.json') as fd:
    config = json.load(fd)

path_to_stockfish = config["stockfish_path"]
path_to_model = config["model_path"]


MODEL_PATH= "model_chess_training/trained_models/model.pth"
bd.BoardElement(bd.Element(torch.tensor([100, 100, 200, 200]), 0, 0.9))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detection = bd.ModelDetect(MODEL_PATH, device)

cap = cv2.VideoCapture(3)


stockfish = Stockfish(path_to_stockfish)

last_fen = ''

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

            fen = board.get_fen_notation()
            fen += ' w KQkq - 0 1'
            if fen != last_fen:
                print(fen)
                last_fen = fen
                try:
                    if stockfish.is_fen_valid(fen):
                        stockfish.set_fen_position(fen)
                        fen = stockfish.get_fen_position()
                        print(fen)
                        # Get the best move from Stockfish
                        best_move = stockfish.get_best_move()
                        print("Best move:", best_move)
                        eval = stockfish.get_evaluation()
                        print("Eval:", eval)
                except Exception as e:
                    print(e)
        # Display the resulting frame
        cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()