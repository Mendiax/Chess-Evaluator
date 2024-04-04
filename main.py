
import json
import time
import cv2
import numpy as np
import torch
import board_detection as bd
from stockfish import Stockfish
import display
# import window_capture

# Load configuration file
with open('config.json') as fd:
    config = json.load(fd)
path_to_stockfish = config["stockfish_path"]
path_to_model = config["model_path"]
target_width = 1280
target_height = 720
THRESHOLD=0.9
# how much pieces can be outside of border (in scale of board)
BORDER_EPS=0.01

# Load model for image detection
MODEL_PATH= "model_chess_training/trained_models/model.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detection = bd.ModelDetect(MODEL_PATH, device)


# Turn on window capture
# wincap = window_capture.WindowCapture('Client')



stockfish = Stockfish(path_to_stockfish)


last_fen = ''
display_text = ''
loop_time = time.time()
fps = 0
while True:
    # get an updated image of the game
    frame = display.WindowCapture.get_frame()

    # preprocess to detect

    frame_detect = display.WindowCapture.prepare_frame_for_detection(frame.copy())
    elements = detection.detect_pieces(frame_detect, threshold=THRESHOLD)
    board_element = detection.get_board(elements)
    if board_element:
        elements = detection.get_elements_inside_board(board_element, elements, BORDER_EPS)

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

                    display_text = ''
                    if eval['type'] == 'cp':
                        adv = eval['value'] / 100.0
                        display_text = f'{adv:.1f}'
                    elif eval['type'] == 'mate':
                        mate_in = abs(eval['value'])
                        color_adv = 'white' if eval['value'] > 0 else 'black'
                        display_text = f'M{mate_in}'


                    print("Eval:", eval)
            except Exception as e:
                print(e)

    gui_frame = display.WindowCapture.get_display_frame(frame, board, (1080, 1120))
    cv2.putText(gui_frame, f'FPS:{fps:.1f}', (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(gui_frame, f'{fen=}', (10, gui_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(gui_frame, f'{display_text}', (10, int(gui_frame.shape[0]/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


    cv2.imshow('Chess Evaluator', gui_frame)

    fps = 1 / (time.time() - loop_time)
    loop_time = time.time()

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cv2.destroyAllWindows()