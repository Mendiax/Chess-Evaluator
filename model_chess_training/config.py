

import os


TOP_FOLDER = os.path.dirname(__file__)
OUTPUT_PATH = os.path.join(TOP_FOLDER, 'trained_models')
PATH_TO_DATASETS= os.path.join(TOP_FOLDER,'datasets','2D Chessboard and Chess Pieces.v4i.yolov7pytorch')
DATASET_LABELS = ['background', 'B', 'K', 'N', 'P', 'Q', 'R', 'b', 'board', 'k', 'n', 'p', 'q', 'r']