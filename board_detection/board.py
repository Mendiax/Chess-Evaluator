from .detect import *

class BoardElement:
    def __init__(self, board : Element) -> None:
        self.borders = board.box

        x_min, y_min, x_max, y_max = self.borders
        width = (x_max - x_min) / 8
        height = (y_max - y_min) / 8
        self.grid_distance = (width, height)

        self.pieces = [
            [None] * 8 for _ in range(8)
        ]

    def clear_board(self):
        self.pieces = [
            [None] * 8 for _ in range(8)
        ]


    def add_pieces(self, pieces : Iterable[Element]):
        """Add all pieces to board grid.

        Args:
            pieces (Iterable[Element]): pieces
        """
        for piece in pieces:
            center = piece.get_center()
            x_min, y_min, x_max, y_max = self.borders
            grid_x = int((center[0] - x_min) // self.grid_distance[0])
            grid_y = int((center[1] - y_min) // self.grid_distance[1])

            # Check if the piece is within the board boundaries
            if 0 <= grid_x < 8 and 0 <= grid_y < 8:
                if self.pieces[grid_y][grid_x] is None or\
                    piece.score > self.pieces[grid_y][grid_x].score:
                    self.pieces[grid_y][grid_x] = piece


    def get_fen_notation(self) -> str:
        """Returns the FEN notation representation of the board."""
        fen_notation = []

        for row in self.pieces:
            fen_row = ''
            empty_count = 0
            for piece in row:
                if piece is None:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += self._get_piece_fen(piece.label)
            if empty_count > 0:
                fen_row += str(empty_count)
            fen_notation.append(fen_row)

        return '/'.join(fen_notation)

    def _get_piece_fen(self, label: int) -> str:
        """Returns the FEN representation of a piece label."""
        piece_labels = ['.', 'B', 'K', 'N', 'P', 'Q', 'R', 'b', 'board', 'k', 'n', 'p', 'q', 'r']
        return piece_labels[label]

    def display_board(self):
        """Displays the board grid in a visually appealing format."""
        for row in self.pieces:
            row_str = ''
            for piece in row:
                if piece is None:
                    row_str += '. '
                else:
                    row_str += self._get_piece_fen(piece.label) + ' '
            print(row_str)