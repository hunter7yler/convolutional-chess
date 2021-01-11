"""
Collection of local chess engines designed to test effectiveness of neural networks
"""
import random
from typing import Dict, List, Union

import chess  # type: ignore
import chess.pgn  # type: ignore

from chessmate.analysis import StandardEvaluation
from chessmate.constants.piece_values import ConventionalPieceValues
from chessmate.heuristics import MVV_LVA
from chessmate.transpositions import TranspositionTable, zobrist_hash_function
from chessmate.utils import get_piece_at
from chessmate.engines import BaseEngine
from nn_gen import board_convert
from nn_gen import move_index

import numpy as np
from tensorflow import keras


# def nn9white_player(board):  # note that "player" receives the board
#     moves = [str(x) for x in list(board.legal_moves)] #string rep
#     bint = np.ndarray(shape=(1, 8, 8, 12))
#     bint[0] = board_convert(str(board))
#     s_array, e_array = w_start(bint).numpy()[0], w_end(bint).numpy()[0]
#     scored = []
#     for move in moves:
#         newboard = board.copy()
#         newboard.push_uci(move)
#         if newboard.is_checkmate():
#             return move
#         start = move[0:2]
#         end = move[2:4]
#         score = (e_array[move_index(end)]) #s_array[move_index(start)] #+
#         scored.append([score, move])
#     scored.sort(reverse=True)
#     choice = scored[0][1]
#     return choice

class HighestValueCNN(BaseEngine):
    """ Engine that prioritizes capturing the highest value piece if
    the option presents itself, uses a neural network to choose between equally valued moves"""

    def __init__(self):
        """ See parent docstring """
        super().__init__()
        self.name = "Capture Highest Value, prioritized with NN"
        self.start_nn = keras.models.load_model('./data/start_white')
        self.end_nn = keras.models.load_model('./data/end_white')

    def evaluate(self, board: chess.Board) -> None:
        """ Assigns highest value to capture moves based off value system """
        self.reset_move_variables()

        bint = np.ndarray(shape=(1, 8, 8, 12))
        bint[0] = board_convert(str(board))
        s_array, e_array = self.start_nn(bint).numpy()[0], self.end_nn(bint).numpy()[0]

        legal_move_list = list(board.legal_moves)
        for m in legal_move_list:
            piece_at_position = get_piece_at(board, str(m)[2:4]).upper()

            if (not board.is_capture(m)) or (not piece_at_position):
                self.legal_moves[m] = 0.0 + s_array[move_index(str(m)[0:2])] #+ (e_array[move_index(str(m)[2:4])])
            else:
                self.legal_moves[m] = self.value_mapping[
                    piece_at_position
                ].value + s_array[move_index(str(m)[0:2])]

        self.material_difference.append(
            self.evaluation_function.evaluate(board)
        )


    def move(self, board: chess.Board) -> chess.Move:
        """Select move that features capture out of random moves
        if one is available. See parent docstring"""
        self.evaluate(board)

        if not [*self.legal_moves]:
            return chess.Move.null()

        highest_capture_val, highest_capture_uci = 0.0, None

        # Find move with highest capture value
        for m in list(self.legal_moves):
            if self.legal_moves[m] > highest_capture_val:
                highest_capture_val, highest_capture_uci = (
                    self.legal_moves[m],
                    m,
                )

        return highest_capture_uci


class CNN(BaseEngine):
    """ Uses a CNN to choose next move """

    def __init__(self):
        """ See parent docstring """
        super().__init__()
        self.name = "Uses a CNN to choose next move"
        self.start_nn = keras.models.load_model('./data/start_white')
        self.end_nn = keras.models.load_model('./data/end_white')

    def evaluate(self, board: chess.Board) -> None:
        """ Assigns highest value to capture moves based off value system """
        self.reset_move_variables()

        bint = np.ndarray(shape=(1, 8, 8, 12))
        bint[0] = board_convert(str(board))
        s_array, e_array = self.start_nn(bint).numpy()[0], self.end_nn(bint).numpy()[0]

        legal_move_list = list(board.legal_moves)
        for m in legal_move_list:
            self.legal_moves[m] = 0.0 + s_array[move_index(str(m)[0:2])] #+ (e_array[move_index(str(m)[2:4])])

        self.material_difference.append(
            self.evaluation_function.evaluate(board)
        )


    def move(self, board: chess.Board) -> chess.Move:
        """Select move that based on CNN applied score"""
        self.evaluate(board)

        if not [*self.legal_moves]:
            return chess.Move.null()

        highest_val, highest_uci = 0.0, None

        for m in list(self.legal_moves):
            if self.legal_moves[m] > highest_val:
                highest_capture_val, highest_uci = (
                    self.legal_moves[m],
                    m,
                )

        return highest_uci