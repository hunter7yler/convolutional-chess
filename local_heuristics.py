""" Local collection of heuristic related evaluation - move sorting,
board evaluation """

from typing import Dict, List
import chess  # type: ignore

from chessmate.constants.piece_values import ConventionalPieceValues
from tensorflow import keras
import numpy as np
from nn_gen import board_convert
from nn_gen import move_index

w_start = keras.models.load_model('./data/start_white')
w_end = keras.models.load_model('./data/end_white')

def NNP(board: chess.Board, piece_values: Dict[str, int] = ConventionalPieceValues) -> List[chess.Move]:
    """
    Neural Net Priority - sort moves based on CNN
    """
    available_captures: Dict[int, List[chess.Move]] = {}
    move_list = list(board.legal_moves)
    bint = np.ndarray(shape=(1, 8, 8, 12))
    bint[0] = board_convert(str(board))
    s_array, e_array = w_start(bint).numpy()[0], w_end(bint).numpy()[0]

    for move in move_list:
        start = str(move)[:2]
        end = str(move)[2:]
        value_diff = (e_array[move_index(end)]) + s_array[move_index(start)]
        if value_diff not in available_captures:
            available_captures[value_diff] = [move]
        else:
            available_captures[value_diff].append(move)

    move_list_sorted = []
    for val_diff in sorted(available_captures, reverse=True):
        move_list_sorted.extend(available_captures[val_diff])
    return move_list_sorted
