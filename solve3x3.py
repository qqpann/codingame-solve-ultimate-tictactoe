import math
import random
import sys
import time
from copy import deepcopy
from typing import List

import numpy as np

DEPTH = 1
SIZE = 3


def init_board():
    return np.zeros((3, 3), dtype=np.int8)


def iter_3x3():
    for x in range(3):
        for y in range(3):
            yield (x, y)


def tic_tac_toe_winner(board: np.ndarray):
    assert len(board) == 3 and len(board[0]) == 3
    # rows -
    for r in range(3):
        tmp = set([board[r][c] for c in range(3)])
        if len(tmp) == 1:
            return list(tmp)[0]
    # cols |
    for c in range(3):
        tmp = set([board[r][c] for r in range(3)])
        if len(tmp) == 1:
            return list(tmp)[0]
    # diagonal \
    tmp = set([board[i][i] for i in range(3)])
    if len(tmp) == 1:
        return list(tmp)[0]
    # diagonal /
    tmp = set([board[i][3 - 1 - i] for i in range(3)])
    if len(tmp) == 1:
        return list(tmp)[0]
    return 0


def has_winner(board: np.ndarray):
    return tic_tac_toe_winner(board) != 0


def get_legal_actions(board: np.ndarray):
    legal = []
    for x in range(SIZE):
        for y in range(SIZE):
            if board[x][y] == 0:
                legal.append((x, y))
    return legal


def is_done(board: np.ndarray):
    if has_winner(board):
        return True
    if len(get_legal_actions(board)) == 0:
        return True
    return False


def act(board: np.ndarray, turn: int, x: int, y: int) -> np.ndarray:
    assert board[x][y] == 0, "Illegal act"
    board[x][y] = turn
    return board


def str_board(board: List[List[int]]):
    res = ""
    for row in board:
        res += "".join(map(str, row)) + "\n"
    return res


def get_opposite_turn(turn: int):
    return 2 if turn == 1 else 1


class UltimateTicTacToe:
    def __init__(self):
        self.board = init_board()
        self.turn = 1

    def __str__(self):
        return str_board(self.board)

    def flip_turn(self):
        self.turn = get_opposite_turn(self.turn)

    def act(self, turn: int, x: int, y: int):
        self.board = act(self.board, turn, x, y)
        self.turn = get_opposite_turn(turn)
        return self

    def play(self):
        while not is_done(self.board):
            # Replace with AI here.
            print(self)
            start = time.perf_counter()
            if self.turn == 1:
                # Human: 1
                x, y = map(int, input("Move>").split())
            else:
                # Bot: 2
                legal_actions = get_legal_actions(self.board)
                print(legal_actions, self.board)
                action_rewards = [
                    minimax(
                        act(self.board.copy(), self.turn, x, y),
                        DEPTH,
                        get_opposite_turn(self.turn),
                        self.turn,
                    )
                    for x, y in legal_actions
                ]
                # print(legal_actions)
                # print(action_rewards)
                max_reward = max(action_rewards)
                max_actions = [
                    a for a, r in zip(legal_actions, action_rewards) if r == max_reward
                ]
                x, y = random.choice(max_actions)
                # x, y = random.choice(self.legal_actions())
            end = time.perf_counter()
            print("Time:", end - start)

            self.act(self.turn, x, y)
            # for x, y in iter_3x3():
            #     print(str_board(self.sub_board(x, y)))
            # print("meta board")

            # print(str_board(self.meta_board()))
            # print(tic_tac_toe_winner(self.meta_board()))

    def solve(self):
        while not is_done(self.board):
            # Get input
            opponent_row, opponent_col = map(int, input().split())
            valid_action_count = int(input())
            valid_actions = []
            for i in range(valid_action_count):
                row, col = [int(j) for j in input().split()]
                valid_actions.append((row, col))
            # ##########

            # Write an action using print
            # To debug: print("Debug messages...", file=sys.stderr, flush=True)
            if opponent_col == -1 and opponent_row == -1:
                pass
            else:
                # Opponent
                self.act(2, opponent_col, opponent_row)
            act = random.choice(get_legal_actions(self.board))
            # Us
            self.act(1, act[0], act[1])
            print(f"{act[0]} {act[1]}")


def score(board: np.ndarray, player: int):
    assert player in [1, 2]
    meta_win = tic_tac_toe_winner(board)
    if meta_win != 0:
        return 10 if meta_win == player else -10
    return 0


def minimax(board: np.ndarray, depth: int, turn: int, maximizing_player: int) -> float:
    """
    MiniMax function
    """
    # Ref: https://en.wikipedia.org/wiki/Minimax
    return alphabeta(board, depth, turn, maximizing_player, -float("inf"), float("inf"))


def alphabeta(
    board: np.ndarray,
    depth: int,
    turn: int,
    maximizing_player: int,
    alpha: float,
    beta: float,
) -> float:
    """
    MiniMax with alpha-beta pruning
    """
    # Ref: https://en.wikipedia.org/wiki/Alpha–beta_pruning
    if depth == 0 or is_done(board):
        return score(board, maximizing_player)

    legal_actions = get_legal_actions(board)
    opponent = 2 if maximizing_player == 1 else 1
    if turn == maximizing_player:
        for x, y in legal_actions:
            child = board.copy()
            child = act(child, turn, x, y)
            alpha = max(
                alpha,
                alphabeta(child, depth - 1, opponent, maximizing_player, alpha, beta),
            )
            if alpha >= beta:
                break
        return alpha
    else:
        for x, y in legal_actions:
            child = board.copy()
            child = act(child, turn, x, y)
            beta = min(
                beta,
                alphabeta(
                    child, depth - 1, maximizing_player, maximizing_player, alpha, beta
                ),
            )
            if alpha >= beta:
                break
        return beta


if __name__ == "__main__":
    game = UltimateTicTacToe()
    # game.solve()
    game.play()
