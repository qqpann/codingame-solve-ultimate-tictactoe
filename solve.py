import math
import random
import sys
import time
from copy import deepcopy
from typing import List

DEPTH = 1


def iter_3x3():
    for x in range(3):
        for y in range(3):
            yield (x, y)


def tic_tac_toe_winner(board: List[List[int]]):
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


def str_board(board: List[List[int]]):
    res = ""
    for row in board:
        res += "".join(map(str, row)) + "\n"
    return res


class UltimateTicTacToe:
    def __init__(self):
        self.board = [[0] * 9 for _ in range(9)]
        self.done = False
        self.turn = 1

    def __str__(self):
        return str_board(self.board)

    def sub_board(self, sx: int, sy: int):
        sx, sy = sx * 3, sy * 3
        sub = []
        for x in range(sx, sx + 3):
            sub.append([self.board[x][y] for y in range(sy, sy + 3)])
        return sub

    def meta_board(self):
        meta = []
        for x in range(3):
            meta.append([tic_tac_toe_winner(self.sub_board(x, y)) for y in range(3)])
        return meta

    def flip_turn(self):
        self.turn = 2 if self.turn == 1 else 1

    def act(self, player: int, x: int, y: int):
        if self.board[x][y] != 0:
            # Illegal act
            self.done = True
            return self
        self.board[x][y] = player
        self.flip_turn()
        return self

    def legal_actions(self):
        legal = []
        for x in range(9):
            for y in range(9):
                if self.board[x][y] == 0:
                    legal.append((x, y))
        return legal

    def play(self):
        while not self.done:
            # Replace with AI here.
            print(self)
            start = time.perf_counter()
            if self.turn == 1:
                x, y = map(int, input("Move>").split())
            else:
                # legal_actions = self.legal_actions()
                # action_rewards = [
                #     minimax(
                #         deepcopy(self).act(self.turn, x, y),
                #         DEPTH,
                #         self.turn,
                #     )
                #     for x, y in legal_actions
                # ]
                # # print(legal_actions)
                # # print(action_rewards)
                # max_reward = max(action_rewards)
                # max_actions = [
                #     a for a, r in zip(legal_actions, action_rewards) if r == max_reward
                # ]
                # x, y = random.choice(max_actions)
                x, y = random.choice(self.legal_actions())
            end = time.perf_counter()
            print("Time:", end - start)

            self.act(self.turn, x, y)
            # for x, y in iter_3x3():
            #     print(str_board(self.sub_board(x, y)))
            # print("meta board")

            # print(str_board(self.meta_board()))
            # print(tic_tac_toe_winner(self.meta_board()))

    def solve(self):
        while True:
            opponent_row, opponent_col = map(int, input().split())
            valid_action_count = int(input())
            valid_actions = []
            for i in range(valid_action_count):
                row, col = [int(j) for j in input().split()]
                valid_actions.append((row, col))

            # Write an action using print
            # To debug: print("Debug messages...", file=sys.stderr, flush=True)
            if opponent_col == -1 and opponent_row == -1:
                pass
            else:
                self.act(2, opponent_col, opponent_row)
            act = random.choice(self.legal_actions())
            print(f"{act[0]} {act[1]}")


def score(game: UltimateTicTacToe, player: int):
    assert player in [1, 2]
    meta_win = tic_tac_toe_winner(game.meta_board())
    if meta_win != 0:
        return 10 if meta_win == player else -10

    sub_wins = [tic_tac_toe_winner(game.sub_board(x, y)) for x, y in iter_3x3()]
    p1 = sub_wins.count(1)
    p2 = sub_wins.count(2)
    return p1 - p2 if player == 1 else p2 - p1


def minimax(game: UltimateTicTacToe, depth: int, maximizing_player: int) -> float:
    """
    MiniMax function
    """
    # Ref: https://en.wikipedia.org/wiki/Minimax
    return alphabeta(game, depth, maximizing_player, -float("inf"), float("inf"))


def alphabeta(
    game: UltimateTicTacToe,
    depth: int,
    maximizing_player: int,
    alpha: float,
    beta: float,
) -> float:
    """
    MiniMax with alpha-beta pruning
    """
    # Ref: https://en.wikipedia.org/wiki/Alpha–beta_pruning
    if depth == 0 or game.done or len(game.legal_actions()) == 0:
        return score(game, maximizing_player)

    legal_actions = game.legal_actions()
    if game.turn == maximizing_player:
        for x, y in legal_actions:
            child = deepcopy(game)
            child.act(game.turn, x, y)
            alpha = max(
                alpha, alphabeta(child, depth - 1, maximizing_player, alpha, beta)
            )
            if alpha >= beta:
                break
        return alpha
    else:
        for x, y in legal_actions:
            child = deepcopy(game)
            child.act(game.turn, x, y)
            beta = min(
                beta, alphabeta(child, depth - 1, maximizing_player, alpha, beta)
            )
            if alpha >= beta:
                break
        return beta


if __name__ == "__main__":
    game = UltimateTicTacToe()
    game.solve()
