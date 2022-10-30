from typing import List


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

    def sub_board(self, n: int):
        sx, sy = n % 3 * 3, n // 3 * 3
        sub = []
        for x in range(sx, sx + 3):
            sub.append([self.board[x][y] for y in range(sy, sy + 3)])
        return sub

    def flip_turn(self):
        self.turn = 2 if self.turn == 1 else 1

    def act(self, player: int, x: int, y: int):
        if self.board[x][y] != 0:
            # Illegal act
            self.done = True
            return
        self.board[x][y] = player
        self.flip_turn()

    def legal_actions(self):
        legal = []
        for x in range(9):
            for y in range(9):
                if self.board[x][y] == 0:
                    legal.append((x, y))
        return legal

    def play(self):
        while not self.done:
            print(self)
            # Replace with AI here.
            x, y = map(int, input("Move>").split())
            self.act(self.turn, x, y)
            for i in range(9):
                print(str_board(self.sub_board(i)))


if __name__ == "__main__":
    game = UltimateTicTacToe()
    game.play()
