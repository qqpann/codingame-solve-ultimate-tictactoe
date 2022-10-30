class UltimateTicTacToe:
    def __init__(self):
        self.board = [[0] * 9 for _ in range(9)]
        self.done = False
        self.turn = 1

    def __str__(self):
        res = ""
        for row in self.board:
            res += "".join(map(str, row)) + "\n"
        return res

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


if __name__ == "__main__":
    game = UltimateTicTacToe()
    game.play()
