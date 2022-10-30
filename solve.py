import math
import sys

while True:
    opponent_row, opponent_col = map(int, input().split())
    valid_action_count = int(input())
    valid_actions = []
    for i in range(valid_action_count):
        row, col = [int(j) for j in input().split()]
        valid_actions.append((row, col))

    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr, flush=True)

    act = {"x": 1, "y": 1}

    print(f"{act['x']} {act['y']}")
