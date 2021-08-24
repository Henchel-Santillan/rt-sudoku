def find_empty(board):
    size = len(board)

    for row in range(size):
        for col in range(size):
            if board[row][col] == 0:
                return row, col
    return None


def valid(board, val, pos):
    size = len(board)

    for col in range(size):
        if col != pos[1] and board[pos[0]][col] == val:
            return False

    for row in range(size):
        if row != pos[0] and board[row][pos[1]] == val:
            return False

    # get upper vertex of 3X3 submatrix
    uv_x, uv_y = 3 * int(pos[0] / 3), 3 * int(pos[1] / 3)

    for row in range(uv_x, uv_x + 3):
        for col in range(uv_y, uv_y + 3):
            if (row, col) != pos and board[row][col] == val:
                return False

    return True


def solve(board):
    empty = find_empty(board)

    if empty is None:
        return True
    else:
        row, col = empty

    for val in range(1, len(board) + 1):
        if valid(board, val, (row, col)):
            board[row][col] = val

            if solve(board):
                return True

            board[row][col] = 0

    return False


def draw(board):
    size = len(board)

    for row in range(size):
        for col in range(size):
            print(str(board[row][col]) + " ", end="")
        print()
