def empty(board):
    for row in range(len(board)):
        for col in range(len(board[0])):
            if board[row][col] == 0:
                return row, col

    return None


def valid(board, val, pos):
    size = len(board)

    # row check
    for col in range(size):
        if col != pos[1] and board[pos[0]][col] == val:
            return False

    # column check
    for row in range(size):
        if row != pos[0] and board[row][pos[1]] == val:
            return False

    # box check: get upper vertex of 3x3 submatrix
    uv_x, uv_y = (pos[0] / 3) * 3, (pos[1] / 3) * 3

    for row in range(uv_x, uv_x + 3):
        for col in range(uv_y, uv_y + 3):
            if (row, col) != pos and board[row][col] == val:
                return False

    return True


def solve(board):
    find_empty = empty(board)

    if find_empty is None:
        return True
    else:
        row, col = find_empty

    for val in range(1, len(board) + 1):
        if valid(board, val, (row, col)):
            board[row][col] = val

            if solve(board):
                return True

            board[row][col] = 0

    return False
