class Solution:
    def gameOfLife(self, board: List[List[int]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        direction_x = [0, 0, 1, 1, 1, -1, -1, -1]
        direction_y = [1, -1, 0, 1, -1, 0, 1, -1 ]
        n = len(board)
        m = len(board[0])
        for i in range(n):
            for j in range(m):
                live_count = 0
                for k in range(8):
                    row = i + direction_x[k]
                    col = j + direction_y[k]
                    if row < 0 or row >= n or col < 0 or col >= m:
                        continue
                    else:
                        live_count += (board[row][col] & 1)
                if board[i][j] & 1:
                    if live_count == 2 or live_count == 3:
                        board[i][j] = board[i][j] | (1 << 1)
                    else:
                        board[i][j] = board[i][j] | (0 << 1)
                else:
                    if live_count == 3:
                        board[i][j] = board[i][j] | (1 << 1)
                    else:
                        board[i][j] = board[i][j] | (0 << 1)

        for i in range(n):
            for j in range(m):
                board[i][j] = board[i][j] >> 1
        return