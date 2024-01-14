class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        rows = [[False for _ in range(9)] for _ in range(9)]
        cols = [[False for _ in range(9)] for _ in range(9)]
        grids = [[False for _ in range(9)] for _ in range(9)]

        for i in range(9):
            for j in range(9):
                if board[i][j] != ".":
                    num = int(board[i][j]) - 1
                    if rows[i][num] or cols[j][num] or grids[(i//3)*3 + (j//3)][num]:
                        return False
                    rows[i][num] = True
                    cols[j][num] = True
                    grids[(i//3)*3 + (j//3)][num] = True

        return True