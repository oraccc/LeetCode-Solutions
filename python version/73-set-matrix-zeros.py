class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        m = len(matrix[0])
        zero_row = [False for _ in range(n)]
        zero_col = [False for _ in range(m)]
        for i in range(n):
            for j in range(m):
                if matrix[i][j] == 0:
                    zero_row[i] = True
                    zero_col[j] = True
        
        for i in range(n):
            if zero_row[i]:
                for j in range(m):
                    matrix[i][j] = 0
        
        for j in range(m):
            if zero_col[j]:
                for i in range(n):
                    matrix[i][j] = 0

        return