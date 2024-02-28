class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        row = len(matrix)
        col = len(matrix[0])

        left = 0
        right = row
        while(left < right):
            mid = left + (right-left)//2
            if matrix[mid][col-1] >= target:
                right = mid
            else:
                left = mid+1

        m = left
        if m == row: return False
        left = 0
        right = col
        while(left < right):
            mid = left + (right-left)//2
            if matrix[m][mid] >= target:
                right = mid
            else:
                left = mid+1
        n = left
        if n == col: return False
        return matrix[m][n] == target