class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        top, left, bottom, right = 0, 0, len(matrix)-1, len(matrix[0])-1
        path = []
        while top <= bottom and left <= right:
            for i in range(left, right+1):
                path.append(matrix[top][i])
            for i in range(top+1, bottom+1):
                path.append(matrix[i][right])
            if top < bottom and left < right:
                for i in range(right-1, left-1, -1):
                    path.append(matrix[bottom][i])
                for i in range(bottom-1, top, -1):
                    path.append(matrix[i][left])
            
            top, left, bottom, right = top+1, left+1, bottom-1, right-1

        return path