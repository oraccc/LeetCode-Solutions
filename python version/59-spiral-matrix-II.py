class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        matrix = [[0]*n for _ in range(n)]
        count = 1

        top = 0
        bottom = n-1
        left = 0
        right = n-1

        while top <= bottom and left <= right:
            for i in range(left, right+1):
                matrix[top][i] = count
                count += 1
            for i in range(top+1, bottom+1):
                matrix[i][right] = count
                count += 1
            
            if top < bottom and left < right:
                for i in range(right-1, left-1, -1):
                    matrix[bottom][i] = count
                    count += 1
                for i in range(bottom-1, top, -1):
                    matrix[i][left] = count
                    count += 1

            top += 1
            bottom -= 1
            left += 1
            right -= 1

        return matrix
        
