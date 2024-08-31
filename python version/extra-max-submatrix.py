class Solution:
    def getMaxMatrix(self, matrix: List[List[int]]) -> List[int]:
        ans = []
        n = len(matrix)
        m = len(matrix[0])
        pre_sum = [[0]*(m+1) for _ in range(n+1)]

        gloal_max = float("-inf")
        for i in range(1, n+1):
            for j in range(1, m+1):
                pre_sum[i][j] = matrix[i-1][j-1]+pre_sum[i-1][j]+pre_sum[i][j-1]-pre_sum[i-1][j-1]

        for top in range(n):
            for bottom in range(top,n):
                local_max = 0
                left = 0
                for right in range(m):
                    local_max = pre_sum[bottom+1][right+1]-pre_sum[bottom+1][left]-pre_sum[top][right+1]+pre_sum[top][left]
                    if local_max > gloal_max:
                        gloal_max = local_max
                        ans = [top, left, bottom, right]
                    if local_max < 0:
                        left = right+1
        return ans
