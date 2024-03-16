class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        n = len(isConnected)
        father = {}
        self.count = n
        def find(x):
            if x not in father:
                return x
            father[x] = find(father[x])
            return father[x]
        
        def merge(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                father[root_x] = root_y
                self.count -= 1
        
        for i in range(n):
            for j in range(i, n):
                if isConnected[i][j] == 1:
                    merge(i, j)
        return self.count