class Solution:
    def snakesAndLadders(self, board: List[List[int]]) -> int:
        n = len(board)
        grid = []
        flag = True
        for i in range(n-1, -1, -1):
            if flag:
                for j in range(0, n, 1):
                    grid.append(board[i][j])
            else:
                for j in range(n-1, -1, -1):
                    grid.append(board[i][j])
            flag = not flag

        queue = [1]
        step = 0
        visited = set([1])
        while queue:
            step += 1
            size = len(queue)
            for i in range(size):
                prev_pos = queue.pop(0)
                for i in range(1, 7):
                    if prev_pos + i > n**2: break
                    next_pos = prev_pos + i
                    if grid[next_pos-1] != -1:
                        next_pos = grid[next_pos-1]
                    if next_pos == n**2: return step
                    if next_pos not in visited:
                        visited.add(next_pos)
                        queue.append(next_pos)
        return -1
                    
                