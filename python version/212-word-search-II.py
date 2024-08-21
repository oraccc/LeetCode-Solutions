class TrieNode:
    def __init__(self):
        self.children = [None]*26
        self.word = ""


class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        curr = self.root
        for c in word:
            pos = ord(c)-ord('a')
            if not curr.children[pos]:
                curr.children[pos] = TrieNode()
            curr = curr.children[pos]
        curr.word = word


class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        trie = Trie()
        for word in words:
            trie.insert(word)
        
        ans = []
        n = len(board)
        m = len(board[0])
        direction = [-1, 0, 1, 0, -1]
        visited = [[False]*m for _ in range(n)]

        def backtracking(curr, i, j):
            char = board[i][j]
            curr = curr.children[ord(char)-ord('a')]

            if not curr:
                return
            if curr.word != "":
                ans.append(curr.word)
            visited[i][j] = True
            for k in range(4):
                row = i + direction[k]
                col = j + direction[k+1]
                if row >= 0 and row < n and col >= 0 and col < m and not visited[row][col]:
                    backtracking(curr, row, col)
            visited[i][j] = False


        for i in range(n):
            for j in range(m):
                backtracking(trie.root, i, j)
        
        return list(set(ans))
        