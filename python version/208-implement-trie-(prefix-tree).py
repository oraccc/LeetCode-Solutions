class TrieNode:
    def __init__(self, is_word=False):
        self.is_word = is_word
        self.child = [None for _ in range(26)]

class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        curr = self.root
        for i in range(len(word)):
            pos = ord(word[i])-ord("a")
            if not curr.child[pos]:
                curr.child[pos] = TrieNode()
            curr = curr.child[pos]
        curr.is_word = True

    def search(self, word: str) -> bool:
        curr = self.root
        for i in range(len(word)):
            pos = ord(word[i])-ord("a")
            if not curr.child[pos]:
                return False
            else:
                curr = curr.child[pos]
        return curr.is_word == True

    def startsWith(self, prefix: str) -> bool:
        curr = self.root
        for i in range(len(prefix)):
            pos = ord(prefix[i])-ord("a")
            if not curr.child[pos]:
                return False
            else:
                curr = curr.child[pos]
        return True



# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)