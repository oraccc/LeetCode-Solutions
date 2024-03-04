class RandomizedSet:

    def __init__(self):
        self.set = []
        self.map = {}

    def insert(self, val: int) -> bool:
        if val in self.map:
            return False
        else:
            self.set.append(val)
            self.map[val] = len(self.set)-1
            return True

    def remove(self, val: int) -> bool:
        if val not in self.map:
            return False
        pos = self.map[val]
        self.set[pos] = self.set[-1]
        self.map[self.set[-1]] = pos
        self.set.pop()
        self.map.pop(val)
        return True

    def getRandom(self) -> int:
        return self.set[random.randint(0, len(self.set)-1)]


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()