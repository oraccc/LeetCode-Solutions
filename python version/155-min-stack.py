class MinStack:

    def __init__(self):
        self.s = []
        self.min_s = []

    def push(self, val: int) -> None:
        if not self.min_s or self.min_s[-1] >= val:
            self.min_s.append(val)
        self.s.append(val)

    def pop(self) -> None:
        if self.min_s and self.min_s[-1] == self.s[-1]:
            self.min_s.pop()
        self.s.pop()

    def top(self) -> int:
        return self.s[-1]

    def getMin(self) -> int:
        return self.min_s[-1]


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()