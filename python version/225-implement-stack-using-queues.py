class MyStack:

    def __init__(self):
        self.q1 = []
        self.q2 = []

    def push(self, x: int) -> None:
        self.q1.append(x)

    def pop(self) -> int:
        while len(self.q1) > 1:
            self.q2.append(self.q1.pop(0))
        tmp = self.q1.pop(0)
        self.q1, self.q2 = self.q2, self.q1
        return tmp

    def top(self) -> int:
        while len(self.q1) > 1:
            self.q2.append(self.q1.pop(0))
        tmp = self.q1.pop(0)
        self.q2.append(tmp)
        self.q1, self.q2 = self.q2, self.q1
        return tmp

    def empty(self) -> bool:
        return len(self.q1) == 0 and len(self.q2) == 0 


# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()