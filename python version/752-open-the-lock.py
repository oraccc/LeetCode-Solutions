class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:

        if "0000" in deadends:
            return -1
        if target == "0000":
            return 0
        
        def change(s):
            new_s = []
            for i in range(4):
                digit = int(s[i])
                for j in [-1, 1]:
                    new_d = (digit+j) % 10
                    new_s.append(s[:i] + str(new_d) + s[i+1:])
            return new_s

        steps = 0
        queue = ["0000"]
        deadends = set(deadends)
        visited = set()
        visited.add("0000")

        while queue:
            steps += 1
            n = len(queue)
            for _ in range(n):
                curr = queue.pop(0)
                nxt = change(curr)
                for each in nxt:
                    if each == target:
                        return steps
                    if each in visited or each in deadends:
                        continue
                    visited.add(each)
                    queue.append(each)
        
        return -1


