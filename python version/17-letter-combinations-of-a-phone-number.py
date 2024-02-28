class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        letter_map = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl", "6": "mno", "7":"pqrs", "8":"tuv", "9":"wxyz"}
        n = len(digits)
        if n == 0: return []
        
        ans = []
        string = []

        def backtracking(index: int):
            if index == n:
                ans.append("".join(string))
            else:
                for each in letter_map[digits[index]]:
                    string.append(each)
                    backtracking(index+1)
                    string.pop()
        
        backtracking(0)
        return ans
        