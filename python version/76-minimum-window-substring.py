class Solution:
    def minWindow(self, s: str, t: str) -> str:
        n = len(t)
        unseen_flag = [False]*128
        unseen_char = [0]*128

        for i in range(n):
            unseen_flag[ord(t[i])] = True
            unseen_char[ord(t[i])] += 1
        
        left = 0
        right = 0
        count = 0
        min_start = 0
        min_len = len(s)+1

        while right < len(s):
            char = s[right]
            if unseen_flag[ord(char)] == True:
                unseen_char[ord(char)] -= 1
                if unseen_char[ord(char)] >= 0:
                    count += 1
            while count == n:
                if right-left+1 < min_len:
                    min_start = left
                    min_len = right-left+1
                if unseen_flag[ord(s[left])]:
                    unseen_char[ord(s[left])] += 1
                    if unseen_char[ord(s[left])] > 0:
                        count -= 1
                left += 1

            right += 1
        if min_len > len(s):
            return ""
        return s[min_start:min_start+min_len]