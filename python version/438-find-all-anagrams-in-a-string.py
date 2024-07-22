class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:

        def check_same(p_count, curr_count):
            for i in range(len(p_count)):
                if p_count[i] != curr_count[i]:
                    return False
            return True
        m = len(s)
        n = len(p)
        ans = []
        p_count = [0]*26
        curr_count = [0]*26
        if n > m:
            return ans
        for i in range(n):
            p_count[ord(p[i])-ord('a')] += 1

        left = right = 0
        for right in range(m):
            if right < n-1:
                curr_count[ord(s[right])-ord('a')] += 1
            else:
                curr_count[ord(s[right])-ord('a')] += 1
                if check_same(p_count, curr_count):
                    ans.append(left)
                curr_count[ord(s[left])-ord('a')] -= 1
                left += 1
                right += 1
        return ans
