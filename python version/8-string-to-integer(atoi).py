class Solution:
    def myAtoi(self, s: str) -> int:
        if len(s) == 0: return 0
        ans = 0
        negative_flag = False
        # bndry = 2147483627 // 10 = 214748362
        int_max, int_min, bndry = 2 ** 31 - 1, -2 ** 31, 2 ** 31 // 10
        s = s.strip()
        if len(s) == 0: return 0
        if s[0] == "-": 
            negative_flag = True
            s = s[1:]
        elif s[0] == "+":
            s = s[1:]

        for char in s:
            if char >= "0" and char <= "9":
                if ans > bndry or ans == bndry and char > "7": return int_max if not negative_flag else int_min
                ans = ans*10 + int(char)
            else: break
        
        if negative_flag: return -ans
        else: return ans
