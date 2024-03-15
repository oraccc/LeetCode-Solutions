class Solution:
    def compareVersion(self, version1: str, version2: str) -> int:
        list1 = version1.split(".")
        list2 = version2.split(".")
        for i in range(max(len(list1), len(list2))):
            num1 = int(list1[i]) if i < len(list1) else 0
            num2 = int(list2[i]) if i < len(list2) else 0
            if num1 > num2: return 1
            elif num1 < num2: return -1
        
        return 0