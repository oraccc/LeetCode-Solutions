class Solution:
    def simplifyPath(self, path: str) -> str:
        names = path.split("/")
        stack = []
        for each in names:
            if each not in ["..", ".", ""]:
                stack.append(each)
            elif each == ".." and stack:
                stack.pop()
        
        new_path = "/".join(stack)
        
        return "/" + new_path