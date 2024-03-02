class Solution:
    def reverseWords(self, s: str) -> str:
        word_list = []
        word = ""
        for char in s:
            if char != " ":
                word += char
            elif word != "":
                word_list.append(word)
                word = ""
        if word != "":
            word_list.append(word)

        word_list = word_list[::-1]
        return " ".join(word_list)