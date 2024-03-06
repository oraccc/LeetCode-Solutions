class Solution:
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        ans = []
        curr_len = []
        remaining = maxWidth
        idx = 0

        def split_number(n, k):
            quotient = n // k 
            remainder = n % k 

            result = []
            for i in range(k):
                result.append(quotient)
                if remainder > 0:
                    result[i] += 1
                    remainder -= 1
            return result

        while idx < len(words):
            word = words[idx]
            if not curr_len:
                curr_len.append(word)
                remaining -= len(word)
                idx += 1
            elif remaining >= len(word)+1:
                curr_len.append(word)
                remaining -= (len(word)+1)
                idx += 1
            else:
                n_word = len(curr_len)
                if n_word == 1:
                    ans.append(curr_len[0]+" "*remaining)
                else:
                    spaces = split_number(remaining, n_word-1)
                    new_str = curr_len[0]
                    for i in range(1, n_word):
                        new_str = new_str + " "*(spaces[i-1]+1) + curr_len[i]
                    ans.append(new_str)
                curr_len = []
                remaining = maxWidth

        ans.append(" ".join(curr_len) + " "*remaining)
        return ans

