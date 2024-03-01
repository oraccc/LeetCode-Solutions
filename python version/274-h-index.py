class Solution:
    def hIndex(self, citations: List[int]) -> int:
        citations = sorted(citations, reverse=True)
        n = len(citations)
        h = 0
        for i in range(n):
            if citations[i] > h:
                h += 1
            else:
                break
        return h