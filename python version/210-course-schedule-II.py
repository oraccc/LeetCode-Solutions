class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        in_degree = [0 for _ in range(numCourses)]
        out_graph = {}
        ans = []
        for pair in prerequisites:
            in_degree[pair[0]] += 1
            if pair[1] not in out_graph:
                out_graph[pair[1]] = [pair[0]]
            else:
                out_graph[pair[1]].append(pair[0])
        queue = []

        for i in range(numCourses):
            if in_degree[i] == 0:
                queue.append(i)

        while queue:
            curr = queue.pop(0)
            ans.append(curr)
            if curr in out_graph:
                for each in out_graph[curr]:
                    in_degree[each] -= 1
                    if in_degree[each] == 0:
                        queue.append(each)
        
        for i in range(numCourses):
            if in_degree[i] != 0:
                return []
        return ans