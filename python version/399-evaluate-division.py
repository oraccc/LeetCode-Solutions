class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        graph = {}
        for (s, e), v in zip(equations, values):
            if s not in graph:
                graph[s] = {}
            if e not in graph:
                graph[e] = {}
            graph[s][e] = v
            graph[e][s] = 1/v
        
        # start bfs
        queue = []
        ans = [-1.0 for _ in range(len(queries))]

        for i in range(len(queries)):
            qs, qe = queries[i]
            if qs not in graph or qe not in graph: continue
            queue = [(qs, 1)]
            visited = set([qs])
            found = False
            while queue and not found:
                point, mul = queue.pop(0)
                for neighbor in graph[point].keys():
                    if neighbor == qe:
                        ans[i] = mul * graph[point][neighbor]
                        found = True
                        break
                    elif neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, mul * graph[point][neighbor]))
        
        return ans


            