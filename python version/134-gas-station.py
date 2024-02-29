class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        n = len(gas)
        i = 0
        
        while i < n:
            count = 0
            curr_gas = 0
            curr_pos = i
            while count < n:
                curr_gas += gas[curr_pos]
                curr_gas -= cost[curr_pos]
                if curr_gas < 0:
                    break
                else:
                    count += 1
                    curr_pos = (curr_pos + 1) % n
            if count == n:
                return i
            else:
                i = i + count + 1
        
        return -1