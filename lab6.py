from typing import List

def climbStairs(n: int) -> int:
    if n == 1: return 1
    if n == 2: return 2
    prev, curr = 1, 2  # answers for [n = 1, n = 2]
    for i in range(3, n + 1):
        prev, curr = curr, prev + curr
        
    return curr

def minCostClimbingStairs(cost: List[int]) -> int:
    cuml = [0] * len(cost)
    cuml[-1], cuml[-2] = cost[-1], cost[-2]
    
    for i in range(len(cost) - 3, -1, -1):  # start at third-to-last
        cuml[i] = cost[i] + min(cuml[i + 1], cuml[i + 2])
    
    return min(cuml[0], cuml[1])

def minCostClimbingStairs2(cost: List[int]) -> int:
    a, b = cost[-1], cost[-2]
    for i in range(len(cost) - 3, -1, -1):
        a, b = b, cost[i] + min(a, b)
        # or even do in-place:
        # cost[i] = cost[i] + min(cost[i+1], cost[i+2])
    
    return min(a, b)


print(minCostClimbingStairs2([7,8,2,2,3,4,23,2,9,12,26]))
