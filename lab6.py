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

def rob(nums: List[int]) -> int:
    n = len(nums)
    if n == 1:
        return nums[0]
    
    memoiz = [0] * n
    memoiz[-1] = nums[-1]
    memoiz[-2] = nums[-2]
    
    for i in range(n - 3, -1, -1):
        # pick the most between values at i + 2 and i + 3, if in bounds
        if i + 3 < n:
            memoiz[i] = nums[i] + max(memoiz[i + 2], memoiz[i + 3])
        else:
            memoiz[i] = nums[i] + memoiz[i + 2]
        
    return max(memoiz[0], memoiz[1])

print(rob([2,7,9,3,1]))
