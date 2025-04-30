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
    
    memoiz = [0] * (n + 2)
    for i in range(n - 1, -1, -1):
        # pick the most between this house + two down, or just one down
        memoiz[i] = max(nums[i] + memoiz[i + 2], memoiz[i + 1])
        
    return memoiz[0]

def robO1Space(nums: List[int]) -> int:
    prev1, prev2 = 0, 0  # dp[i+1], dp[i+2]
    for x in reversed(nums):
        curr = max(x + prev2, prev1)
        prev2, prev1 = prev1, curr
    return prev1

def robII(nums: List[int]):
    n = len(nums)
    if n == 1:
        return nums[0]
    
    def aux(sublist: List[int]):
        m = len(sublist)
        if m == 0:
            return 0
        if m == 1:
            return sublist[0]
        if m == 2:
            return max(sublist)
        memoiz = [0] * (m + 2)
        for i in range(m - 1, -1, -1):
            # pick the most between this house + two down, or just one down
            memoiz[i] = max(nums[i] + memoiz[i + 2], memoiz[i + 1])
            
        return memoiz[0]
    
    results = [aux(nums[:n-1]), aux(nums[1:])]
    return max(results)

print(robII([2, 3, 2]))