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

def longestPalindrome(s: str) -> str:
    n = len(s)
    ans = s[0]
    dp = [[False] * len(s) for _ in range(n)]
    for i in range(n):
        dp[i][i] = True
        if i < n - 1:
            if s[i] == s[i + 1]:
                dp[i][i + 1] = True
                ans = s[i: i + 2]
            
    for subl in range(3, n + 1):
        for i in range(0, n - subl + 1):
            j = i + subl - 1  # j is inclusive right index
            substr = s[i:j+1]
            print(i, j, substr)
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                ans = s[i:j + 1] if j - i + 1 > len(ans) else ans
            else:
                dp[i][j] = False
                                
    return ans

def longestPalindromeTwoPTr(s: str) -> str:
    ans = ''
    
    def _check_palindrome(ans_t, ansL, l, r):
        # step outwards until find a palindrome substring
        while l >= 0 and r < len(s) and s[l] == s[r]:
            if r - l + 1 > ansL:
                ansL = r - l + 1
                ans_t = s[l: r + 1]
            l -= 1
            r += 1
        return ans_t
    
    # odd length
    for i in range(len(s)):
        ans = _check_palindrome(ans, len(ans), i, i)
            
    # even length
    for i in range(len(s) - 1):
        ans = _check_palindrome(ans, len(ans), i, i + 1)
            
    return ans
            

print('ANSWER:', longestPalindromeTwoPTr('rac33car'))
