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
    
    def _expand_around_center(ans_t, ansL, l, r):
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
        ans = _expand_around_center(ans, len(ans), i, i)
            
    # even length
    for i in range(len(s) - 1):
        ans = _expand_around_center(ans, len(ans), i, i + 1)
            
    return ans

def countSubstrings(s: str) -> int:
    """Given a string s, return the number of palindromic substrings in it."""
    ans = [0]
    def _expand_around_center(l, r):
        # for some reason on leetcode, it's slower to call this function twice 
        # instead of copying the code twice
        while l >= 0 and r < len(s) and s[l] == s[r]:
            ans[0] += 1
            l -= 1
            r += 1
    
    # skip one-char palindromes
    ans[0] += len(s)
    for i in range(len(s)):
        _expand_around_center(i, i + 2)  # odd
        _expand_around_center(i, i + 1)  # even
        
    return ans[0]        
    
    # DP approach: still slower on leetcode
        # n = len(s)
        # ans = [n]
        # dp = [[False] * n for _ in range(n)]
        # for i in range(n):
        #     dp[i][i] = True
        #     if i < n - 1:
        #         if s[i] == s[i + 1]:
        #             dp[i][i + 1] = True
        #             ans[0] += 1
                    
        # for d in range(2, n):
        #     for i in range(n - d):
        #         j = i + d
        #         # print(i, j)
        #         # s[i:j] is a palindrome iff s[i+1: j-1] is a palindrome (inner str, inclusive indexes)
        #         if s[i] == s[j] and dp[i+1][j-1]:
        #             dp[i][j] = True
        #             ans[0] += 1
        #             # print('\tyey')
                    
        # return ans[0]

def numDecodings(s: str) -> int:
    code = {str(i): chr(64 + i) for i in range(1, 27)}
    if s[0] == '0' or '00' in s:
        return 0
    
    ct = [0]
    def dfs(i, curr):
        curr += s[i]
        if curr not in code:
            return
        if i == len(s) - 1:
            ct[0] += 1
            return
        
        
        if len(curr) == 2:
            dfs(i + 1, '')  # 'xx', 'y' ...
        elif len(curr) == 1:
            dfs(i + 1, curr)
            dfs(i + 1, '')
        
    dfs(0, '')
    return ct[0]
        
def numDecodingsDP(s: str) -> int:    
    code = {str(i) for i in range(1, 27)}
    mem = {}
    
    def dp(i: int) -> int:
        if i == len(s):
            return 1
        if i in mem:
            return mem[i]
        if s[i] == '0':
            return 0
        
        # single digit- start new one
        result = dp(i + 1)
        
        # if these two digits are valid, go to i + 2
        if i + 1 < len(s) and s[i: i+2] in code:
            result += dp(i + 2)
            
        mem[i] = result
        return result
    
    return dp(0)

def coinChangeSlow(coins: list[int], amount: int) -> int:
    memoi = [float('inf')] * (amount + 1)
    def dfs(x):
        if x == 0:
            return 0
        if x < 0:
            return float('inf')
        if memoi[x] != float('inf'):
            return memoi[x]
        
        min_coins = float('inf')
        for c in coins:
            res = dfs(x - c)
            min_coins = min(min_coins, 1 + res)
        memoi[x] = min_coins
        return min_coins
            
    ans = dfs(amount)
    if ans != float('inf'):
        return ans
    else:
        return -1
            
def coinChange(coins: list[int], amount: int) -> int:
    """optimized bottom up DP"""
    memoi = [float('inf')] * (amount + 1)
    memoi[0] = 0

    for n in range(1, amount + 1):
        for c in coins:
            if n - c >= 0:
                memoi[n] = min(memoi[n], 1 + memoi[n - c])

    if memoi[amount] != float('inf'):
        return memoi[amount]
    return -1

def uniquePaths(m: int, n: int) -> int:
    # bottom-up: goal is in dp[-1][-1]
    dp = [1] * n
    for i in range(1, m):
        for j in range(1, n):
            dp[j] = dp[j] + dp[j - 1]
        # print(dp)

    return dp[-1]

def maxProduct(nums: List[int]) -> int:
    global_max = nums[0]
    curr_max = nums[0]
    curr_min = nums[0]
    for n in nums[1:]:
        prev_max, prev_min = curr_max, curr_min
        curr_max = max(n * prev_max, n * prev_min, n)
        curr_min = min(n * prev_min, n * prev_max, n)
        global_max = max(curr_max, global_max)

    return global_max
        
def wordBreak(s: str, wordDict: List[str]) -> bool:
    dp = [False] * (len(s) + 1)
    dp[-1] = True  # base case
    
    for i in range(len(s) - 1, -1, -1):
        for word in wordDict:
            if i + len(word) <= len(s) and s[i: i+len(word)] == word and dp[i + len(word)]:
                dp[i] = True
                break  # out of inner loop
        
    return dp[0]

def lengthOfLIS_dfs(nums: List[int]) -> int:
    dp = [-1] * len(nums)
    def dfs(i):
        if i == len(nums) - 1:
            return 1
        if dp[i] > -1:
            return dp[i]
        
        # iterate through nums after i
        # if n > nums[i]: this is a branch. Either add it or don't
        dfs_results = []
        for j in range(i + 1, len(nums)):
            if nums[j] > nums[i]:
                if dp[j] > 0:
                    dfs_results.append(dp[j])
                    continue
                dfs_results.append(dfs(j))  # this is starting dfs for the next num (new i)
        
        print(f"idx {i}: temps = {dfs_results}")
        dp[i] = 1 + max(dfs_results, default=0)
        return 1 + max(dfs_results, default=0)
    
    return max(dfs(c) for c in range(len(nums)))

def lengthOfLIS(nums: List[int]) -> int:
    """iterative"""
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
                
    return max(dp)

import bisect
def lengthOfLIS_efficient(nums: List[int]) -> int:
    tails = [nums[0]]
    for n in nums[1:]:
        if n > tails[-1]:
            tails.append(n)
        else:
            replace_idx = bisect.bisect_left(tails, n)
            tails[replace_idx] = n
    return len(tails)

def canPartition(nums: list[int]) -> bool:
    if sum(nums) % 2 == 1:
        return False
    
    target = sum(nums) // 2
    dp = {0}
    for i in range(len(nums) - 1, -1, -1):
        temp = set()
        for x in dp:
            if x + nums[i] == target:
                return True
            temp.add(x + nums[i])
        dp = dp.union(temp)
            
    return target in dp

def longestCommonSubsequenceDfs(text1: str, text2: str) -> int:
    # SLOWER DFS version- see other for DP
    cache = {}  # (i, j): lcs
    def dfs(i, j):
        if i >= len(text1) or j >= len(text2):
            # base case: oob
            cache[(i, j)] = 0
            return 0

        if (i, j) in cache:
            return cache[(i, j)]
        
        if text1[i] == text2[j]:
            result = 1 + dfs(i + 1, j + 1)
        else:
            result = max(dfs(i, j + 1), dfs(i + 1, j))
        
        cache[(i, j)] = result
        return result
        
    return dfs(0, 0)

def longestCommonSubsequence(text1: str, text2: str) -> int:
    # DP version: faster
    arr = [[0] * (len(text2) + 1)  for _ in range(len(text1) + 1)]

    for i in range(len(text1) - 1, -1, -1):
        for j in range(len(text2) - 1, -1, -1):
            if text1[i] == text2[j]:
                # can extend subsequence by 1: look diagonally down and right
                arr[i][j] = arr[i + 1][j + 1] + 1
            else:
                # cannot extend subsequence bc letters differ:
                # take max of right and down neighbors/subproblems
                arr[i][j] = max(arr[i + 1][j], arr[i][j + 1])
    
    for _ in arr:
        print(_)
    return arr[0][0]

def coinChangeII_dfs(amount: int, coins: List[int]) -> int:
    def dfs(i, curr_amt) -> int:
        if curr_amt > amount:
            return 0  # exceeded amount; can't possibly have a correct case here
        if curr_amt == amount:
            return 1  # if we've gotten here, this is exactly 1 correct case
        if i >= len(coins):
            return 0  # out of bounds on coins; no further options, can't have a correct case
        
        ct = 0
        # continue choosing coin i
        ct += dfs(i, curr_amt + coins[i])
        
        # or don't choose it
        ct += dfs(i + 1, curr_amt)
        
        return ct
        
    return dfs(0, 0)

def coinChangeII(amount: int, coins: List[int]) -> int:
    dp = [[0] * (len(coins) + 1) for _ in range(amount)]
    dp.append([1] * (len(coins) + 1))
    
    # for _ in dp:
    #     print(_)
    # print('=' * 50)
        
    # start from bottom right; go left then up
    for curr_amt in range(amount - 1, -1, -1):
        for coin_idx in range(len(coins) - 1, -1, -1):
            # down: same coin, increase by coin amount
            down = dp[curr_amt + coins[coin_idx]][coin_idx] if curr_amt + coins[coin_idx] <= amount else 0
            # right: move on to next coin, stay at same curr_amt
            right = dp[curr_amt][coin_idx + 1]  # no need to check oob coin bc padded rightmost col before
            dp[curr_amt][coin_idx] = down + right
    # for _ in dp:
    #     print(_)
    return dp[0][0]
    
from collections import defaultdict
def findTargetSumWaysDfs(nums: List[int], target: int) -> int:
    cache = defaultdict(int)
    def dfs(i, cum_sum) -> int:
        if i == len(nums):
            if cum_sum == target:
                cache[(i, cum_sum)] = 1
                return 1
            cache[(i, cum_sum)] = 0
            return 0
        
        if (i, cum_sum) in cache:
            return cache[(i, cum_sum)]
        
        # branch: + positive number
        cum_sum += nums[i]
        add = dfs(i + 1, cum_sum)
        
        # reset
        cum_sum -= nums[i]
        # branch: + negative number
        cum_sum -= nums[i]
        sub = dfs(i + 1, cum_sum)
        
        cum_sum += nums[i]
        
        cache[(i, cum_sum)] = add + sub
        return add + sub
    
    return dfs(0, 0)

def findTargetSumWays(nums: List[int], target: int) -> int:
    n = len(nums)
    S = sum(nums)
    if abs(target) > abs(S):
        return 0
    
    # P + N = S
    # P - N = target
    # ============== +
    # 2P = S + target
    P = (S + target) // 2
    if (S + target) / 2 != P:
        return 0
    # becomes: count num of subsets of nums, that add up to P
    dp = [[0] * (P + 1) for _ in range(n + 1)]
    dp[0][0] = 1  # num subsets of 
    for _ in dp:
        print(_)
        
    for i in range(1, n + 1):
        # i is really index starting from 1, due to 0th row being base case
        for s in range(P + 1):
            include = dp[i - 1][s - nums[i - 1]] if s - nums[i - 1] >= 0 else 0
            exclude = dp[i - 1][s]
            dp[i][s] = include + exclude
            
    return dp[n][P]

print(findTargetSumWays([1,1,2,1], 3))
