def subsets(nums: list[int]) -> list[list[int]]:
    result = []

    subset = []
    def aux(idx):
        """idx: index of nums"""
        if idx >= len(nums):
            # reached end of this subset
            result.append(subset.copy())
            return
        
        # "left" branch: add the number to this subset
        subset.append(nums[idx])
        aux(idx + 1)
        # "right" branch: don't add the number to this subset
        subset.pop()  # undo current move
        aux(idx + 1)

    aux(0)
    return result

def subsetsIter(nums: list[int]) -> list[list[int]]:
    result = [[]]
    for n in nums:
        # either add n to each subset or don't
        len_result = len(result)
        for s in range(len_result):
            added = result[s].copy()
            added.append(n)
            result.append(added)
    return result

def combinationSum(candidates: list[int], target: int) -> list[list[int]]:
    result = []

    def aux(subset, total: int, i: int):
        if total == target:
            result.append(subset.copy())
            # print(f"success: {subset}")
            return
        if i >= len(candidates) or total > target:
            # failure
            # print(f"   fail: {subset}\ti = {i}\ttotal = {total}")
            return
        
        # decision 1: choose this number. add to total and keep it
        subset.append(candidates[i])
        aux(subset, total + candidates[i], i)
        # decision 2: exclude this number from all possible future subsets
        subset.pop()
        aux(subset, total, i + 1)

    aux([], 0, 0)
    return result


def combinationSumBackwards(candidates: list[int], target: int) -> list[list[int]]:
    result = []

    def aux(subset, target: int, i: int):
        if target == 0:
            result.append(subset.copy())
            return
        if i < 0 or target < 0:
            # failure
            return
        
        # decision 1: choose this number. add to total and keep it
        subset.append(candidates[i])
        aux(subset, target - candidates[i], i)
        # decision 2: exclude this number from all possible future subsets
        subset.pop()
        aux(subset, target, i - 1)

    aux([], target, len(candidates) - 1)
    return result


def permute(nums: list[int]) -> list[list[int]]:
    def aux(i):
        if i >= len(nums):
            return [[]]
        
        # recursive case: consider nums[i:]
        out = []
        for sublist in aux(i + 1):
            # insert nums[i] at all possible indexes
            # sublist: e.g. [2, 3] or [3, 2] or even []
            for j in range(len(sublist) + 1):
                # probably stupid...
                sublist.insert(j, nums[i])
                out.append(sublist.copy())
                sublist.pop(j)

        return out

    return aux(0)


def subsetsWithDup(nums: list[int]) -> list[list[int]]:
    result = []
    nums = sorted(nums)

    subset = []
    def aux(i):
        if i >= len(nums):
            result.append(subset.copy())
            return
        
        # choose to include nums[i] (same as subset I problem)
        subset.append(nums[i])
        aux(i + 1)
        # choose to exclude nums[i], BUT advance i till find unique number
        prev = subset.pop()
        while i < len(nums) and nums[i] == prev:
            i += 1
        aux(i)

    aux(0)

    return result


def combinationSum2(candidates: list[int], target: int) -> list[list[int]]:
    result = []
    candidates = sorted(candidates)

    def aux(subset, total: int, i: int):
        if total == target:
            result.append(subset.copy())
            return
        if i >= len(candidates) or total > target:
            # failure
            return
        
        # decision 1: choose this number. add to total and keep it
        subset.append(candidates[i])
        aux(subset, total + candidates[i], i + 1)
        # decision 2: exclude this number from all possible future subsets
        prev = subset.pop()
        while i + 1 < len(candidates) and candidates[i] == prev:
            i += 1
        aux(subset, total, i + 1)

    aux([], 0, 0)
    return result

# print(combinationSum2([1, 5, 3, 3, 2, 9], 9))


def exist(board: list[list[str]], word: str) -> bool:
    m, n = len(board), len(board[0])
    visited = set()  # set of (i, j)

    def dfs(i, j, w):
        if w == len(word):
            return True
        if (i < 0 or j < 0 or
            i == m or j == n or
            (i, j) in visited or
            board[i][j] != word[w]):
            return False
        
        visited.add((i, j))
        res = (dfs(i + 1, j, w + 1) or 
               dfs(i, j + 1, w + 1) or
               dfs(i - 1, j, w + 1) or
               dfs(i, j - 1, w + 1))
        visited.remove((i, j))
        return res
    
    for i in range(m):
        for j in range(n):
            if dfs(i, j, 0):
                return True
            
    return False


def is_palindrome(s: str):
    print("considering", s)
    return s == s[::-1]

def partition(s: str) -> list[list[str]]:
    # decide where to place the "commas"
    result = []
    
    def aux(start: int, curr_partition: list[str]):
        # place comma before s[idx], i.e. "abc": (1) is a, bc
        if start == len(s):
            result.append(curr_partition.copy())
            return
        
        for end in range(start + 1, len(s) + 1):
            substr = s[start:end]
            if is_palindrome(substr):
                curr_partition.append(substr)
                aux(end, curr_partition)
                curr_partition.pop()
        
    aux(0, [])
    
    return result


def letterCombinations(digits: str) -> list[str]:
    res = []
    digit_map = {
        '2': 'abc',
        '3': 'def',
        '4': 'ghi',
        '5': 'jkl',
        '6': 'mno',
        '7': 'qprs',
        '8': 'tuv',
        '9': 'wxyz'
    }
    
    def backtrack(i, curr):
        if len(curr) == len(digits):
            res.append(curr)
            return
        
        for c in digit_map[digits[i]]:
            backtrack(i + 1, curr + c)
            
    if digits:
        backtrack(0, '')
        
    return res

from collections import Counter
def wordSearch(board: list[list[str]], word: str) -> bool:
    m, n = len(board), len(board[0])
    if len(word) > m * n:
        return False
    
    # validate board based on char freqs
    board_freqs = Counter(c for subl in board for c in subl)
    word_char_freqs = Counter(word)
    for k in word_char_freqs:
        if word_char_freqs[k] > board_freqs[k]:
            return False
    
    dirs = ((0, 1), (1, 0), (-1, 0), (0, -1))
    
    def dfs(i: int, j: int, word_idx: int):
        if board[i][j] != word[word_idx]:
            return False
        
        if word_idx == len(word) - 1:
            return True
        
        # print(f"looking at {(i, j)}: {board[i][j]}")
        temp = board[i][j]
        board[i][j] = '#'  # replace visited set
        
        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n and board[ni][nj] != '#':
                if dfs(ni, nj, word_idx + 1):
                    return True
        # restore board's char to original
        board[i][j] = temp
            
        return False
    
    for r in range(m):
        for c in range(n):
            if board[r][c] == word[0]:
                if dfs(r, c, 0):
                    return True
            
    return False

def combine(n: int, k: int) -> list[list[int]]:
    ans = []
    subset = []
    def aux(start):
        if len(subset) == k:
            ans.append(subset[:])
            return
        rem = k - len(subset)
        # for i in range(start, n + 1):
        for i in range(start, n + 1 - rem + 1):
            # optimize by rewriting the following break condition
            # into the upper bound of the for loop
            # if i + rem > n + 1: ==> if i > n + 1 - rem:
            #     break
            subset.append(i)
            aux(i + 1)
            subset.pop()
        
    aux(1)
    return ans

def readBinaryWatch(turnedOn: int) -> list[str]:
    """
    hhhh: no leading zeros
    mmmmmm: must have leading zero if < 10
    note: hour cannot be 12
    """
    if turnedOn > 8:
        return []
    
    def get_hour(hs: list[int]) -> bool:
        t = 0
            # t = int(eval(f'0b{''.join([str(i) for i in hs])}'))
        for pow, d in enumerate(reversed(hs)):
            t += d * (2 ** pow)
        return t 
            
    def get_min(ms: list[int]) -> bool:
        t = 0
            # t = int(eval(f'0b{''.join([str(i) for i in ms])}'))
        for pow, d in enumerate(reversed(ms)):
            t += d * (2 ** pow)
        return t

    ans = []
    subset = [0] * 10
    def aux(start, count):
        # count: how many lights on so far
        if count == turnedOn:
            candidate = subset[:]
            if len(subset) < 10:
                candidate.extend([0] * (10 - count))
            # split vals and get hs/ms
            print("considering", candidate)
            hs = candidate[:4]
            ms = candidate[4:]
            h = get_hour(hs)
            m = get_min(ms)
            # print(f"h: {h}\tm: {m}")
            if h < 12 and m < 60:
                ans.append(f'{h}:{m:02d}')
            return
            
        rem = turnedOn - count
        for i in range(start, 9 - rem + 2):           
            # if not possible hour or not possible min:
            # if already [1, 1] in hs: skip making this 1 up till index 3
            if subset[:2] == [1, 1] and (i == 2 or i == 3):
                continue
            subset[i] = 1
            aux(i + 1, count + 1)
            subset[i] = 0
            # aux(i + 1, count)
            # subset.pop()
        
    aux(0, 0)
    return ans


# ans = set(readBinaryWatch(2))
# exp = set(["0:03","0:05","0:06","0:09","0:10","0:12","0:17","0:18","0:20","0:24","0:33","0:34","0:36","0:40","0:48","1:01","1:02","1:04","1:08","1:16","1:32","2:01","2:02","2:04","2:08","2:16","2:32","3:00","4:01","4:02","4:04","4:08","4:16","4:32","5:00","6:00","8:01","8:02","8:04","8:08","8:16","8:32","9:00","10:00"])
# print(ans - exp)
# print(exp - ans)
