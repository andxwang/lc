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

b = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
for _ in b:
    print(_)
print(wordSearch(b, 'ABCB'))
