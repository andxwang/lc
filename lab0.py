from typing import List


def twoSum(nums: list[int], target: int) -> list[int]:
    hmap = {}
    for i, n in enumerate(nums):
        hmap[n] = i
    for i, n in enumerate(nums):
        if target - n in hmap:
            complement_idx = hmap[target - n]  # save so don't access twice
            if complement_idx != i:
                return [i, complement_idx]

from collections import defaultdict
def char_counter(s: str):
    counts = [0] * 26
    for c in s:
        counts[ord(c) - ord('a')] += 1
    return tuple(counts)

def groupAnagrams(strs: list[str]) -> list[list[str]]:
    # use (26,) tuple of letters as key for hash map
    ana_dict = defaultdict(list)
    for s in strs:
        ana_dict[char_counter(s)].append(s)
    return list(ana_dict.values())

def findMin(nums):
    p = 0
    q = len(nums) - 1
    while p < q:
        m = p + (q - p) // 2  # lower
        if nums[m] > nums[q]:
            # pivot must be between [m + 1, q]
            # print("\tright")
            p = m + 1

        else:
            # pivot must be between [p, m]
            # print("\tleft")
            q = m

        # print(nums[p:q+1])
    
    # print(p, q)
    return nums[p]

def valid(piles, h, k):

    hrs = 0  # number of hours to eat all the bananas
    for p in piles:
        hrs += (p // k)
        if p % k != 0:
            hrs += 1
    return hrs <= h

def minEatingSpeed(piles, h):
    l = 1
    r = max(piles)
    while l < r:
        m = l + (r - l) // 2
        # check if m is a valid k
        if valid(piles, h, m):
            # shift to left half, include m
            r = m
        else:
            # m can't be the answer, so exclude it
            l = m + 1

    # print(l, r)

    return l

def search(nums, target):
    p = 0
    q = len(nums) - 1
    while p < q:
        m = p + (q - p) // 2  # lower
        if target == nums[m]:
            return m
        if nums[m] > nums[q]:
            # pivot must be between [m + 1, q]
            print("\tright")
            p = m + 1

        else:
            # pivot must be between [p, m]
            print("\tleft")
            q = m

        print(nums[p:q+1])

    arr = nums[p:] + nums[:p]
    print(arr)
    l = 0
    r = len(arr) - 1
    while l <= r:
        m = l + (r - l) // 2
        if arr[m] == target:
            return (m + p) % len(nums)
        elif arr[m] < target:
            l = m + 1
        else:
            r = m - 1
    
    # print(p, q)
    return -1

from collections import defaultdict
class TimeMap:

    def __init__(self):
        self.timemap = defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.timemap[key].append((timestamp, value))

    def get(self, key: str, timestamp: int) -> str:
        if key not in self.timemap:
            return ""
        # need to find the largest prev time that is <= timestamp
        values = self.timemap[key]
        if timestamp < values[0][0]:
            return ""
        elif timestamp > values[-1][0]:
            return values[-1][1]
        
        l = 0
        r = len(values) - 1
        while l < r:
            m = l + (r - l + 1) // 2  # m is UPPER
            if timestamp == values[m][0]:
                return values[m][1]
            elif timestamp < values[m][0]:
                # move right ptr, exclude
                r = m - 1
            else:
                # move left ptr, include
                l = m
        return values[l][1]

def lengthOfLongestSubstring(s):
    if len(s) == 0:
        return 0
    if len(s) == 1:
        return 1
    left = 0
    right = 1
    longest = 1
    counter = set(s[0])
    while right < len(s):
        while s[right] in counter:
            counter.remove(s[left])
            left += 1
        counter.add(s[right])
        longest = max(longest, len(counter))
        right += 1

    return longest


from collections import Counter

def num_replacements(counter):
    """Calculate the smallest # of char replacements to make a string all the same char."""
    most = max(counter.values())
    return sum(counter.values()) - most

def characterReplacement(s, k):
    left = 0
    longest = 1
    counter = {s[0]: 1}
    for right in range(1, len(s)):
        counter[s[right]] = counter.get(s[right], 0) + 1
        while num_replacements(counter) > k:
            counter[s[left]] -= 1
            left += 1
        longest = max(longest, right - left + 1)

    return longest

def moveZeroes(nums: list[int]) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    zeros_ct = 0
    for i in range(len(nums) - 1, -1, -1):
        if nums[i] == 0:
            nums.pop(i)
            zeros_ct += 1
            
    for n in range(zeros_ct):
        nums.append(0)
        
def moveZeroesEfficient(nums: list[int]) -> None:
    left = 0
    for right in range(len(nums)):
        if nums[right] != 0:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
        
def plusOne(digits: list[int]) -> list[int]:
    carry = 1
    for i in range(len(digits) - 1, -1, -1):
        digits[i] += carry
        if digits[i] == 10:
            digits[i] = 0
            carry = 1
        else:
            carry = 0
    
    if carry:
        digits.insert(0, 1)
    
    return digits

def removeDuplicates(nums: list[int]) -> int:
    if len(nums) == 1 or len(nums) == 2:
        return len(nums)
    
    k = 2  # next valid idx
    for i in range(2, len(nums)):
        print(nums, end='\t')
        if nums[i] == nums[k - 2]:
            # triple
            print(f"k is {k}; i is {i}; skip")
            pass
        else:
            print(f"k is {k}; i is {i}; copy")
            nums[k] = nums[i]
            k += 1
    return k

def topKFrequent(nums: list[int], k: int) -> list[int]:
    counts = Counter(nums)

    # map count to list of nums: all nums in that list appear count times
    values = [[] for i in range(0, len(nums)+1)]  # index represents count
    for n, c in counts.items():
        values[c].append(n)

    # print(values)
    res = []
    for i in range(len(values)-1, 0, -1):
        if len(values[i]) != 0:
            # this only works because the solution is guaranteed
            res.extend(values[i])
            k -= len(values[i])
        if k == 0:
            return res

def lastStoneWeight(stones: list[int]) -> int:

    while len(stones) > 1:
        stones.sort(reverse=False)
        if stones[-1] == stones[-2]:
            stones.pop()
            stones.pop()
        else:
            if stones[-1] >= stones[-2]:
                larger = stones.pop()
                stones[-1] = larger - stones[-1]
            else:
                smaller = stones.pop()
                stones[-1] -= smaller
    if stones:
        return stones[0]
    return 0

from collections import Counter
def distinctAverages(nums: list[int]) -> int:
    nums.sort()
    print(nums)
    l = 0
    r = len(nums) - 1
    avgs = set()
    while l < r:
        print(nums[l], nums[r])
        avgs.add(int((nums[r] + nums[l]) / 2 * 10))  # could even do * 5
        l += 1
        r -= 1
    
    print(avgs)
    return len(avgs)

def addBinary(a: str, b: str) -> str:
    i = 0
    carry = 0
    res = []
    while i < min(len(a), len(b)):
        s = int(a[-i - 1]) + int(b[-i - 1]) + carry

        if s > 1:
            carry = 1
            s -= 2

        else:
            carry = 0
        res.append(str(s))
        i += 1

    longer = a if len(a) > len(b) else b
    while i < len(longer):
        s = carry + int(longer[-i - 1])
        if s > 1:
            carry = 1
            s -= 2
        else:
            carry = 0
        res.append(str(s))
        i += 1

    if carry:
        res.append('1')

    return ''.join(reversed(res))

def reverseVowels(s: str) -> str:
    _vowels = set("aeiouAEIOU")
    targets = []
    for c in reversed(s):
        if c in _vowels:
            targets.append(c)
    
    result = []
    v = 0
    for i, c in enumerate(s):
        if s[i] in _vowels:
            result.append(targets[v])
            v += 1
        else:
            result.append(c)
            
    return ''.join(result)
    
def convertTime(current: str, correct: str) -> int:
    current = [int(s) for s in current.split(':')]
    correct = [int(s) for s in correct.split(':')]
    units = [60, 15, 5, 1]
    diff = correct[0] * 60 + correct[1] - (current[0] * 60 + current[1])
    
    total = 0
    for unit in units:
        # amt = diff // unit
        # diff = diff % unit
        amt, diff = divmod(diff, unit)
        total += amt
    
    return total

def diagonalPrime(nums: list[list[int]]) -> int:
    def _is_prime(n):
        for i in range(2, n // 2):
            if n % i == 0:
                return False
        return True
    
    curr = 0
    h = len(nums)
    for i in range(h):
        x = nums[i][i]
        if x > curr and _is_prime(x):
            curr = x
        x = nums[i][h - i - 1]
        if x > curr and _is_prime(x):
            curr = x

    return curr

def canThreePartsEqualSum(arr: list[int]) -> bool:
    """Constraints:
    3 <= arr.length <= 5 * 104
    -104 <= arr[i] <= 104
    """
    s = sum(arr)
    if s % 3 != 0:
        return False
    
    goal_sum = s // 3
    left_sum = 0
    for l in range(len(arr) - 2):
        left_sum += arr[l]
        if left_sum == goal_sum:
            mid_sum = 0
            for r in range(l + 1, len(arr) - 1):
                mid_sum += arr[r]
                if mid_sum == goal_sum:
                    if sum(arr[r + 1:]) == goal_sum:
                        return True
            
    return False
    
def merge(nums1: list[int], m: int, nums2: list[int], n: int) -> None:
    """
    Do not return anything, modify nums1 in-place instead.
    """
    i = m - 1
    j = n - 1
    a = m + n - 1
    
    while j >= 0 and i >= 0:
        if nums2[j] > nums1[i]:
            nums1[a] = nums2[j]
            j -= 1
        else:
            nums1[a] = nums1[i]
            i -= 1
        a -= 1
    
    while a >= 0:
        if i >= 0:
            nums1[a] = nums1[i]
            i -= 1
        if j >= 0:
            nums1[a] = nums2[j]
            j -= 1
        a -= 1

l1 = [5, 6, 7, 8, 0, 0, 0]
l2 = [1, 2, 3]
# l2 = [5, 6, 7, 8]

def leftRightDifference(nums: List[int]) -> List[int]:
    lsum = 0
    rsum = sum(nums)
    ans = []

    for n in nums:
        lsum += n
        ans.append(abs(lsum - rsum))
        rsum -= n
    
    return ans


# Implement pow(x, n), which calculates x raised to the power n (i.e., xn).
def myPow(x: float, n: int) -> float:
    def aux(x, n):
        if n == 0:
            return 1
        if n == 1:
            return x
        
        a = aux(x, n // 2)
        if n % 2 == 0:
            return a * a
        else:
            return a * a * x
        
    if n < 0:
        return 1 / aux(x, -n)
    return aux(x, n)
    
def myPowIter(x: float, n: int) -> float:
    # iteraively square number and halve n
    # if n is odd, multiply an additional x and continue

    if n == 0:
        return 1
    
    ans = 1
    neg = n < 0
    n = abs(n)

    while n > 0:
        if n % 2 == 1:
            ans = ans * x
        x *= x
        n = n // 2

    return ans if not neg else 1 / ans

print(myPowIter(2.000, 6))
    
