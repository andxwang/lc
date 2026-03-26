from typing import List
from collections import defaultdict


def twoSum(nums: list[int], target: int) -> list[int]:
    hmap = {}
    for i, n in enumerate(nums):
        hmap[n] = i
    for i, n in enumerate(nums):
        if target - n in hmap:
            complement_idx = hmap[target - n]  # save so don't access twice
            if complement_idx != i:
                return [i, complement_idx]


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
    
def checkRain(mountains: list[int]):
    """Given n non-negative integers representing an elevation map where the width of each bar is 1, 
    compute how much water it can trap after raining."""
    l, r = 0, len(mountains) - 1
    left_max = mountains[l]
    right_max = mountains[r]
    ans = 0

    while l < r:
        if mountains[l] < mountains[r]:
            # water at l depends only on left_max
            if mountains[l] < left_max:
                ans += left_max - mountains[l]
            left_max = max(left_max, mountains[l])
            l += 1
        elif mountains[r] < mountains[l]:
            if mountains[r] < right_max:
                ans += right_max - mountains[r]
            right_max = max(right_max, mountains[r])
            r -= 1
        else:
            left_max = max(left_max, mountains[l])
            l += 1

    return ans


def print_rain_bars(mountains: list[int]):
    if not mountains:
        print("[]")
        return

    n = len(mountains)
    max_h = max(mountains)

    # compute left and right max heights for each position
    left_max = [0] * n
    right_max = [0] * n
    lm = 0
    for i in range(n):
        lm = max(lm, mountains[i])
        left_max[i] = lm
    rm = 0
    for i in range(n - 1, -1, -1):
        rm = max(rm, mountains[i])
        right_max[i] = rm

    # draw from top level down to 1
    rows = []
    for level in range(max_h, 0, -1):
        row = []
        for i, h in enumerate(mountains):
            if h >= level:
                row.append('#')        # mountain block
            else:
                # if bounded on both sides to at least this level, it's water
                if min(left_max[i], right_max[i]) >= level:
                    row.append('~')    # water
                else:
                    row.append(' ')    # empty
        rows.append(' '.join(row))

    # print the chart
    for r in rows:
        print(r)
    print('-' * (2 * n - 1))            # baseline
    print(' '.join(str(h) for h in mountains))
