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
