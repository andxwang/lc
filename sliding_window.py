from collections import Counter


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
