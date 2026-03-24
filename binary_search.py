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
