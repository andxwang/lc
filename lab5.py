import heapq
from typing import List
class KthLargestSlow:

    def __init__(self, k: int, nums: List[int]):
        self.arr = sorted(nums[:], reverse=False)
        print(self.arr)
        self.k = k
        # self.kth = self.arr[-k]

    def add(self, val: int) -> int:
        l, r = 0, len(self.arr)
        while l < r:
            m = l + (r - l) // 2
            if val <= self.arr[m]:
                r = m
            else:
                l = m + 1
        self.arr.insert(l, val)
        print("new:", self.arr)
        return self.arr[-self.k]


class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        # min heap with k largest nums
        self.minHeap = nums
        self.k = k
        heapq.heapify(self.minHeap)
        print(self.minHeap, type(self.minHeap))
        while len(self.minHeap) > k:
            heapq.heappop(self.minHeap)

    def add(self, val: int) -> int:
        heapq.heappush(self.minHeap, val)
        if len(self.minHeap) > self.k:
            heapq.heappop(self.minHeap)
        return self.minHeap[0]  # retrieve min

def lastStoneWeight(stones: List[int]) -> int:
    while len(stones) >= 2:
        heapq._heapify_max(stones)
        a = heapq.heappop(stones)
        heapq._heapify_max(stones)
        b = heapq.heappop(stones)
        print(a, b)
        if a != b:
            heapq.heappush(stones, abs(a - b))
            
    if stones:
        return stones[0]
    return 0

def kClosest(points: List[List[int]], k: int) -> List[List[int]]:
    max_heap = []
    for p1, p2 in points:
        dist = -(p1*p1 + p2*p2)  # negative to make "max heap"
        heapq.heappush(max_heap, (dist, [p1, p2]))
        if len(max_heap) > k:
            # if more than k elements, remove smallest dist/point pair, aka "most negative" == largest dist
            heapq.heappop(max_heap)
            
    return [p for (_, p) in max_heap]

print(kClosest([[1,3],[-2,2],[6,8]], 2))
