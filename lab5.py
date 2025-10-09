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

def findKthLargest(nums: List[int], k: int) -> int:
    min_heap = []
    for n in nums:
        heapq.heappush(min_heap, n)
        if len(min_heap) > k:
            heapq.heappop(min_heap)
            
    return min_heap[0]  # smallest of largest k elements = kth largest

from collections import Counter, defaultdict
def leastInterval(tasks: List[str], n: int) -> int:
    tasks = list(map(list, Counter(tasks).most_common()))
    last = {}  # store 'X': last time X occurred
    time = 0
    result = []  # only for debugging
    while len(tasks) > 0:
        time += 1
        tasks.sort(key=lambda t: t[1], reverse=True)
        # iterate through tasks and find the first available
        for i, (x, c) in enumerate(tasks):
            if x not in last or time - n > last[x]:
                break
        else:
            # idle
            # time += 1
            result.append("idle")
            continue
        x, c = tasks[i]
        last[x] = time
        result.append(x)


        tasks[i][1] -= 1
        if tasks[i][1] == 0:
            tasks.pop(i)
        # time += 1

    print("results:", '->'.join(result))
    return time


# print(leastInterval(["A","C","A","B","D","A","B"], 4))
# print(leastInterval(["A","A","A","B","B","B"], 2))
# print(leastInterval(["A","C","A","B","D","B"], 1))

from collections import deque
def leastInterval2(tasks: List[str], n: int) -> int:
    tasks = list((-v, k) for k, v in Counter(tasks).items())
    heapq.heapify(tasks)

    cooldown_queue = deque()  # (freq, task, next avail time)
    time = 0
    while tasks or cooldown_queue:
        time += 1
        while cooldown_queue:
            freq, task, next_avail_time = cooldown_queue[0]
            if time >= next_avail_time:
                # task is ready; add it back to heap and popleft from queue
                heapq.heappush(tasks, (freq, task))
                _ = cooldown_queue.popleft()
            else:
                break
        # pop the most frequent task
        if len(tasks) > 0:
            freq, task = heapq.heappop(tasks)
            print(f"time {time}: task {task}")
            if freq + 1 < 0:
                cooldown_queue.append((freq + 1, task, time + n + 1))  # freq + 1 bc it's negative
        else:
            if cooldown_queue:
                # idle
                _, _, next_avail_time = cooldown_queue[0]
                time = next_avail_time - 1
                print(f"idling until time {time}")
        

    return time

# print(leastInterval2(["A","C","A","B","D","A","B"], 10))
# print(leastInterval2(["A","A","A","B","B","B"], 2))
print(leastInterval2(list('SADHFUIHEWIUNNBFDJKHGKLJDSFHFKLJRESHFILUAWEGFBFKDLJBBJNFDJBNSDKFJNBEUIYSORNIUYORHBGIOWAHBGIWAUEHFEIAWUHEFKLJSADHF'), 26))

class Twitter:

    def __init__(self):
        self.time = 0
        self.followMap = defaultdict(set)  # x: y means x follows y
        self.tweetMap = defaultdict(list)  # user: list of tweets

    def postTweet(self, userId: int, tweetId: int) -> None:
        self.tweetMap[userId].append((self.time, tweetId))
        # print(f"adding tweet {tweetId} to user {userId}")
        self.time +=1

    def getNewsFeed(self, userId: int) -> List[int]:
        # print(f"getting news feed for user {userId}: {self.tweetMap}")
        own_tweets = self.tweetMap[userId][:]
        following_tweets = [t for other in self.followMap[userId] for t in self.tweetMap[other]]
        # print(f"\town: {own_tweets} following: {following_tweets}")
        own_tweets.extend(following_tweets)
        own_tweets = sorted(own_tweets, key=lambda t: t[0], reverse=True)
        return [t[1] for t in own_tweets[:10]]

    def follow(self, followerId: int, followeeId: int) -> None:
        self.followMap[followerId].add(followeeId)
        # print(f"{followerId} followed {followeeId}: map is now {self.followMap}")

    def unfollow(self, followerId: int, followeeId: int) -> None:
        if followerId in self.followMap:
            self.followMap[followerId].remove(followeeId)
            # print(f"{followerId} unfollowed {followeeId}: map is now {self.followMap}")
