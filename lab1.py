from pprint import pprint

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    @classmethod
    def create_linked_list(cls, arr):
        """
        Creates a new linked list from a given array.

        Args:
            arr (list): The input array.

        Returns:
            ListNode: The head node of the newly created linked list.
        """
        if not arr:
            return None

        head = cls(arr[0])
        current = head

        # Iterate through the array and create new nodes
        for val in arr[1:]:
            new_node = cls(val)
            current.next = new_node
            current = new_node

        return head

    def __str__(self) -> str:
        result = []
        current = self
        while current:
            result.append(str(current.val))
            current = current.next
        return " -> ".join(result)

def reverseList(head: ListNode):
    stack = []
    ptr = head
    while ptr:
        stack.append(ptr.val)
        ptr = ptr.next
    
    new_head = None
    new_ptr = new_head
    while len(stack) > 0:
        new_ptr = ListNode(stack.pop())
        new_ptr = new_ptr.next

    return new_head

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

def reorderList(head: ListNode):
    """
    Do not return anything, modify head in-place instead.
    """
    nodes = []
    curr = head
    while curr:
        nodes.append(curr)
        curr = curr.next

    total = len(nodes) - 1  # total num of operations is n - 1
    i = 0
    
    l, r = 0, len(nodes) - 1
    while l < r and i < total:
        if i < total:
            nodes[l].next = nodes[r]
            l += 1
            i += 1
        if i < total:
            nodes[r].next = nodes[l]
            r -= 1
            i += 1
    # now l and r are at the same node
    nodes[l].next = None

def reorderListLL(head: ListNode):
    mid = head
    temp = head
    while temp:
        temp = temp.next
        if temp and temp.next:
            temp = temp.next
            mid = mid.next
    # now mid points to floor(midpoint)
    # reverse all past mid
    prev = None
    curr = mid.next
    while curr:
        next = curr.next
        curr.next = prev
        prev = curr
        curr = next
    mid.next = None
    # prev is the new head of the reversed part
    head1, head2 = head, prev
    # rearrange nodes in order: every iteration, set temp and then rename h1 and h2
    while head2:
        temp = head1.next
        head1.next = head2
        head1, head2 = head2, temp

class NodeR:
    def __init__(self, x: int, next=None, random=None):
        self.val = int(x)
        self.next = next
        self.random = random

    def __str__(self) -> str:
        result = []
        current = self
        while current:
            result.append(f"[{str(current.val)}, {current.random.val if current.random else None}]")
            current = current.next
        return " -> ".join(result)

def copyRandomList(head: NodeR):
    if head is None:
        return None
    curr = head

    # first instantiate the new nodes
    nodes_map = {}
    while curr:
        nodes_map[curr] = NodeR(curr.val)
        curr = curr.next

    # now, iterate through the list again and assign the next and random node
    for old_node, new_node in nodes_map.items():
        if old_node.next:
            new_node.next = nodes_map[old_node.next]
        if old_node.random:
            new_node.random = nodes_map[old_node.random]

    return nodes_map[head]

def copyRandomListSmart(head: NodeR):
    # 1. duplicate the list within: 1 -> 1' -> 2 -> 2' -> ...
    curr = head
    while curr:
        next = curr.next
        curr.next = NodeR(curr.val)
        curr.next.next = next
        curr = curr.next.next
    # print(head)

    # 2. assign random ptrs of newly inserted nodes
    curr = head
    while curr: 
        if curr.random is not None:
            curr.next.random = curr.random.next
        curr = curr.next.next

    # print(head)

    # 3. restore original list
    curr = head
    prev = prehead = NodeR(0)

    while curr:
        next = curr.next.next

        copy = curr.next
        curr.next = next
        prev.next = copy
        prev = copy

        curr = next

    print(prehead.next)

def hasCycle(head: ListNode) -> bool:
    nodes_map = set()
    curr = head
    while curr:
        nodes_map.add(curr)
        if curr.next in nodes_map:
            return True
        curr = curr.next

    return False


def hasCycleO1(head: ListNode) -> bool:
    slow = fast = head
    while slow.next and slow.next.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            print(slow.val)
            return True
    return False


def findDuplicate(nums: list[int]):
    """without modifying the array nums and uses only constant extra space"""
    # we know the range is [1, len(nums) - 1]
    # nums[i] is the index of num
    slow = fast = nums[0]
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    slow2 = nums[0]
    while slow2 != slow:
        slow2 = nums[slow2]
        slow = nums[slow]
    return slow

def deleteDuplicates(head: ListNode | None) -> ListNode | None:
    curr = head
    while curr and curr.next:
        if curr.val == curr.next.val:
            curr.next = curr.next.next
        else:
            curr = curr.next
    return head


class DLNode:

    def __init__(self, key=0, val=0, prev=None, next=None) -> None:
        self.prev = prev
        self.next = next
        self.key = key
        self.val = val


class DoubleLL:

    def __init__(self, head: DLNode = None, tail: DLNode = None) -> None:
        self.head = head
        self.tail = tail

    def __str__(self):
        nodes = []
        current_node = self.head
        while current_node:
            nodes.append(f"({current_node.key}: {current_node.val})")
            current_node = current_node.next
        return '[ ' + " <-> ".join(nodes) + ' ]'
    
    def remove(self, node: DLNode):
        """remove `node` from doubly ll"""
        if not node:
            return
        
        if node.prev:
            node.prev.next = node.next
        else:
            # node is head
            self.head = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            # node_to_move is tail
            self.tail = node.prev

        node.prev = None
        node.next = None
            
        
    def insert(self, node: DLNode):
        """insert `node` at end"""
        if not node:
            return
        if not self.head:
            self.head = node
            self.tail = node
            return
        node.prev = self.tail
        self.tail.next = node
        self.tail = node


    @classmethod
    def create_linked_list(cls, keys, vals):
        new_list = cls()
        for key, val in zip(keys, vals):
            new_node = DLNode(key, val)
            if not new_list.head:
                new_list.head = new_node
                new_list.tail = new_node
            else:
                new_list.tail.next = new_node
                new_node.prev = new_list.tail
                new_list.tail = new_node
        return new_list


class LRUCache:

    def __init__(self, capacity: int):
        self.cachemap = {}
        self.dll = DoubleLL()  # doubly linked list with head -> LRU, tail -> MRU
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self.cachemap:
            return -1
        
        # move cachemap[key] DLL ptr to MRU
        node_to_move = self.cachemap[key]
        self.dll.remove(node_to_move)
        self.dll.insert(node_to_move)

        return node_to_move.val

    def put(self, key: int, value: int) -> None:        
        if key in self.cachemap:
            self.dll.remove(self.cachemap[key])

        # insert new node
        new_node = DLNode(key, value)
        self.dll.insert(new_node)
        self.cachemap[key] = new_node

        # check cache overflow
        if len(self.cachemap) > self.capacity:
            lru = self.dll.head
            self.dll.remove(lru)
            del self.cachemap[lru.key]

            
# dl = DoubleLL.create_linked_list([0, 1, 2, 3], [10, 20, 30, 40])
# print(dl)
# dl.remove(dl.head)
# print(dl)
# dl.insert(DLNode(6, 9))
# print(dl)

# dl = DoubleLL()
# dl.insert(DLNode(0, 1))
# print(dl)

def run(lol: list[list[int]]):
    lru = LRUCache(lol[0][0])
    for l in lol[1:]:
        if len(l) == 1:
            # get
            print(f"get {l[0]}: {lru.get(l[0])}")
        elif len(l) == 2:
            # put
            print(f"put {l}")
            lru.put(l[0], l[1])
        print(lru.dll)

def getIntersectionNode(headA: ListNode, headB: ListNode) -> None | ListNode:
    set_a = set()
    ptrA = headA
    ptrB = headB
    
    while ptrA:
        set_a.add(ptrA)
        ptrA = ptrA.next
        
    while ptrB:
        if ptrB in set_a:
            return ptrB
        ptrB = ptrB.next
        
    return None
    
def searchInsert(nums: list[int], target: int) -> int:
    if target < nums[0]:
        return 0
    if target > nums[-1]:
        return len(nums)
    
    l = 0
    r = len(nums) - 1
    
    while l < r:
        m = l + (r - l) // 2  # lower
        if nums[m] == target:
            return m
        if nums[m] > target:
            r = m
        else:
            l = m + 1
            
    return l


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

def reverseBetween(head: ListNode | None, left: int, right: int) -> ListNode | None:
    if left == right:
        return head
    
    idx = 1
    curr = head
    before_left = None
    
    while idx < left:
        before_left = curr
        curr = curr.next
        idx += 1
    left_node = curr  # first node to be reversed
    
    print(left_node)
    print(before_left)
    
    prev = None  # this doesn't matter?
    while idx <= right:
        next = curr.next
        curr.next = prev
        prev = curr
        curr = next
        idx += 1
        
    # curr is After right
    # prev is Right
    if before_left:
        before_left.next = prev
    else:
        head = prev
    left_node.next = curr
        
    return head

def removeElements(head: ListNode | None, val: int) -> ListNode | None:
    if head is None:
        return None
    
    prehead = ListNode(0)
    prehead.next = head
    
    prev = prehead
    curr = head
    while curr:
        if curr.val == val:
            prev.next = curr.next
        else:
            prev = curr
        curr = curr.next
        
    return prehead.next


def findMaxFish(grid: list[list[int]]) -> int:
    height, width = len(grid), len(grid[0])
    max_fish = 0
    visited = set()
    
    for r in range(height):
        for c in range(width):
            if grid[r][c] == 0 or (r, c) in visited:
                continue
            stack = []
            stack.append((r, c))  # starting cell
            fish_caught = 0
            while stack:
                i, j = stack.pop()
                if (i, j) not in visited:
                    visited.add((i, j))
                    if grid[i][j] > 0:
                        fish_caught += grid[i][j]
                        # up, left, down, right
                        if i > 0:
                            stack.append((i - 1, j))
                        if j > 0:
                            stack.append((i, j - 1))
                        if i < height - 1:
                            stack.append((i + 1, j))
                        if j < width - 1:
                            stack.append((i, j + 1))
                            
            max_fish = max(max_fish, fish_caught)
                    
    return max_fish


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
    
def isPalindromePrimitive(head: ListNode | None) -> bool:
    vals = []
    curr = head
    while curr:
        vals.append(curr.val)
        curr = curr.next
        
    l, r = 0, len(vals) - 1
    while l < r:
        if vals[l] != vals[r]:
            return False
        l += 1
        r -= 1
        
    return True

def isPalindrome(head: ListNode | None) -> bool:
    slow, fast, prev = head, head, None
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    prev, slow, prev.next = slow, slow.next, None  # have to set prev.next = None to prevent inf loop
    while slow:
        slow.next, prev, slow = prev, slow, slow.next  # in order left to right
        
    # second half of ll is now reversed, and prev points to the last node
    x, y = head, prev
    while y:
        if x.val != y.val:
            return False
        x, y = x.next, y.next
        
    return True

def validPalindromeII(s: str) -> bool:
    """check if string is palindrome after deleting at most one character from it"""
    l, r = 0, len(s) - 1
    while l <= r:
        if s[l] != s[r]:
            a = s[l+1 : r+1]  # exclude l: ..._xxr...
            b = s[l : r]  # exclude r: ...xx_...
            return a == a[::-1] or b == b[::-1]
        l += 1
        r -= 1
    return True

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
    
