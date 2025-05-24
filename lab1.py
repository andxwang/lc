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
