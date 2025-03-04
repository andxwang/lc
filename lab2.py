
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __str__(self):
        lines, *_ = self._display_aux()
        return '\n'.join(lines)

    def _display_aux(self):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if self.right is None and self.left is None:
            line = f'{self.val}'
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if self.right is None:
            lines, n, p, x = self.left._display_aux()
            s = f'{self.val}'
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if self.left is None:
            lines, n, p, x = self.right._display_aux()
            s = f'{self.val}'
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self.left._display_aux()
        right, m, q, y = self.right._display_aux()
        s = f'{self.val}'
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first + u * ' ' + second for first, second in zipped_lines]
        return [first_line, second_line] + lines, n + m + u, max(p, q) + 2, n + u // 2


def sortedArrayToBST(nums: list[int]) -> TreeNode | None:
    if len(nums) == 0:
        return None
    if len(nums) == 1:
        return TreeNode(nums[0])
    m = len(nums) // 2  # left
    root = TreeNode(nums[m])
    root.left = sortedArrayToBST(nums[:m])
    root.right = sortedArrayToBST(nums[m+1:])
            
    return root


def invertTree(root: TreeNode | None) -> TreeNode | None:
    if root is None:
        return None
    
    # root.left, root.right = invertTree(root.right), invertTree(root.left)

    # first swap the immediate children
    root.left, root.right = root.right, root.left

    # then, recursively swap the subtrees
    invertTree(root.left)
    invertTree(root.right)

    return root

def diameterOfBinaryTree(root: TreeNode | None) -> int:
    # sum of height(left) and height(right) is this node's diameter
    ans = [0]
    
    def get_height(node: TreeNode | None):
        if not node:
            return 0
        height_left = get_height(node.left)
        height_right = get_height(node.right)
        
        ans[0] = max(ans[0], height_left + height_right)
        return max(height_left, height_right ) + 1
    
    get_height(root)

    return ans[0]

def isSameTree(p: TreeNode | None, q: TreeNode | None) -> bool:
    # if both are None, return True
    if not p and not q:
        return True
    # o/w both must be not None to be same tree
    if not p or not q:
        return False
    if p.val == q.val:
        return isSameTree(p.left, q.left) and isSameTree(p.right, q.right)
    return False

from collections import defaultdict, deque
level_map = defaultdict(list)

def levelOrder(root: TreeNode | None) -> list[list[int]]:
    levelOrderAux(root, 0)
    return level_map.values()

def levelOrderAux(node: TreeNode | None, level: int):
    if not node:
        return
    level_map[level].append(node.val)
    levelOrderAux(node.left, level + 1)
    levelOrderAux(node.right, level + 1)

def levelOrderQueue(root: TreeNode | None) -> list[list[int]]:
    if not root:
        return []
    
    res = []
    q = deque()
    q.append(root)

    while q:
        level = []
        length = len(q)
        for i in range(length):
            # want to just pop the ones originally in the queue
            node = q.popleft()
            level.append(node.val)
            if node.left: q.append(node.left)
            if node.right: q.append(node.right)
        res.append(level)

    return res


def isSubtree(root: TreeNode | None, subRoot: TreeNode | None) -> bool:

    if not root:
        return False
    
    if isSameTree(root, subRoot):
        return True
    
    return isSubtree(root.left, subRoot) or isSubtree(root.right, subRoot)
    
def lowestCommonAncestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    """
    All Node.val are unique.
    p != q
    p and q will exist in the BST.
    we allow a node to be a descendant of itself
    """

    p_list = []
    q_list = []

    def binSearchAlt(root: TreeNode, node: TreeNode, path: list):
        if root.val == node.val:
            path.append(root.val)
            return root
        if node.val > root.val:
            path.append(root.val)
            return binSearchAlt(root.right, node, path)
        else:
            path.append(root.val)
            return binSearchAlt(root.left, node, path)
        
    binSearchAlt(root, p, p_list)
    binSearchAlt(root, q, q_list)

    # find last element in the two lists that are the same
    i = 0
    for i in range(1, min(len(p_list), len(q_list))):
        if p_list[i] != q_list[i]:
            return p_list[i - 1]
    return p_list[i]

def lcaSimple(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    large = max(p.val, q.val)
    small = min(p.val, q.val)

    if large < root.val:
        return lcaSimple(root.left, p, q)
    elif small > root.val:
        return lcaSimple(root.right, p, q)
    else:
        return root

def goodNodes(root: TreeNode) -> int:
    stack = []  # list of (node, curr_max)
    answer = 0
    stack.append((root, root.val))
    while stack:
        node, curr_max = stack.pop(-1)
        if node.val >= curr_max:
            answer += 1
        # print(node.val)
        if node.left:
            stack.append((node.left, max(curr_max, node.left.val)))
        if node.right:
            stack.append((node.right, max(curr_max, node.right.val)))
        # print([x.val for x in stack])
    return answer

def goodNodesDfs(root: TreeNode) -> int:
    res = [0]
    def dfs(node: TreeNode | None, curr_max: int):
        if node is None:
            return
        if node.val >= curr_max:
            res[0] += 1
        dfs(node.left, max(curr_max, node.val))
        dfs(node.right, max(curr_max, node.val))

    dfs(root, root.val)

    return res[0]

def isValidBST(root: TreeNode | None) -> bool:
    def dfs(node: TreeNode | None, left: int, right: int):
        if node is None:
            return True
        
        # left < node.val < right
        if node.val <= left or node.val >= right:
            return False
        
        return dfs(node.left, left, node.val) and dfs(node.right, node.val, right)
    
    return dfs(root, -2e31, 2e31 - 1)

def inOrderTraverseTree(root: TreeNode | None) -> list[int]:
    stack = []
    result = []
    if not root:
        return []
    curr = root
    while stack or curr:
        while curr:
            stack.append(curr)
            curr = curr.left
        # reached bottom of left
        curr = stack.pop()
        result.append(curr.val)
        curr = curr.right
    return result


def kthSmallest(root: TreeNode | None, k: int) -> int:
    # in order traversal the tree
    idx = 1
    curr = root
    stack = []
    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        if idx == k:
            return curr.val
        curr = curr.right
        idx += 1

def preOrderTraverse(root: TreeNode | None) -> list[int]:
    if not root:
        return []
    result = []
    
    def aux(node: TreeNode | None):
        if not node:
            return
        result.append(node.val)
        aux(node.left)
        aux(node.right)

    aux(root)
    return result

def buildTree(preorder: list[int], inorder: list[int]) -> TreeNode | None:
    if not preorder or not inorder:
        return None
    
    root = TreeNode(preorder[0])
    mid = inorder.index(root.val)  # index to split inorder list: left vs right subtrees

    root.left = buildTree(preorder[1:1+mid], inorder[:mid])
    root.right = buildTree(preorder[1+mid:], inorder[mid+1:])

    return root

def maxPathSum(root: TreeNode) -> int:
    result = [root.val]

    def dfs(node: TreeNode | None):
        if not node:
            return 0
        
        left_max = max(dfs(node.left), 0)
        right_max = max(dfs(node.right), 0)

        result[0] = max(result[0], left_max + right_max + node.val)
        # this node: return the max path without splitting both ways
        return max(left_max + node.val, right_max + node.val)
    
    dfs(root)

    return result[0]

def delNodes(root: TreeNode | None, to_delete: list[int]) -> list[TreeNode]:
    if not root:
        return []
    
    result = []
    to_delete_set = set(to_delete)

    def aux(node: TreeNode | None, is_root: bool):
        if not node:
            return None
        if node.val in to_delete_set:
            node.left = aux(node.left, True)
            node.right = aux(node.right, True) 
            return None
        if is_root:
            result.append(node)
        node.left = aux(node.left, False)
        node.right = aux(node.right, False)
        return node
        
    aux(root, True)
    return result

def hasPathSum(root: TreeNode | None, targetSum: int) -> bool:
    if not root:
        return False
    
    def aux(node: TreeNode | None, currSum: int):
        if not node.left and not node.right:
            # leaf node
            return currSum + node.val == targetSum
        
        return (node.left is not None and aux(node.left, node.val + currSum)) or \
            (node.right is not None and aux(node.right, node.val + currSum))
                
    return aux(root, 0)

def hasPathSumAlt(root: TreeNode | None, targetSum: int) -> bool:
    if not root:
        return False
    
    if not root.left and not root.right:
        return root.val == targetSum
    
    return hasPathSumAlt(root.left, targetSum - root.val) or hasPathSumAlt(root.right, targetSum - root.val)


r0 = TreeNode(1)
r0.left = TreeNode(2)
r0.right = TreeNode(3)
r0.left.left = TreeNode(4)
r0.left.right = TreeNode(5)
r0.left.left.left = TreeNode(6)
r0.left.right.right = TreeNode(8)
r0.left.right.right.left = TreeNode(9)
r0.left.right.right.left.left = TreeNode(10)
# print(r0)

r1 = TreeNode(15)
r1.left = TreeNode(10)
r1.right = TreeNode(20)
r1.left.left = TreeNode(8)
r1.left.right = TreeNode(12)
r1.left.right.right = TreeNode(13)
r1.right.left = TreeNode(17)
r1.right.right = TreeNode(25)
r1.left.left.left = TreeNode(6)
r1.left.left.right = TreeNode(9)
# print(r1)

r2 = TreeNode(1)
r2.left = TreeNode(2)
r2.right = TreeNode(3)
r2.left.left = TreeNode(4)
r2.left.right = TreeNode(5)
r2.right.left = TreeNode(6)
r2.right.right = TreeNode(7)
# print(r2)

r3 = TreeNode(1)
r3.left = TreeNode(2)
# print(r3)

