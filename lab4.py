def numIslands(grid: list[list[str]]) -> int:
    # count connected components
    # doesn't use visited set because modifies grid in-place
    height, width = len(grid), len(grid[0])
    ccs = 0

    for r in range(height):
        for c in range(width):
            if grid[r][c] == '1':
                stack = [(r, c)]
                ccs += 1
                # turn all neighbors into 0
                while stack:
                    i, j = stack.pop()
                    # everything in stack must be valid
                    grid[i][j] = '0'
                    for (ni, nj) in ((i - 1, j), (i, j - 1), (i + 1, j), (i, j + 1)):
                        # if ni in range(height) and nj in range(width):
                        if 0 <= ni < height and 0 <= nj < width and grid[ni][nj] == '1':
                            stack.append((ni, nj))

    return ccs


def maxAreaOfIsland(grid: list[list[str]]) -> int:
    height, width = len(grid), len(grid[0])
    max_area = 0
    for r in range(height):
        for c in range(width):
            if grid[r][c] == 1:
                curr_area = 0
                stack = [(r, c)]
                while stack:
                    i, j = stack.pop()
                    if grid[i][j] == 0:
                        continue
                    grid[i][j] = 0
                    curr_area += 1
                    print("adding from", (i, j))
                    for ni, nj in [(i - 1, j), (i, j - 1), (i + 1, j), (i, j + 1)]:
                        if ni in range(height) and nj in range(width) and grid[ni][nj] == 1:
                            stack.append((ni, nj))

                print("Finished island with area", curr_area)
                max_area = max(max_area, curr_area)

    return max_area


# grid = [
#     [1, 1, 1, 1, 0],
#     [1, 1, 0, 1, 0],
#     [1, 1, 0, 0, 0],
#     [0, 0, 1, 0, 1]
# ]


def spiralOrder(matrix: list[list[int]]) -> list[int]:
    """strategy:
    whenever possible, do in order:
    - go right
    - go down
    - go left
    - go up
    keep tracker of curr direction
    """
    visited = set()
    m, n = len(matrix), len(matrix[0])
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U
    out = []
    dir_idx = 0
    curr = (0, 0)
    while len(visited) < m * n:
        i, j = curr
        visited.add((i, j))
        out.append(matrix[i][j])

        di, dj = dirs[dir_idx]
        if not (i + di in range(m) and j + dj in range(n) and (i + di, j + dj) not in visited):
            dir_idx = (dir_idx + 1) % 4
            di, dj = dirs[dir_idx]

        curr = (i + di, j + dj)

    return out


from collections import deque
def orangesRotting(grid: list[list[int]]) -> int:
    height, width = len(grid), len(grid[0])
    fresh_oranges = 0
    queue = deque()
    for r in range(height):
        for c in range(width):
            if grid[r][c] == 1:
                fresh_oranges += 1
            if grid[r][c] == 2:
                queue.append((r, c, 0))

    if fresh_oranges == 0:
        return 0

    max_time = 0
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    while queue:
        i, j, t = queue.popleft()
        print(f"time {t}:\t{(i, j)}: {grid[i][j]}")
        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if ni in range(height) and nj in range(width) and grid[ni][nj] == 1:
                grid[ni][nj] = 2
                queue.append((ni, nj, t + 1))
                max_time = t + 1
                fresh_oranges -= 1

    return max_time if fresh_oranges == 0 else -1

def pacificAtlanticPrimitive(heights: list[list[int]]) -> list[list[int]]:
    # is there an existing path from cell (r, c) to ( (0, j) or (i, 0) ) and ( (0, w-1) or (h-1, 0))?
    height, width = len(heights), len(heights[0])
    dirs = ((1, 0), (0, 1), (-1, 0), (0, -1))

    def check_valid(r, c):
        stack = [(r, c)]
        visited = set()
        pac, atl = False, False
        while stack and (not pac or not atl):
            i, j = stack.pop()
            visited.add((i, j))
            print(f"visiting {i, j}")
            if i == 0 or j == 0:
                pac = True
            if i == height - 1 or j == width - 1:
                atl = True
            for di, dj in dirs:
                ni, nj = i + di, j + dj
                if ni in range(height) and nj in range(width) and \
                    (ni, nj) not in visited and \
                    heights[ni][nj] <= heights[i][j]:
                    stack.append((ni, nj))

        return pac and atl

    # print(f"result: {check_valid(2, 2)}")
    res = []
    for r in range(height):
        for c in range(width):
            if check_valid(r, c):
                res.append((r, c))

    return res

def pacificAtlantic(heights: list[list[int]]) -> list[list[int]]:
    height, width = len(heights), len(heights[0])
    dirs = ((1, 0), (0, 1), (-1, 0), (0, -1))
    # these sets represent which cells can touch the atlantic or pacific
    pac = set()
    atl = set()
    for i in range(height):
        pac.add((i, 0))
        atl.add((i, width - 1))
    for j in range(width):
        pac.add((0, j))
        atl.add((height - 1, j))

    stack = list(pac)
    while stack:
        # add the neighbors iff they're >=
        i, j = stack.pop()
        pac.add((i, j))
        # heights[i][j] = -1
        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if ni in range(height) and nj in range(width) and \
                heights[ni][nj] >= heights[i][j] and \
                (ni, nj) not in pac:
                stack.append((ni, nj))

    stack = list(atl)
    while stack:
        i, j = stack.pop()
        atl.add((i, j))
        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if ni in range(height) and nj in range(width) and \
                    heights[ni][nj] >= heights[i][j] and \
                    (ni, nj) not in atl:
                stack.append((ni, nj))

    return list(atl.intersection(pac))

def pacificAtlanticRecursive(heights: list[list[int]]) -> list[list[int]]:
    height, width = len(heights), len(heights[0])
    dirs = ((1, 0), (0, 1), (-1, 0), (0, -1))

    pac = set()
    atl = set()

    def dfs(i, j, visited):
        if (i, j) in visited:
            return
        visited.add((i, j))

        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if 0 <= ni < height and 0 <= nj < width:
                if heights[ni][nj] >= heights[i][j]:
                    dfs(ni, nj, visited)

    for r in range(height):
        dfs(r, 0, pac)
    for c in range(width):
        dfs(0, c, pac)

    for r in range(height):
        dfs(r, width - 1, atl)
    for c in range(width):
        dfs(height - 1, c, atl)

    return list(atl.intersection(pac))


# gr = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]
# for r, c in pacificAtlantic(gr):
#     print(f"{r, c}: {gr[r][c]}")
# print(pacificAtlanticRecursive(gr))
            

class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

def create_graph(adj_list):
    if not adj_list:
        return None
    
    node_map = {}  # Maps value to Node
    
    # Create nodes
    for i in range(1, len(adj_list) + 1):
        node_map[i] = Node(i)
    
    # Assign neighbors
    for i in range(len(adj_list)):
        node = node_map[i + 1]
        node.neighbors = [node_map[neighbor] for neighbor in adj_list[i]]
    
    return node_map[1]  # Return the reference to the first node

def print_graph(node):
    if not node:
        print("Empty graph")
        return
    
    visited = set()
    queue = [node]
    
    while queue:
        current = queue.pop(0)
        if current.val in visited:
            continue
        visited.add(current.val)
        print(f"Node {current.val}: {[neighbor.val for neighbor in current.neighbors]}")
        queue.extend(current.neighbors)


def cloneGraph(node: Node | None) -> Node | None:
    if not node:
        return None
    
    node_map = {node: Node(node.val)}  # map old to new node
    stack = [node]
    
    while stack:
        curr = stack.pop()
        for nbor in curr.neighbors:
            if nbor not in node_map:
                node_map[nbor] = Node(nbor.val)
                stack.append(nbor)
            node_map[curr].neighbors.append(node_map[nbor])
            
    return node_map[node]

from typing import List
def islandsAndTreasure(grid: List[List[int]]) -> None:
    height, width = len(grid), len(grid[0])
    dirs = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    queue = deque()
    visited = set()
    
    for i in range(height):
        for j in range(width):
            if grid[i][j] == 0:
                queue.append((i, j, 0))

    visited = set()
    
    while queue:
        i, j, d = queue.popleft()
        visited.add((i, j))
        grid[i][j] = min(grid[i][j], d)
        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if 0 <= ni < height and 0 <= nj < width and \
                grid[ni][nj] > 0 and (ni, nj) not in visited:
                queue.append((ni, nj, d + 1))
    
def surroundedRegions(board: List[List[str]]) -> None:
    """
    Do not return anything, modify board in-place instead.
    """
    height, width = len(board), len(board[0])
    dirs = [(0, 1), (1, 0), (-1, 0), (0, -1)]
    
    def dfs(stack, fill_char):
        while stack:
            i, j = stack.pop()
            board[i][j] = fill_char
            for di, dj in dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < height and 0 <= nj < width:
                    if board[ni][nj] == 'O':
                        stack.append((ni, nj))

    # first add only border cells that are 'O' to stack, including top, bottom, left, right
    border_cells = set()
    for r in range(height):
        if board[r][0] == 'O':
            border_cells.add((r, 0))
        if board[r][width - 1] == 'O':
            border_cells.add((r, width - 1))
    for c in range(width):
        if board[0][c] == 'O':
            border_cells.add((0, c))
        if board[height - 1][c] == 'O':
            border_cells.add((height - 1, c))
    
    stack = list(border_cells)
    
    # first run: mark border cell connected components with 'T'
    # then, run again and mark all other 'O' cells with 'X'
    while stack:
        dfs(stack, 'T')
        
    # now, run again and mark all 'O' cells with 'X'
    # and all 'T' cells with 'O'
    for r in range(height):
        for c in range(width):
            if board[r][c] == 'O':
                board[r][c] = 'X'
            elif board[r][c] == 'T':
                board[r][c] = 'O'

    return

from collections import deque
def canFinish(numCourses: int, prerequisites: List[List[int]]) -> bool:
    """topological sort graph problem"""
    count = 0
    indegrees = {}
    courses = []
    for _ in range(numCourses):
        indegrees[_] = 0
        courses.append([])
    for a, b in prerequisites:
        indegrees[a] += 1
        courses[b].append(a)
        
    queue = deque()
    for k, v in indegrees.items():
        if v == 0:
            queue.append(k)
        
    while queue:
        n = queue.popleft()
        if indegrees[n] == 0:
            count += 1
            # decrease n's neighbor's indegrees by 1
            for nbor in courses[n]:
                indegrees[nbor] -= 1
                if indegrees[nbor] == 0:
                    queue.append(nbor)
        
    return count == numCourses
            
def findOrder(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    topo_sort = []
    indegrees = [0] * numCourses
    courses = [[] for _ in range(numCourses)]
    
    for post, pre in prerequisites:
        courses[pre].append(post)
        indegrees[post] += 1
    
    queue = deque()
    for i, n in enumerate(indegrees):
        if n == 0:
            queue.append(i)
            
    while queue:
        n = queue.popleft()
        topo_sort.append(n)
        # "remove" from graph
        for nbor in courses[n]:
            indegrees[nbor] -= 1
            if indegrees[nbor] == 0:
                queue.append(nbor)
                
    return topo_sort #if len(topo_sort) == numCourses else []

def validTree(n: int, edges: List[List[int]]) -> bool:
    # ensure min comes before max in edge tuple
    edges = [(min(a, b), max(a, b)) for (a, b) in edges]
    graph = [set() for _ in range(n)]
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
        
    # first check connectivity
    visited = set()
    stack = [0]
    while stack:
        x = stack.pop()
        visited.add(x)
        for nbor in graph[x]:
            if nbor not in visited:
                stack.append(nbor)
    if len(visited) != n:
        return False
    
    visited_nodes = set()
    visited_edges = set()
    stack = [0]
    while stack:
        x = stack.pop()
        if x in visited_nodes:
            return False
        visited_nodes.add(x)
        for nbor in graph[x]:
            # check edges
            nbor_edge = (min(x, nbor), max(x, nbor))
            if nbor_edge not in visited_edges:
                visited_edges.add(nbor_edge)
                stack.append(nbor)
    return True
    
    
def countComponents(n: int, edges: List[List[int]]) -> int:
    visited = set()
    graph = [[] for _ in range(n)]
    for (a, b) in edges:
        graph[a].append(b)
        graph[b].append(a)
    
    def bfs(i):
        queue = deque([i])
        visited.add(i)
        while queue:
            x = queue.popleft()
            visited.add(x)
            for nbor in graph[x]:
                if nbor not in visited:
                    queue.append(nbor)
            
    ccs = 0
    for i in range(n):
        if i not in visited:
            ccs += 1
            bfs(i)
            
    return ccs

print(countComponents(6, [[0,1], [1,2], [2,3], [4,5]]))
