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


gr = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]
# for r, c in pacificAtlantic(gr):
#     print(f"{r, c}: {gr[r][c]}")
print(pacificAtlanticRecursive(gr))
