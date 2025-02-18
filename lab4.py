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


gr = [[2,1,1,2],[1,1,0,1],[0,1,1,1]]
# gr = [[0,0]]
for _ in gr:
    print(_)
print(orangesRotting(gr))
