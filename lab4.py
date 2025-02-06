
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
                curr_area = 1
                stack = [(r, c)]
                grid[r][c] = 0
                while stack:
                    i, j = stack.pop()
                    # print(grid)
                    for ni, nj in [(i - 1, j), (i, j - 1), (i + 1, j), (i, j + 1)]:
                        if ni in range(height) and nj in range(width) and grid[ni][nj] == 1:
                            grid[ni][nj] = 0
                            print("adding from", (ni, nj))
                            curr_area += 1
                            stack.append((ni, nj))
                    
                print("Finished island with area", curr_area)
                max_area = max(max_area, curr_area)
    
    return max_area


grid = [
    [1,1,1,1,0],
    [1,1,0,1,0],
    [1,1,0,0,0],
    [0,0,1,0,1]
]
print(maxAreaOfIsland(grid))
        