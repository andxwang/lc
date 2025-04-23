def climbStairs(n: int) -> int:
    """You are climbing a staircase. It takes n steps to reach the top.
    Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
    """
    if n == 1: return 1
    if n == 2: return 2
    prev, curr = 1, 2  # answers for [n = 1, n = 2]
    for i in range(3, n + 1):
        prev, curr = curr, prev + curr
        
    return curr

print(climbStairs(15))
