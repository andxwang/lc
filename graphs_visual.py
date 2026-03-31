from graphviz import Graph
from collections import deque

def get_ccs(isConnected):
    visited = set()
    n = len(isConnected)
    components = []  # list of sets
    for i in range(n):
        if i not in visited:
            stack = [i]
            this_component = set()
            visited.add(i)
            this_component.add(i)
            while stack:
                curr = stack.pop()
                for nei_idx, nei in enumerate(isConnected[curr]):
                    if nei == 1 and nei_idx not in visited:
                        this_component.add(nei_idx)
                        visited.add(nei_idx)
            components.append(this_component)
    return components

def visualize_findCircleNum(isConnected):
    """
    Visualize the adjacency matrix as an undirected graph, coloring each connected component (province) differently.
    Args:
        isConnected (list[list[int]]): adjacency matrix
    Returns:
        graphviz.Graph: the rendered graph object
    """
    n = len(isConnected)
    g = Graph('Provinces', engine='neato')

    components = get_ccs(isConnected)

    palette = ["red", "blue", "green", "orange", "purple", "brown", "cyan", "magenta", "gold", "pink"]
    node_colors = {}
    for idx, comp in enumerate(components):
        color = palette[idx % len(palette)]
        for node in comp:
            node_colors[node] = color

    for i in range(n):
        g.node(str(i), label=f"City {i}", style="filled", fillcolor=node_colors[i])

    for i in range(n):
        for j in range(i+1, n):
            if isConnected[i][j]:
                g.edge(str(i), str(j))

    return g


# 0-1-2, 3-4, 5-6
cit = [
    [1,1,1,0,0,0,0],
    [1,1,1,0,0,0,0],
    [1,1,1,0,0,0,0],
    [0,0,0,1,1,0,0],
    [0,0,0,1,1,0,0],
    [0,0,0,0,0,1,1],
    [0,0,0,0,0,1,1],
]
print(get_ccs(cit))
g = visualize_findCircleNum(cit)
g.render('provinces', view=True, format='png')
