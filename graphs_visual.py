from graphviz import Graph, Digraph
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


def visualize_toposort(numCourses, prerequisites):
    """
    Visualize the course prerequisite graph as a directed acyclic graph (DAG).
    Args:
        numCourses (int): total number of courses
        prerequisites (list[list[int]]): list of [course, prerequisite] pairs where prerequisite must be taken before course
    Returns:
        graphviz.Graph: the rendered graph object
    """
    from collections import defaultdict
    
    g = Digraph('Course Prerequisites', engine='dot')
    g.attr(rankdir='LR')  # Left to right layout
    
    # Build the graph structure
    graph = defaultdict(list)
    indegrees = {i: 0 for i in range(numCourses)}
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        indegrees[course] += 1
    
    # Color based on indegree (prerequisites required)
    # Courses with 0 prerequisites are "root" courses (green)
    # Courses with many prerequisites are deeper in the DAG (red)
    colors = {}
    max_indegree = max(indegrees.values()) if indegrees.values() else 0
    
    for course, indeg in indegrees.items():
        if indeg == 0:
            colors[course] = "lightgreen"
        elif indeg == max_indegree:
            colors[course] = "lightcoral"
        else:
            # gradient between green and red
            ratio = indeg / max(max_indegree, 1)
            colors[course] = f"#ff{int(255 * (1 - ratio)):02x}{int(255 * (1 - ratio)):02x}"
    
    # Add nodes
    for course in range(numCourses):
        g.node(str(course), label=str(course), style="filled", fillcolor=colors[course])
    
    # Add edges: prereq -> course (shows flow from prerequisite to course)
    for course, prereq in prerequisites:
        g.edge(str(prereq), str(course))
    
    return g


# Example 1: Simple course prerequisite graph
print("=== Example 1: Simple Prerequisites ===")
numCourses1 = 4
prerequisites1 = [[1, 0], [2, 0], [3, 1], [3, 2]]
g1 = visualize_toposort(numCourses1, prerequisites1)
g1.render('course_prerequisites_1', view=True, format='png')

# Example 2: Linear chain
print("=== Example 2: Linear Chain ===")
numCourses2 = 5
prerequisites2 = [[1, 0], [2, 1], [3, 2], [4, 3]]
g2 = visualize_toposort(numCourses2, prerequisites2)
g2.render('course_prerequisites_2', view=True, format='png')

# Example 3: More complex
print("=== Example 3: Complex Prerequisites ===")
numCourses3 = 7
prerequisites3 = [[1, 0], [2, 0], [3, 1], [4, 2], [4, 3], [5, 4], [4, 6]]
g3 = visualize_toposort(numCourses3, prerequisites3)
g3.render('course_prerequisites_3', view=True, format='png')
