import numpy as np
import networkx as nx
from skimage.morphology import skeletonize

class FloodRouter:
    """
    Handles graph creation from road masks and pathfinding logic
    including flood simulation.
    """
    def __init__(self, mask):
        """
        Initialize with a binary mask (0=background, 1=road).
        """
        # Ensure mask is binary boolean
        self.mask = mask > 0
        self.skeleton = skeletonize(self.mask)
        self.graph = self._build_graph()

    def _build_graph(self):
        """
        Convert skeletonized image to NetworkX graph.
        Nodes are (y, x) coordinates of road pixels.
        Edges connect adjacent road pixels (8-connectivity).
        """
        rows, cols = np.where(self.skeleton)
        # Create a set for O(1) lookups
        nodes = set(zip(rows, cols))
        
        G = nx.Graph()
        
        # Add all road pixels as nodes
        # In a real optimized system, we'd only add intersections and endpoints,
        # but for this scale, pixel-graph is fine and simpler to implement.
        for r, c in nodes:
            G.add_node((r, c))
            
            # Check neighbors (8-connectivity)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in nodes:
                        # Weight could be distance (1 for cardinal, sqrt(2) for diagonal)
                        dist = np.sqrt(dr**2 + dc**2)
                        G.add_edge((r, c), (nr, nc), weight=dist)
        return G

    def find_nearest_node(self, point):
        """
        Find the nearest road node to a given (y, x) point.
        """
        if not self.graph.nodes:
            return None
            
        nodes = np.array(self.graph.nodes)
        point = np.array(point)
        dists = np.sum((nodes - point)**2, axis=1)
        idx = np.argmin(dists)
        return tuple(nodes[idx])

    def find_path(self, start_point, end_point, flooded_nodes=None):
        """
        Find shortest path from start to end.
        Optionally avoid flooded_nodes.
        """
        start_node = self.find_nearest_node(start_point)
        end_node = self.find_nearest_node(end_point)
        
        if start_node is None or end_node is None:
            return None, "No road nodes found"

        # Create a view of the graph without flooded nodes if necessary
        if flooded_nodes:
            # We can't easily modify G in place without affecting state,
            # so we use a subgraph or just a check during traversal.
            # NetworkX allows removing nodes, but let's copy to be safe 
            # or just assume the user will create a new router for heavy mods.
            # actually, copying a large graph is slow.
            # Better: remove nodes temporarily then add back, or copy.
            # Given image size, copy is probably acceptable (~256x256 or 1024x1024 nodes max).
            # Actually, let's just use G.subgraph logic.
            
            available_nodes = set(self.graph.nodes) - set(flooded_nodes)
            if start_node not in available_nodes or end_node not in available_nodes:
                 return None, "Start or End point is flooded!"
            
            subgraph = self.graph.subgraph(available_nodes)
            try:
                path = nx.shortest_path(subgraph, start_node, end_node, weight='weight')
                return path, "Success"
            except nx.NetworkXNoPath:
                return None, "No path exists (blocked by flood)"
        else:
            try:
                path = nx.shortest_path(self.graph, start_node, end_node, weight='weight')
                return path, "Success"
            except nx.NetworkXNoPath:
                return None, "No path exists"

    def get_flooded_nodes(self, center, radius):
        """
        Identify nodes within radius of center (y, x).
        """
        cy, cx = center
        flooded = []
        for r, c in self.graph.nodes:
            if (r - cy)**2 + (c - cx)**2 <= radius**2:
                flooded.append((r, c))
        return set(flooded)
