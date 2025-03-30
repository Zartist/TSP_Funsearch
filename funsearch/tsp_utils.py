# @title
from __future__ import annotations
import numpy as np
from typing import Tuple

datasets = {}

# def parse_tsp_file(filepath: str):
#     with open(filepath, "r") as file:
#         lines = file.readlines()

#     tsp_data = {
#         # "name": None,
#         "dimension": None,
#         "edge_weight_type": None,
#         "node_coords": []
#     }

#     parsing_nodes = False  

#     for line in lines:
#         line = line.strip()
#         # if line.startswith("NAME"):
#         #     tsp_data["name"] = line.split(":")[1].strip()
#         if line.startswith("DIMENSION"):
#             tsp_data["dimension"] = int(line.split(":")[1].strip())
#         elif line.startswith("EDGE_WEIGHT_TYPE"):
#             tsp_data["edge_weight_type"] = line.split(":")[1].strip()
#         elif line.startswith("NODE_COORD_SECTION"):
#             parsing_nodes = True
#         elif line == "EOF":
#             break
#         elif parsing_nodes:
#             parts = line.split()
#             if len(parts) == 3:
#                 _, x, y = parts
#                 tsp_data["node_coords"].append((int(x), int(y)))

#     return tsp_data  

# 解析 TSP 数据
# tsp_instances = parse_tsp_file("./data/att48/att48.tsp")
datasets['ATT48'] = {'att48': {'dimension': 48, 'edge_weight_type': 'ATT', 
                               'node_coords': [(6734, 1453), (2233, 10), (5530, 1424), (401, 841), (3082, 1644), (7608, 4458), 
                                               (7573, 3716), (7265, 1268), (6898, 1885), (1112, 2049), (5468, 2606), (5989, 2873),
                                                (4706, 2674), (4612, 2035), (6347, 2683), (6107, 669), (7611, 5184), (7462, 3590), 
                                                (7732, 4723), (5900, 3561), (4483, 3369), (6101, 1110), (5199, 2182), (1633, 2809), 
                                                (4307, 2322), (675, 1006), (7555, 4819), (7541, 3981), (3177, 756), (7352, 4506), 
                                                (7545, 2801), (3245, 3305), (6426, 3173), (4608, 1198), (23, 2216), (7248, 3779), 
                                                (7762, 4595), (7392, 2244), (3484, 2829), (6271, 2135), (4985, 140), (1916, 1569), 
                                                (7280, 4899), (7509, 3239), (10, 2676), (6807, 2993), (5185, 3258), (3023, 1942)]}}
# print(datasets)

import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from typing import Dict, Tuple

def mst_bound_tsp(node_coords: List[Tuple[int, int]]) -> float:
    """Computes lower bound for TSP using a greedy approach."""
    n = len(node_coords)
    if n < 2:
        return 0.0  # No travel needed if there's less than 2 nodes
    
    total_distance = 0.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            # Calculate Euclidean distance between all pairs of nodes
            dist = np.linalg.norm(np.array(node_coords[i]) - np.array(node_coords[j]))
            total_distance += dist
    
    return total_distance

def mst_bound_dataset_tsp(instances: Dict[str, Dict]) -> float:
    l1_bounds = []
    for name in instances:
        instance = instances[name]
        # Each instance may have multiple sets of node coordinates, so we handle that here
        node_coords = instance["node_coords"]  # Get the node coordinates for TSP
        l1_bounds.append(mst_bound_tsp(node_coords))
    
    print(f"Mean MST lower bound: {np.mean(l1_bounds)}")
    return np.mean(l1_bounds)

# Example usage:
opt_tsp_bounds = {}
for name, dataset in datasets.items():
    print(f"Processing instance: {name}")
    # Check if the dataset contains multiple entries under each instance
    opt_tsp_bounds[name] = mst_bound_dataset_tsp(dataset)

# If running as a script
if __name__ == '__main__':
    print(datasets['ATT48'].keys())
