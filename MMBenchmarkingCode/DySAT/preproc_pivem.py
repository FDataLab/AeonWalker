# import os
# import json
# import numpy as np
# import networkx as nx
# from scipy.sparse import csr_matrix
# import pickle

# INPUT_DIR = "../HTGN/data/input/newds"
# OUTPUT_ROOT = "data/pivem"
# LOGS_DIR = "logs/DySAT_default"
# TARGET_BINS = 20
# MIN_NODES = 3
# MIN_EDGES = 3

# if not os.path.exists(OUTPUT_ROOT):
#     os.makedirs(OUTPUT_ROOT)
# if not os.path.exists(LOGS_DIR):
#     os.makedirs(LOGS_DIR)
    
# def remap_slices_graphs(slices_links, slices_features):
#     # 1. Collect all unique nodes globally
#     all_nodes = set()
#     for G in slices_links:
#         all_nodes.update(G.nodes())
#     all_nodes = sorted(all_nodes)
    
#     # 2. Map from global node to index [0 .. N-1]
#     node_map = {node: idx for idx, node in enumerate(all_nodes)}
#     num_global_nodes = len(all_nodes)
    
#     remapped_graphs = []
#     remapped_features = []
    
#     for G, feats in zip(slices_links, slices_features):
#         # 3. Build new graph with remapped nodes
#         H = nx.MultiGraph()
#         # Add *all* global nodes (even if missing in snapshot) as isolates
#         H.add_nodes_from(range(num_global_nodes))
        
#         # Add edges remapped by node_map
#         for u, v, data in G.edges(data=True):
#             H.add_edge(node_map[u], node_map[v], **data)
        
#         remapped_graphs.append(H)
        
#         # 4. Build identity features matrix of size num_global_nodes
#         feat_mat = csr_matrix(np.identity(num_global_nodes))
#         remapped_features.append(feat_mat)
    
#     return remapped_graphs, remapped_features

      



# def read_edges_file(file_path):
#     edges = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) < 3:
#                 continue
#             u = int(parts[0])
#             v = int(parts[1])
#             t = int(parts[2])
#             edges.append((u, v, t))
#     return edges

# def bin_edges(edges, gap, min_time):
#     bins = {}
#     for u, v, t in edges:
#         b = (t - min_time) // gap
#         if b not in bins:
#             bins[b] = []
#         bins[b].append((u, v))
#     return bins

# def generate_flags(dataset_name, num_snapshots):
#     return {
#         "dataset": dataset_name,
#         "base_model": "DySAT",
#         "model": "default",
#         "structural_layer_config": "128",
#         "temporal_layer_config": "128",
#         "structural_head_config": "16,8,8",
#         "temporal_head_config": "16",
#         "position_ffn": "True",
#         "use_residual": "False",
#         "featureless": "True",
#         "run_parallel": "False",
#         "learning_rate": 0.001,
#         "batch_size": 512,
#         "epochs": 200,
#         "weight_decay": 0.0005,
#         "temporal_drop": 0.5,
#         "spatial_drop": 0.1,
#         "neg_sample_size": 10,
#         "neg_weight": 1.0,
#         "max_gradient_norm": 1.0,
#         "GPU_ID": 0,
#         "window": -1,
#         "walk_len": 20,
#         "min_time": 2,
#         "max_time": num_snapshots,
#         "val_freq": 1,
#         "test_freq": 1
#     }

# for fname in sorted(os.listdir(INPUT_DIR)):
#     if not fname.endswith(".edges"):
#         continue

#     dataset_name = fname.replace(".edges", "")
#     file_path = os.path.join(INPUT_DIR, fname)
#     output_dir = os.path.join(OUTPUT_ROOT, dataset_name)
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     print("Processing:", fname)

#     if fname != "contacts.edges":
#         continue
#     edges = read_edges_file(file_path)
#     if len(edges) == 0:
#         continue

#     timestamps = [t for _, _, t in edges]
#     min_time = min(timestamps)
#     max_time = max(timestamps)
#     gap = max(1, (max_time - min_time) // TARGET_BINS)

#     print("Time gap for binning:", gap)

#     binned = bin_edges(edges, gap, min_time)
#     valid_bins = sorted(binned.keys())

#     slices_links = []
#     slices_features = []

#     prev_nodes = set()

#     for idx, b in enumerate(valid_bins):
#         G = nx.MultiGraph()
#         for u, v in binned[b]:
#             if u != v:
#                 G.add_edge(u, v)
#         print(G.edges())
#         G.remove_nodes_from(list(nx.isolates(G)))

#         if G.number_of_nodes() >= MIN_NODES and G.number_of_edges() >= MIN_EDGES:
#             # Add nodes from previous snapshot to maintain continuity (optional)
#             # Uncomment if you want to keep all nodes over time (like reference)
#             G.add_nodes_from(prev_nodes)
#             prev_nodes = set(G.nodes())

#             slices_links.append(G)

#             # Prepare feature matrix: identity matrix of size num_nodes
#             num_nodes = G.number_of_nodes()
#             features = csr_matrix(np.identity(num_nodes))
#             slices_features.append(features)

#             print(len(slices_links))
#             print("Num nodes:", num_nodes)
#             print("Num edges:", G.number_of_edges())

#             # after building slices_links (list of nx.MultiGraph) and slices_features (list of csr_matrix)

#             print(len(slices_links))

#             slices_links_remap, slices_features_remap = remap_slices_graphs(slices_links, slices_features)

#             # Save remapped graphs and features as before:
#             np.savez(os.path.join(output_dir, "graphs.npz"), graph=np.array(slices_links_remap, dtype=object))
#             np.savez(os.path.join(output_dir, "features.npz"), feats=np.array(slices_features_remap, dtype=object))




#     flags = generate_flags(dataset_name, len(slices_links))    
#     flags_path = os.path.join(LOGS_DIR, "flags_%s.json" % dataset_name)
#     with open(flags_path, "w") as f:
#         json.dump(flags, f, indent=4)

#     print("Saved to", output_dir, "and", flags_path)

# print("All .edges files processed.")


# import json
# import os
# import numpy as np
# import networkx as nx
# from datetime import timedelta
# from collections import defaultdict
# from scipy.sparse import csr_matrix

# # Load edges
# def load_edges(filepath):
#     edges = []
#     timestamps = []
#     with open(filepath, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) < 3:
#                 continue
#             u, v, t = int(parts[0]), int(parts[1]), int(parts[2])
#             edges.append((u, v, t))
#             timestamps.append(t)
#     return edges, timestamps

# # Parameters
# SLICE_SECONDS = 60 * 60 * 24 * 30 * 2  # 2 months in seconds
# EDGE_PATH = '../HTGN/data/input/newds/contacts.edges'  # replace with your .edges file path

# edges, timestamps = load_edges(EDGE_PATH)
# min_time, max_time = min(timestamps), max(timestamps)

# print("Min time:", min_time, "Max time:", max_time)
# print("# interactions:", len(edges))

# # Sort edges by timestamp
# edges.sort(key=lambda x: x[2])

# # Slice edges
# slices_links = defaultdict(lambda: nx.MultiGraph())
# for u, v, t in edges:
#     slice_id = (t - min_time) // SLICE_SECONDS
#     if u not in slices_links[slice_id]:
#         slices_links[slice_id].add_node(u)
#     if v not in slices_links[slice_id]:
#         slices_links[slice_id].add_node(v)
#     slices_links[slice_id].add_edge(u, v, timestamp=t)

# # Assign identity features
# all_nodes = sorted(set(u for G in slices_links.values() for u in G.nodes()))
# node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
# N = len(all_nodes)
# identity = np.identity(N)

# slices_features = {}
# for sid in slices_links:
#     features = {}
#     for node in slices_links[sid].nodes():
#         features[node] = identity[node_to_idx[node]]
#     slices_features[sid] = features

# # Remap function
# def remap(slices_graph, slices_features):
#     all_nodes = list(set(n for G in slices_graph.values() for n in G.nodes()))
#     node_idx = {n: i for i, n in enumerate(all_nodes)}
#     idx_node = {i: n for n, i in node_idx.items()}

#     slices_graph_remap = []
#     slices_features_remap = []

#     for sid in sorted(slices_graph.keys()):
#         G = nx.MultiGraph()
#         for n in slices_graph[sid].nodes():
#             G.add_node(node_idx[n])
#         for u, v, data in slices_graph[sid].edges(data=True):
#             G.add_edge(node_idx[u], node_idx[v], timestamp=data['timestamp'])
#         slices_graph_remap.append(G)

#         features = []
#         for i in G.nodes():
#             features.append(slices_features[sid][idx_node[i]])
#         features = csr_matrix(np.array(features))
#         slices_features_remap.append(features)

#     return slices_graph_remap, slices_features_remap

# # Apply remap
# slices_links_remap, slices_features_remap = remap(slices_links, slices_features)

# # Save
# os.makedirs('data/contacts')
# np.savez('data/contacts/graphs.npz',  graph=np.array(slices_links_remap, dtype=object))
# np.savez('data/contacts/features.npz', feats=slices_features_remap)


# # Generate flags dynamically
# flags = {
#     "structural_head_config": "16,8,8",
#     "temporal_layer_config": "128",
#     "val_freq": 1,
#     "dataset": "contacts",
#     "epochs": 200,
#     "walk_len": 20,
#     "min_time": int(min(slices_links.keys())),
#     "weight_decay": 0.0005,
#     "neg_sample_size": 10,
#     "temporal_drop": 0.5,
#     "GPU_ID": 0,
#     "use_residual": "False",
#     "structural_layer_config": "128",
#     "window": -1,
#     "spatial_drop": 0.1,
#     "position_ffn": "True",
#     "learning_rate": 0.001,
#     "batch_size": 512,
#     "max_gradient_norm": 1.0,
#     "featureless": "True",
#     "test_freq": 1,
#     "run_parallel": "False",
#     "temporal_head_config": "16",
#     "base_model": "DySAT",
#     "max_time": int(max(slices_links.keys())),
#     "model": "default",
#     "neg_weight": 1.0
# }

# with open("logs/DySAT_default/flags_contacts.json", "w") as f:
#     json.dump(flags, f, indent=2)
# print("flags saved")


import os
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
from collections import defaultdict
from scipy.sparse import csr_matrix

# -------------------------
# Config
# -------------------------
EDGE_PATH = '../HTGN/data/input/newds/contacts.edges'  # Update if needed
OUTPUT_PATH = 'data/contacts/graphs.npz'
SLICE_MONTHS = 2  # Time granularity for graph snapshots

# -------------------------
# Load raw edges
# -------------------------
def load_edges(filepath):
    edges = []
    timestamps = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            u, v, t = int(parts[0]), int(parts[1]), int(parts[2])
            dt = datetime.utcfromtimestamp(t)
            edges.append((u, v, dt))
            timestamps.append(dt)
    return edges, timestamps

edges, timestamps = load_edges(EDGE_PATH)
min_time, max_time = min(timestamps), max(timestamps)
START_DATE = min_time + timedelta(days=200)
MAX_DATE = max_time - timedelta(days=200)

print("Start date:", START_DATE, "| Max date:", MAX_DATE)
print("Min time:", min_time, "Max time:", max_time)
print("# interactions:", len(edges))

# -------------------------
# Slice edges into graphs
# -------------------------
edges.sort(key=lambda x: x[2])
slices_links = defaultdict(lambda: nx.MultiGraph())

for u, v, t in edges:
    if t > MAX_DATE:
        months_diff = (MAX_DATE - START_DATE).days // 30
    else:
        months_diff = (t - START_DATE).days // 30

    slice_id = max(months_diff // SLICE_MONTHS, 0)
    prev_slice_id = slice_id - 1

    if slice_id > 0 and slice_id not in slices_links:
        slices_links[slice_id] = nx.MultiGraph()
        if prev_slice_id in slices_links:
            slices_links[slice_id].add_nodes_from(slices_links[prev_slice_id].nodes(data=True))

    slices_links[slice_id].add_edge(u, v, date=t)

# -------------------------
# Identity features
# -------------------------
max_nodes = max(len(g.nodes()) for g in slices_links.values())
temp_identity = np.identity(max_nodes)

slices_features = defaultdict(dict)
for sid in slices_links:
    for idx, node in enumerate(slices_links[sid].nodes()):
        slices_features[sid][node] = temp_identity[idx]

# -------------------------
# Remap node IDs across slices
# -------------------------
def remap(slices_graph, slices_features):
    all_nodes = []
    for slice_id in slices_graph:
        assert len(slices_graph[slice_id].nodes()) == len(slices_features[slice_id])
        all_nodes.extend(slices_graph[slice_id].nodes())

    all_nodes = sorted(set(all_nodes))
    node_idx = {node: idx for idx, node in enumerate(all_nodes)}
    idx_node = [node for node in all_nodes]

    slices_graph_remap = []
    slices_features_remap = []

    for slice_id in sorted(slices_graph.keys()):
        G_multi = slices_graph[slice_id]
        G = nx.Graph()
        for u, v, data in G_multi.edges(data=True):
            G.add_edge(node_idx[u], node_idx[v], date=data['date'])

        # Ensure all nodes appear
        for n in G_multi.nodes():
            G.add_node(node_idx[n])

        slices_graph_remap.append(G)

        features_remap = []
        for n in G.nodes():
            features_remap.append(slices_features[slice_id][idx_node[n]])
        features_remap = csr_matrix(np.squeeze(np.array(features_remap)))
        slices_features_remap.append(features_remap)

    return slices_graph_remap, slices_features_remap

slices_links_remap, slices_features_remap = remap(slices_links, slices_features)

# -------------------------
# Final save
# -------------------------
os.makedirs(os.path.dirname(OUTPUT_PATH))
np.savez(OUTPUT_PATH, graph=np.array(slices_links_remap, dtype=object))
# Optional: Save features
# np.savez('data/contacts/features.npz', feats=slices_features_remap)

print("Saved:", OUTPUT_PATH)
