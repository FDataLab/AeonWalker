import pandas as pd
import networkx as nx
import torch
import os
from collections import defaultdict

def getID(node_id, nodes_dict):
    if node_id not in nodes_dict:
        idx = len(nodes_dict)
        nodes_dict[node_id] = idx
    return nodes_dict[node_id], nodes_dict

def load_edges(path, sep="\t"):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["from", "to", "time", "weight"], engine="python")
    return df

def process_dynamic_graph(df, gap=86400, keep_last_n=None, min_nodes=3, min_edges=3):
    df['time'] = df['time'].astype(int)
    df['bin'] = df['time'] // gap
    df.sort_values(by='time', inplace=True)

    graphs = []
    nodes_dict = {}
    grouped = df.groupby('bin')
    all_bins = sorted(grouped.groups.keys())

    if keep_last_n is not None:
        all_bins = all_bins[-keep_last_n:]

    for t in all_bins:
        g = nx.Graph()
        edges = grouped.get_group(t)[["from", "to"]].values
        for u, v in edges:
            if u == v:
                continue
            g.add_edge(str(u), str(v))
        g.remove_nodes_from(list(nx.isolates(g)))
        
        # Only keep graphs with enough structure
        if g.number_of_nodes() >= min_nodes and g.number_of_edges() >= min_edges:
            graphs.append(g)

    edges_list = []
    for g in graphs:
        edges = []
        for u, v in g.edges():
            uid, nodes_dict = getID(u, nodes_dict)
            vid, nodes_dict = getID(v, nodes_dict)
            edges.append([uid, vid])
        edges_list.append(edges)

    return edges_list, graphs, nodes_dict


def save_edges(edges_list, out_path):
    torch.save(edges_list, out_path)
    print(f"Saved edge list to {out_path}")

def process_dataset(name, input_root="../input/newds", output_root="../input/processed", target_bins=20):
    input_path = os.path.join(input_root, f"{name}.edges")
    output_path = os.path.join(output_root, name, f"{name}.pt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = load_edges(input_path, sep="\t")
    min_time, max_time = df['time'].min(), df['time'].max()
    gap = max(1, (max_time - min_time) // target_bins)
    print(f"\n=== Processing {name} ===")
    print(f"Dynamic gap: {gap} seconds")

    edges_list, graphs, nodes_dict = process_dynamic_graph(df, gap=gap, keep_last_n=target_bins)

    for i, g in enumerate(graphs):
        print(f"@ graph {i} | nodes: {g.number_of_nodes()} | edges: {g.number_of_edges()}")

    print(f"Total unique nodes (after remapping): {len(nodes_dict)}")
    save_edges(edges_list, output_path)

def main():
    datasets = ["facebook200", "contacts", "hypertext", "infectious", "neurips10"]
    for name in datasets:
        process_dataset(name)

if __name__ == "__main__":
    main()