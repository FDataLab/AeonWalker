import os
import numpy as np
import torch
from torch_geometric.utils import train_test_split_edges
from torch_geometric.data import Data
import pickle
from make_edges_orign import mask_edges_det, mask_edges_prd, mask_edges_prd_new_by_marlin
from make_edges_new import get_edges, get_prediction_edges, get_new_prediction_edges


def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def prepare_dir(output_folder):
    mkdirs(output_folder)
    log_folder = mkdirs(output_folder)
    return log_folder


def load_vgrnn_dataset(dataset):
    assert dataset in ['enron10', 'dblp']  # using vgrnn dataset
    print('>> loading on vgrnn dataset')
    with open('../data/input/raw/{}/adj_time_list.pickle'.format(dataset), 'rb') as handle:
        adj_time_list = pickle.load(handle, encoding='iso-8859-1')
    print('>> generating edges,negative edges and new edges, wait for a while ...')
    data = {}
    edges, biedges = mask_edges_det(adj_time_list)  # list
    pedges, nedges = mask_edges_prd(adj_time_list)  # list
    new_pedges, new_nedges = mask_edges_prd_new_by_marlin(adj_time_list)  # list
    print('>> processing finished!')
    assert len(edges) == len(biedges) == len(pedges) == len(nedges) == len(new_nedges) == len(new_pedges)
    edge_index_list, pedges_list, nedges_list, new_nedges_list, new_pedges_list = [], [], [], [], []
    for t in range(len(biedges)):
        edge_index_list.append(torch.tensor(np.transpose(biedges[t]), dtype=torch.long))
        pedges_list.append(torch.tensor(np.transpose(pedges[t]), dtype=torch.long))
        nedges_list.append(torch.tensor(np.transpose(nedges[t]), dtype=torch.long))
        new_pedges_list.append(torch.tensor(np.transpose(new_pedges[t]), dtype=torch.long))
        new_nedges_list.append(torch.tensor(np.transpose(new_nedges[t]), dtype=torch.long))

    data['edge_index_list'] = edge_index_list
    data['pedges'], data['nedges'] = pedges_list, nedges_list
    data['new_pedges'], data['new_nedges'] = new_pedges_list, new_nedges_list  # list
    data['num_nodes'] = int(np.max(np.vstack(edges))) + 1

    data['time_length'] = len(edge_index_list)
    data['weights'] = None
    print('>> data: {}'.format(dataset))
    print('>> total length:{}'.format(len(edge_index_list)))
    print('>> number nodes: {}'.format(data['num_nodes']))
    return data


def load_new_dataset(dataset):
    print('>> loading on new dataset')
    data = {}
    rawfile = '../data/input/processed/{}/{}.pt'.format(dataset, dataset)
    edge_index_list = torch.load(rawfile)  # format: list:[[[1,2],[2,3],[3,4]]]
    undirected_edges = get_edges(edge_index_list)
    num_nodes = int(np.max(np.hstack(undirected_edges))) + 1
    pedges, nedges = get_prediction_edges(undirected_edges)  # list
    new_pedges, new_nedges = get_new_prediction_edges(undirected_edges, num_nodes)

    data['edge_index_list'] = undirected_edges
    data['pedges'], data['nedges'] = pedges, nedges
    data['new_pedges'], data['new_nedges'] = new_pedges, new_nedges  # list
    data['num_nodes'] = num_nodes
    data['time_length'] = len(edge_index_list)
    data['weights'] = None
    print('>> data: {}'.format(dataset))
    print('>> total length: {}'.format(len(edge_index_list)))
    print('>> number nodes: {}'.format(data['num_nodes']))
    return data


def load_vgrnn_dataset_det(dataset):
    assert dataset in ['enron10', 'dblp']  # using vgrnn dataset
    print('>> loading on vgrnn dataset')
    with open('../data/input/raw/{}/adj_time_list.pickle'.format(dataset), 'rb') as handle:
        adj_time_list = pickle.load(handle, encoding='iso-8859-1')
    print('>> generating edges, negative edges and new edges, wait for a while ...')
    data = {}
    edges, biedges = mask_edges_det(adj_time_list)  # list
    pedges, nedges = mask_edges_prd(adj_time_list)  # list
    new_pedges, new_nedges = mask_edges_prd_new_by_marlin(adj_time_list)  # list
    print('>> processing finished!')
    assert len(edges) == len(biedges) == len(pedges) == len(nedges) == len(new_nedges) == len(new_pedges)
    edge_index_list, pedges_list, nedges_list, new_nedges_list, new_pedges_list = [], [], [], [], []
    for t in range(len(biedges)):
        edge_index_list.append(torch.tensor(np.transpose(biedges[t]), dtype=torch.long))
        pedges_list.append(torch.tensor(np.transpose(pedges[t]), dtype=torch.long))
        nedges_list.append(torch.tensor(np.transpose(nedges[t]), dtype=torch.long))
        new_pedges_list.append(torch.tensor(np.transpose(new_pedges[t]), dtype=torch.long))
        new_nedges_list.append(torch.tensor(np.transpose(new_nedges[t]), dtype=torch.long))

    data['edge_index_list'] = edge_index_list
    data['pedges'], data['nedges'] = pedges_list, nedges_list
    data['new_pedges'], data['new_nedges'] = new_pedges_list, new_nedges_list  # list
    data['num_nodes'] = int(np.max(np.vstack(edges))) + 1

    data['time_length'] = len(edge_index_list)
    data['weights'] = None
    print('>> data: {}'.format(dataset))
    print('>> total length:{}'.format(len(edge_index_list)))
    print('>> number nodes: {}'.format(data['num_nodes']))
    return data


def load_new_dataset_det(dataset, time_bin_interval=36000):
    print('>> loading and binning new dataset with timestamps')

    edge_file = f'../../data/input/newds/{dataset}.edges'
    raw_edges = []

    # Load and parse edges with timestamps
    with open(edge_file, 'r') as f:
        for line in f:
            try:
                source, target, timestamp, edge_type = line.strip().split()
                raw_edges.append((int(source), int(target), int(timestamp)))
            except ValueError:
                print(f"Skipping invalid edge line: {line.strip()}")
                continue

    # Check if we loaded any edges
    if not raw_edges:
        raise ValueError(f"No edges found in dataset: {dataset}!")

    # Sort by timestamp to ensure correct ordering
    raw_edges.sort(key=lambda x: x[2])

    # Ensure the timestamps are in ascending order
    for i in range(1, len(raw_edges)):
        if raw_edges[i][2] < raw_edges[i - 1][2]:
            print(f"Warning: Timestamps are not in ascending order at index {i}.")
            break

    # Binning edges by timestamp
    start_time = raw_edges[0][2]
    end_time = raw_edges[-1][2]
    num_bins = (end_time - start_time) // time_bin_interval + 1

    binned_edges = [[] for _ in range(num_bins)]
    for src, tgt, ts in raw_edges:
        bin_index = (ts - start_time) // time_bin_interval
        if bin_index < num_bins:  # Ensure no out-of-bounds error
            binned_edges[bin_index].append((src, tgt))

    # Verify that at least one snapshot has data
    non_empty_bins = [bin for bin in binned_edges if bin]
    if not non_empty_bins:
        raise ValueError("No valid edges after binning. All bins are empty!")

    # Ensure correct number of nodes and valid edges
    all_edges = [e for snapshot in binned_edges for e in snapshot]
    num_nodes = int(np.max(np.array(all_edges))) + 1

    # Ensure that `get_edges` accepts List[List[Tuple[int, int]]]
    undirected_edges = get_edges(binned_edges)

    pedges, nedges = get_prediction_edges(undirected_edges)
    new_pedges, new_nedges = get_new_prediction_edges(undirected_edges, num_nodes)

    data = {
        'edge_index_list': undirected_edges,
        'pedges': pedges,
        'nedges': nedges,
        'new_pedges': new_pedges,
        'new_nedges': new_nedges,
        'num_nodes': num_nodes,
        'time_length': len(undirected_edges),
        'weights': None,
    }

    print(f'>> data: {dataset}')
    print(f'>> total snapshots: {len(undirected_edges)}')
    print(f'>> number of nodes: {num_nodes}')

    return data






def loader(dataset='enron10'):
    # if cached, load directly
    data_root = '../../data/input/cached/{}/'.format(dataset)
    filepath = mkdirs(data_root) + '{}.data'.format(dataset)
    
    if os.path.isfile(filepath):
        print(f'loading {dataset} directly')
        return torch.load(filepath)

    # if not cached, process and cache
    print('>> data is not found, processing ...')
    if dataset in ['enron10', 'dblp']:
        data = load_vgrnn_dataset(dataset)
    elif dataset in ['as733', 'fbw', 'HepPh30', 'disease']:
        data = load_new_dataset(dataset)
    elif dataset in ['contacts', 'facebook200', 'hypertext', 'infectious', 'neurips10']:  # New dataset
        data = load_new_dataset_det(dataset)  # Use the det function if it’s a new dataset requiring train/test split
    
    torch.save(data, filepath)
    print('saved!')
    return data

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='enron10')
    args = parser.parse_args()

    data = loader(args.dataset)
    print(">> Data loaded successfully.")