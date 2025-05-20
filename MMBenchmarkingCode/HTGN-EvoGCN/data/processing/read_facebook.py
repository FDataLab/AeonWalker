"""
http://konect.uni-koblenz.de/networks/FacebookWall
Size	46,952 vertices (users)
Volume	876,993 edges (wall posts)
Unique volume	274,086 edges (wall posts)
Average degree (overall)	37.357 edges / vertex
Fill	0.00012433 edges / vertex2
Maximum degree	2,696 edges

gap = 1
the latest [-35:] i.e. 36 timestamp
undirected without self-loop dynamic graphs and isolated nodes

we take out and save the last 36 graphs......
@ graph 0 # of nodes 1267 # of edges 967
@ graph 1 # of nodes 1559 # of edges 1274
@ graph 2 # of nodes 1587 # of edges 1214
@ graph 3 # of nodes 1725 # of edges 1378
@ graph 4 # of nodes 1928 # of edges 1582
@ graph 5 # of nodes 2204 # of edges 1913
@ graph 6 # of nodes 2848 # of edges 2683
@ graph 7 # of nodes 3649 # of edges 4075
@ graph 8 # of nodes 4312 # of edges 4824
@ graph 9 # of nodes 4598 # of edges 5321
@ graph 10 # of nodes 5074 # of edges 6142
@ graph 11 # of nodes 5197 # of edges 6043
@ graph 12 # of nodes 5983 # of edges 7510
@ graph 13 # of nodes 6347 # of edges 8330
@ graph 14 # of nodes 6552 # of edges 8356
@ graph 15 # of nodes 6878 # of edges 9049
@ graph 16 # of nodes 7489 # of edges 9949
@ graph 17 # of nodes 7971 # of edges 10355
@ graph 18 # of nodes 8241 # of edges 10820
@ graph 19 # of nodes 8508 # of edges 11050
@ graph 20 # of nodes 9032 # of edges 11477
@ graph 21 # of nodes 8705 # of edges 10487
@ graph 22 # of nodes 8617 # of edges 10213
@ graph 23 # of nodes 7993 # of edges 8851
@ graph 24 # of nodes 9099 # of edges 10856
@ graph 25 # of nodes 9359 # of edges 11177
@ graph 26 # of nodes 10087 # of edges 12152
@ graph 27 # of nodes 10455 # of edges 12276
@ graph 28 # of nodes 10497 # of edges 12154
@ graph 29 # of nodes 11539 # of edges 13342
@ graph 30 # of nodes 13692 # of edges 16568
@ graph 31 # of nodes 14854 # of edges 17539
@ graph 32 # of nodes 16537 # of edges 20337
@ graph 33 # of nodes 18847 # of edges 23634
@ graph 34 # of nodes 21743 # of edges 28168
@ graph 35 # of nodes 23740 # of edges 31742
time gap is 30
total edges: 180011
total nodes: 45435
"""

import networkx as nx
import datetime
import pickle
import pandas as pd
import torch

gap = 30
nodes_dict = {}

def save_nx_graph(nx_graph, dataset):
    path = '../input/{}/'.format(dataset) + 'adj_time_list.pickle'
    with open(path, 'wb') as f:
        pickle.dump(nx_graph, f, protocol=pickle.HIGHEST_PROTOCOL)  # the higher protocol, the smaller file
        print('saved!')

def save_edges(edges, dataset):
    path = '../input/processed/{}/{}.pt'.format(dataset, dataset)
    torch.save(edges, path)
    print('saved!')

def getID(node_id, nodes_dict):
    if node_id not in nodes_dict.keys():
        idx = len(nodes_dict)
        nodes_dict[node_id] = idx
    else:
        idx = nodes_dict[node_id]
    return idx, nodes_dict

if __name__ == '__main__':
    # --- load dataset ---
    dataset = 'fbw'
    df = pd.read_csv('../input/raw/{}'.format(dataset), sep=' | \t', names=['from','to','weight?','time'], header=None, comment='%')
    df['time'] = df['time'].apply(lambda x: int(datetime.datetime.utcfromtimestamp(x).strftime('%Y%m%d')))
    all_days = len(pd.unique(df['time']))
    print('# of all edges: ', len(df))
    print('all unique days: ', all_days)
    print(df.head(5))

    # --- check the time oder, if not ascending, resort it ---
    tmp = df['time'][0]
    for i in range(len(df['time'])):
        if df['time'][i] > tmp:
            tmp = df['time'][i]
        elif df['time'][i] == tmp:
            pass
        else:
            print('not ascending --> we resorted it')
            print(df[i-2:i+2])
            df.sort_values(by='time', ascending=True, inplace=True)
            df.reset_index(inplace=True)
            print(df[i-2: i+2])
            break
        if i == len(df['time'])-1:
            print('ALL checked --> ascending!!!')

    # --- generate graph and dyn_graphs ---
    cnt_graphs = 0
    graphs = []
    g = nx.Graph()
    tmp = df['time'][0]   # time is in ascending order
    for i in range(len(df['time'])):
        if tmp == df['time'][i]:        # if is in current day
            g.add_edge(str(df['from'][i]), str(df['to'][i]))
            if i == len(df['time'])-1:  # EOF ---
                cnt_graphs += 1
                # graphs.append(g.copy())  # ignore the last day
                print('processed graphs ', cnt_graphs, '/', all_days, 'ALL done......\n')
        elif tmp < df['time'][i]:       # if goes to next day
            cnt_graphs += 1
            if (cnt_graphs//gap) >= (all_days//gap-36) and cnt_graphs%gap == 0: # the last 50 graphs but ignore the latest 10 'and' the gap
                g.remove_edges_from(g.selfloop_edges())
                g.remove_nodes_from(list(nx.isolates(g)))
                graphs.append(g.copy())     # append previous g; for a part of graphs to reduce ROM
                g = nx.Graph()            # reset graph, based on the real-world application
            if cnt_graphs % 50 == 0:
                print('processed graphs ', cnt_graphs, '/', all_days)
            tmp = df['time'][i]
            g.add_edge(str(df['from'][i]), str(df['to'][i]))
        else:
            print('ERROR -- EXIT -- please double check if time is in ascending order!')
            exit(0)

    # --- take out and save part of graphs ----
    print('total graphs: ', len(graphs))
    print('we take out and save the last 36 graphs......')
    raw_graphs = graphs[1:]    # the last graph has some problem... we ignore it!

    # remap node index:
    G = nx.Graph() # whole graph, to count number of nodes and edges
    graphs = [] # graph list, to save remapped graphs
    nodes_dict = {} # node re-id index, to save mapped index
    edges_list = [] # edge_index lsit, sparse matrix
    for i, raw_graph in enumerate(raw_graphs):
        g = nx.Graph()
        for edge in raw_graph.edges:
            idx_i, nodes_dict = getID(edge[0], nodes_dict)
            idx_j, nodes_dict = getID(edge[1], nodes_dict)
            g.add_edge(idx_i, idx_j)
        graphs.append(g) # append to graph list
        edges_list.append(list(g.edges)) # append to edge list
        G.add_edges_from(g.edges) # append to the whole graphs
        print('@ graph', i, '# of nodes', len(graphs[i].nodes()), '# of edges', len(graphs[i].edges()))
    print('time gap is {}'.format(gap))
    print('total edges: {}'.format(G.number_of_edges()))
    print('total nodes: {}'.format(G.number_of_nodes()))
    save_edges(edges_list, dataset)
    print(max(nodes_dict.values())+1)