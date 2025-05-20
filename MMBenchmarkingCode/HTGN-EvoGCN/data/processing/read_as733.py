"""
https://snap.stanford.edu/data/as-733.html

The dataset contains 733 daily instances which span an interval of 785 days from November 8 1997 to January 2 2000

max nodes 6467; max edges 13895
gap = 1
the latest [-26,-2] i.e. 24 step except the last one
undirected without self-loop dynamic graphs and isolated nodes

@ graph 0 # of nodes 767 # of edges 1734 1734
@ graph 1 # of nodes 1470 # of edges 3131 3131
@ graph 2 # of nodes 1486 # of edges 3172 3172
@ graph 3 # of nodes 1477 # of edges 3142 3142
@ graph 4 # of nodes 1476 # of edges 3132 3132
@ graph 5 # of nodes 2071 # of edges 4233 4233
@ graph 6 # of nodes 2086 # of edges 4283 4283
@ graph 7 # of nodes 2062 # of edges 4233 4233
@ graph 8 # of nodes 2090 # of edges 4289 4289
@ graph 9 # of nodes 2070 # of edges 4240 4240
@ graph 10 # of nodes 2073 # of edges 4241 4241
@ graph 11 # of nodes 2102 # of edges 4307 4307
@ graph 12 # of nodes 2067 # of edges 4218 4218
@ graph 13 # of nodes 2080 # of edges 4271 4271
@ graph 14 # of nodes 2095 # of edges 4291 4291
@ graph 15 # of nodes 2083 # of edges 4263 4263
@ graph 16 # of nodes 2058 # of edges 4227 4227
@ graph 17 # of nodes 2063 # of edges 4233 4233
@ graph 18 # of nodes 2092 # of edges 4285 4285
@ graph 19 # of nodes 2089 # of edges 4270 4270
@ graph 20 # of nodes 2122 # of edges 4334 4334
@ graph 21 # of nodes 2120 # of edges 4314 4314
@ graph 22 # of nodes 2132 # of edges 4347 4347
@ graph 23 # of nodes 2107 # of edges 4303 4303
"""

import numpy as np
import networkx as nx
import datetime
import os
import pickle
import torch
root = '../input/raw/as733/'

def getID(node_id, nodes_dict):
    if node_id not in nodes_dict.keys():
        idx = len(nodes_dict)
        nodes_dict[node_id] = idx
    else:
        idx = nodes_dict[node_id]
    return idx, nodes_dict

def detect_exentence_file(date):
    file_location = root + date_2_string(date)
    print(file_location)
    return os.path.isfile(file_location)


def date_2_string(date):  # change the date format to
    year = date.year
    month = date.month
    day = date.day
    string_date = str(year * 10000 + month * 100 + day)
    string_date = 'as' + string_date + '.txt'
    return string_date


def string_2_date(date_str):
    year = date_str[0:4]
    month = date_str[4:6]
    day = date_str[6:8]
    date = datetime.datetime(year=int(year), month=int(month), day=int(day))
    return date


def generate_dynamic_graph(start_date='19991009', time_step_number=10, stop_at_irregular_interval=False):
    """
    earlist date is 19971108
    last date is 20000102

    the form of input is a string of date such as '19991015'. Note that I did not implement any date check

    :param start_date:
    :param time_step_number:
    :param stop_at_irregular_interval:
    :return:
    """
    user_chosen_date = string_2_date(start_date)
    dyanmic_netowks = []
    last_available_date = datetime.datetime(2000, 1, 2)

    remaining_graph = time_step_number
    while (remaining_graph > 0):
        if (user_chosen_date - last_available_date).days > 0:
            print("no more file available, stop generate more file")
            break
        elif detect_exentence_file(user_chosen_date) == True:

            remaining_graph -= 1
            file_name = date_2_string(user_chosen_date)
            graph = generate_a_graph(file_name)
            dyanmic_netowks.append(graph.copy())
            print(remaining_graph)
            # print(len(graph.nodes())," graph node number")
            user_chosen_date += datetime.timedelta(days=1)
        elif stop_at_irregular_interval == False:
            print("file does not exit at ", user_chosen_date, "date skipped")
            user_chosen_date += datetime.timedelta(days=1)
        else:
            print("file does not exit at ", user_chosen_date, "stop generate more network")
            break
    print("dynamic network length:", len(dyanmic_netowks))
    return dyanmic_netowks


def generate_a_graph(file_name):
    graph_data = np.genfromtxt(root + file_name, dtype=str)
    graph = nx.Graph()

    for i in range(len(graph_data)):
        graph.add_edge(str(graph_data[i][0]), str(graph_data[i][1]))

    graph.remove_edges_from(graph.selfloop_edges())
    graph.remove_nodes_from(list(nx.isolates(graph)))
    return graph

def save_edges(edges, dataset):
    path = '../input/processed/{}/{}.pt'.format(dataset, dataset)
    torch.save(edges, path)
    print('saved!')

def save_nx_graph(nx_graph, path='nx_graph_temp.dataset'):
    # 1.save graph data
    path = '../input/cached/as733/'
    with open(path + 'as733.graph', 'wb') as f:
        pickle.dump(nx_graph, f, protocol=pickle.HIGHEST_PROTOCOL)  # the higher protocol, the smaller file

    # 2. save sparse adj matrix
    with open(path + 'adj_time_list.pickle', 'wb') as f:
        adjs_list = []
        G = nx.Graph()
        for graph in nx_graph:
            adjs_list.append(nx.adjacency_matrix(graph))
            G.add_edges_from(graph.edges)  # append to the whole graph
        pickle.dump(adjs_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 3.test graph data
    with open(path + 'as733.graph', 'rb') as f:
        nx_graph_reload = pickle.load(f)
    try:
        print('Check if it is correctly dumped and loaded: ', nx_graph_reload.edges() == nx_graph.edges(),
              ' It contains only ONE graph')
    except:
        for i in range(len(nx_graph)):
            print('Check if it is correctly dumped and loaded: ', nx_graph_reload[i].edges() == nx_graph[i].edges(),
                  ' for Graph ', i)
    print('total nodes: {}'.format(G.number_of_nodes()))
    print('total edges: {}'.format(G.number_of_edges()))


if __name__ == '__main__':
    """
    start_date = datetime.datetime(1997,11,8)
    last_date = datetime.datetime(2000,1,2)
    days = (last_date-start_date).days
    print(days)
    #785 internal however which 733 days are missing
    print(date_2_string(start_date))
    detect_exentence_file()
    """
    graphs = generate_dynamic_graph(start_date='19991013', time_step_number=51, stop_at_irregular_interval=False)
    graphs = graphs[-30:]  # the last graph has some problem... we ignore it!
    # save_nx_graph(nx_graph=graphs, path='AS733.pkl')

    # remap node id
    node_dict={}
    edges_list = []
    for graph in graphs:
        edges = []
        for edge in graph.edges:
            edge_i, node_dict = getID(edge[0], node_dict)
            edge_j, node_dict = getID(edge[1], node_dict)
            edges.append([edge_i, edge_j])
        edges_list.append(edges)

    # remove the comment and save the dataset
    # save_edges(edges_list, dataset='as733')
    G=nx.Graph()
    for i in range(len(graphs)):
        print('@ graph', i, '# of nodes', len(graphs[i].nodes()), '# of edges', len(graphs[i].edges()), len(edges_list[i]))
        G.add_edges_from(graphs[i].edges)
    print('total edges: {}'.format(G.number_of_edges()))
    print('total nodes: {}'.format(G.number_of_nodes()))