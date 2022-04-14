''' Written by Anthony Gitter '''

import argparse
from itertools import islice
import networkx as nx

def parse_nodes(node_file):
    ''' Parse a list of sources or targets and return a set '''
    with open(node_file) as node_f:
        lines = node_f.readlines()
        nodes = set(map(str.strip, lines))
    return nodes


def construct_digraph(edges_file):
    ''' Parse a list of weighted undirected edges.  Construct a weighted
    directed graph in which an undirected edge is represented with a pair of
    directed edges.  Use the specified weight as the edge weight and a default
    capacity of 1.
    '''
    digraph = nx.DiGraph()
    default_capacity = 1.0
    
    with open(edges_file) as edges_f:
        for line in edges_f:
            tokens = line.strip().split()
            node1 = tokens[0]
            node2 = tokens[1]
            w = float(tokens[2])
            
            # Add the pair of directed edges with a weight and capacity
            digraph.add_edge(node1, node2, weight=w, capacity=default_capacity)

    return digraph


def print_graph(graph):
    ''' Print the edges in a graph '''
    print('\n'.join(sorted(map(str, graph.edges(data=True)))))


def add_sources_targets(digraph, sources, targets, flow):
    ''' Similar to ResponseNet, add an artificial source node that is connected
    to the real source nodes with directed edges.  Unlike ResponseNet, these
    directed edges should have weight of 0 and Infinite capacity.  Also add an
    artificial target node that has directed edges from the real target nodes
    with the same weights and capacities as the source node edges.  The new
    nodes must be named "source" and "target".
    '''
    default_weight = 0
    default_capacity = float('inf')
    
    # Add the source edges and target edges with the default weight and capacity    
    # Add the source and target nodes
    digraph.add_node('source', demand=-flow)

    # add demand to the source node
    digraph.add_node('target', demand=flow)   

    # Add the source edges
    for source in sources:
        digraph.add_edge('source', source, weight=default_weight, capacity=default_capacity)
    
    # Add the target edges
    for target in targets:
        digraph.add_edge(target, 'target', weight=default_weight, capacity=default_capacity)


def remove_zero_flow(flow_dict):
    ''' Removes edges with flow of 0 from the flow dictionary returned by
    networkx.min_cost_flow
    '''
    nonzero_flow_dict = dict()
    for node1, neighbors in flow_dict.items():
        nonzero_neighbors = dict()
        for node2, flow in neighbors.items():
            if flow > 0:
                nonzero_neighbors[node2] = flow
        if len(nonzero_neighbors) > 0:
            nonzero_flow_dict[node1] = nonzero_neighbors

    return nonzero_flow_dict


def flow_dict_to_list(flow_dict):
    ''' Convert a flow dictionary from networkx.min_cost_flow into a list
    of directed edges with the flow.  Edges are represented as tuples.
    '''
    edge_list = list()
    for node1, neighbors in flow_dict.items():
        for node2, flow in neighbors.items():
            edge_list.append((node1, node2, flow))

    # Sort edges by node1 and break ties with node2
    return sorted(sorted(edge_list, key=lambda edge: edge[1]), key=lambda edge: edge[0])


def min_cost_flow(digraph, flow, output):
    ''' Use the min cost flow algorithm to distribute the specified amount
    of flow from sources to targets.  The artificial source should have
    demand = -flow and the target should have demand = flow.  output is the
    filename of the output file.  The graph should have artificial nodes
    named "source" and "target".
    '''
    assert 'source' in digraph
    assert 'target' in digraph

    # the artificial source should have demand = -flow and the target should have demand = flow
    # set up and run the min cost flow algorithm, storing the results in a variable called flow_dict
    # obtain the cost of the flow using networkx's cost_of_flow, storing the results in a variable called cost

    # TODO: set up and run the min cost flow algorithm, storing the results
    # in a variable called flow_dict

    # TODO: obtain the cost of the flow using networkx's cost_of_flow,
    # storing the results in a variable called cost
    flow_dict, cost = nx.capacity_scaling(digraph)

    flow_list = flow_dict_to_list(flow_dict)
    print('The transmitted flow is {:.3g} and the cost is {:.3g}'.format(flow, cost))

    with open(output, 'w') as out_f:
        for flow_edge in flow_list:
            out_f.write('{}\n'.format(flow_edge))


def path_cost(digraph, path):
    ''' Given a directed graph and a path returned by k_shortest_paths,
    compute the path cost.  Assume the path is in the graph.  Returns the cost.
    '''
    total = 0
    for i in range(len(path) - 1):
        cost = digraph[path[i]][path[i+1]]['weight']
        total += cost
    return total


def k_shortest_paths(digraph, k, output):
    ''' Use the k-shortest paths algorithm to find the lowest weight source-
    target paths. output is the filename of the output file.  The graph should
    have artificial nodes named "source" and "target".
    '''
    assert 'source' in digraph
    assert 'target' in digraph

    # TODO: obtain the k shortest weighted paths and store then in a variable
    # called paths.  Use the example code at
    # http://networkx.readthedocs.io/en/stable/reference/generated/networkx.algorithms.simple_paths.shortest_simple_paths.html
    # to obtain a list of the k shortest paths from the shortest_simple_paths
    # function   
    paths = nx.shortest_simple_paths(digraph, 'source', 'target', weight='weight')

    print('Found {} shortest weighted paths'.format(len(paths)))
    
    with open(output, 'w') as out_f:
        for path in paths:
            cost = path_cost(digraph, path)
            out_f.write('{}, cost={:.3g}\n'.format(path, cost))


def main(args):
    ''' Parse a weighted edge list, source list, and target list.  Run
    min cost flow or k-shortest paths on the graph to find source-target
    paths.  Write the solutions to a file.
    '''
    k = args.k
    flow = args.flow   
    assert (k is None) ^ (flow is None), 'exactly one of k and flow can be provided'
    
    sources = parse_nodes(args.sources)
    print('Parsed {} source nodes'.format(len(sources)))

    targets = parse_nodes(args.targets)
    print('Parsed {} target nodes'.format(len(targets)))
    
    digraph = construct_digraph(args.edges)
    print('Parsed a graph with {} nodes and {} directed edges'.format(digraph.order(), digraph.size()))

    # Uncomment this line to print the graph data structure to stdout
    print_graph(digraph)

    add_sources_targets(digraph, sources, targets, flow)
    print('The graph has {} nodes and {} directed edges after adding the ' \
        'artificial source and target'.format(digraph.order(), digraph.size()))
    
    if k is None:
        min_cost_flow(digraph, flow, args.out)
    else:
        k_shortest_paths(digraph, k, args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--edges',
                        help='edge file path',
                        type=str,
                        required=True)
    parser.add_argument('--sources',
                        help='source node file path',
                        type=str,
                        required=True)
    parser.add_argument('--targets',
                        help='target node file path',
                        type=str,
                        required=True)
    parser.add_argument('--flow',
                        help='the flow through the graph, must set the flow ' \
                        'or k argument but not both',
                        type=float)
    parser.add_argument('--k',
                        help='the number of shortest paths to find, must ' \
                        'the k or flow argument but not both',
                        type=int)
    parser.add_argument('--out',
                        help='output file path',
                        type=str,
                        required=True)


    args = parser.parse_args()
    main(args)
