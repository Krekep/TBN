import numpy as np
import networkx as nx
from node import BayesNode


def inference(node: str, x: str, e: dict, samples: dict) -> dict:
    """
    Estimate the probability of a node given a set of evidences (P(node=x|e))

    Parameters
    ----------
    node: str
        node to estimate
    x: str
        value of node
    e: dict
        evidences
    samples: dict
        samples from the network

    Returns
    -------
        dict: probability distribution of node given e
    """
    # count samples that agree with e
    count = 0
    for s in samples.values():
        if all(s[key] == value for key, value in e.items()):
            count += 1

    # count samples that agree with e and node
    count_node = 0
    for s in samples.values():
        if all(s[key] == value for key, value in e.items()) and s[node] == x:
            count_node += 1

    # estimate probability
    if count == 0:
        p = 0
    else:
        p = count_node / count

    return {node: {x: p, 'evidence': e}}


def choice(values: list, probabilities: list) -> str:
    """
    Return a value from a list according to its probability distribution.

    Parameters
    ----------
    values: list
        List of values that each node can take.
    probabilities: list
        List of probabilities of each value.

    Returns
    -------
        str: a value from the list of values
    """
    if len(values) != len(probabilities):
        raise ValueError('Length of values and probabilities must be the same.')

    return np.random.choice(values, p=probabilities)


class BayesNetwork:
    def __init__(self, nodes: dict[str, BayesNode], values: list):
        """
        Bayes network data structure, undirected by default.

        Parameters
        ----------
        nodes: dict
            Dictionary of nodes. Each node is a BayesNode object
        values: list
            List of values that each node can take
        """
        self.nodes = nodes
        self.values = values
        G = nx.DiGraph(self.get_edges())

        if nx.is_directed_acyclic_graph(G):
            self.g = G
        else:
            raise ValueError('Network is not acyclic.')

        for node in self.get_nodes():
            if node not in self.g.nodes:
                self.g.add_node(node)

    def get_nodes(self) -> list:
        return list(self.nodes.keys())

    def get_edges(self) -> list:
        edges = []
        for key, node in self.nodes.items():
            if node.get_parents() is not None:
                for parent in node.get_parents():
                    edges.append((parent, key))
        return edges

    def sampling(self, n: int = 1, init: dict = None) -> dict:
        """
        Sample ancestors n times from the network

        Parameters
        ----------
        n: int
            number of times to sample
        init: dict
            initial state of the network

        Returns
        -------
            dict: n samples
        """
        if init is None:
            init = dict()

        samples = {}

        nodes = list(nx.topological_sort(self.g))

        for i in range(n):
            s = {}
            for node_name in nodes:
                node = self.nodes[node_name]

                if node_name in init:
                    s[node_name] = init[node_name]
                else:
                    cpt = node.get_cpt()
                    parents = node.get_parents()

                    if parents is None:
                        s[node_name] = choice(self.values, cpt['p'])
                    else:
                        s[node_name] = choice(self.values, cpt[tuple([s[parent] for parent in parents])])
            samples[i + 1] = s
        return samples

    def __str__(self):
        """
        Print on stdout the structure of the network.
        """
        res = "Network"

        for p, c in self.g.edges:
            res += ' ' + p + ' -> ' + c + '\n'

        for node in self.g.nodes:
            if self.g.out_degree(node) + self.g.in_degree(node) == 0:
                res += ' ' + node + '\n'
        res += '\n'

        return res

    def plot(self):
        """
        Plot the graph of the network.
        """

        # convert graph to graphviz
        graph = nx.nx_agraph.to_agraph(self.g)
        graph.layout('dot')

        # change color
        for node in graph.iternodes():
            node.attr['color'] = 'lightblue'
            node.attr['style'] = 'filled'

        return graph
