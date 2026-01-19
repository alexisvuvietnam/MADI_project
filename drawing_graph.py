import pyagrum as gum
from pyagrum import PDAG
from FCI_family import exist_edge, FCI_PC, FCI_Zhang, FCI_ETHZ, AFCI
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class FinalGraph:
    variables: list
    edges: set
    arcs: set

def create_instance(df, func=FCI_Zhang):
    variables = list(df.columns)
    graph = func(df)
    _, n = df.shape
    edges = set()
    arcs = set()
    attributes = ["-", "o", ">"]
    for i in range(n):
        for j in range(i + 1, n):
            if exist_edge(graph, i, j):
                if graph[i, j] == graph[j, i]:
                    edges.add((i, j))
                elif attributes.index(graph[i, j]) > attributes.index(graph[j, i]) :
                    arcs.add((i, j))
                else:
                    arcs.add((j, i))
    return FinalGraph(variables, edges, arcs)

def create_graph(instance):
    gum_graph = gum.PDAG()
    for i in range(len(instance.variables)):
        gum_graph.addNodeWithId(i)
    for (i, j) in instance.edges:
        gum_graph.addEdge(i, j)
    for (i, j) in instance.arcs:
        gum_graph.addArc(i, j)
    return gum_graph