import pyagrum as gum
import itertools
import numpy as np
import pandas as pd

class FCI_totally_gum:

    def __init__(self, df, alpha=0.05):
        self.variables = list(df.columns)
        self.n = len(self.variables)
        self.graph = gum.PDAG(gum.PDAG.completeGraph(self.n))
        self.alpha = alpha

        # Create BNLearner
        self.learner = gum.BNLearner(df)

        # Separator sets
        self.separators = dict()
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self.separators[(i, j)] = set()

    def isIndependent(self, X, Y, Z):
        _, p = self.learner.G2(X, Y, Z)
        return p > self.alpha

    def skeleton(self):
        d = 0
        flag = True
        while flag:
            edges = self.graph.edges()
            flag = False
            for (i, j) in edges:
                if self.graph.existsEdge(i, j):
                    adj_i = self.graph.boundary(i)
                    adj_j = self.graph.boundary(j)
                    check_list = [(j, adj_i), (i, adj_j)]
                    for (x, adj) in check_list:
                        if len(adj) > d:
                            flag = True
                            for Z in itertools.combinations([k for k in adj if k != x], d):
                                if self.isIndependent(self.variables[i], self.variables[j], [self.variables[k] for k in Z]):
                                    self.separators[(i, j)] = self.separators[(i, j)].union(set(Z))
                                    self.separators[(j, i)] = self.separators[(j, i)].union(set(Z))
                                    self.graph.eraseEdge(i, j)
                                    break
            d += 1
        
    def rule0(self, i, j, k):
        if self.graph.existsEdge(i, j) and self.graph.existsEdge(j, k) and not self.graph.existsEdge(i, k):
            if j not in self.separators[(i, k)]:
                self.graph.eraseEdge(i, j)
                self.graph.eraseEdge(k, j)
                self.graph.addArc(i, j)
                self.graph.addArc(k, j)

def Short_FCI(df, alpha=0.05):
    fci = FCI_totally_gum(df, alpha=alpha)
    fci.skeleton()
    # triplets = list(itertools.permutations(range(fci.n), r=3))
    # for t in triplets:
    #     fci.rule0(t[0], t[1], t[2])
    mr = gum.MeekRules()
    fci.graph = mr.propagate(fci.graph)
    #fci.graph = mr.propagateToDAG(fci.graph)
    return fci