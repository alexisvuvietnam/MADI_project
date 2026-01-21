import pyagrum as gum
from dataclasses import dataclass
import itertools
import numpy as np
import pandas as pd

# FCI CLASS

class FCI:
    arrow_attributes = ["-", "o", ">"]

    def __init__(self, df, alpha=0.05):
        self.variables = list(df.columns)
        self.n = len(self.variables)
        self.alpha = alpha

        # Create BNLearner
        self.learner = gum.BNLearner(df)
        
        # Implement initial graph
        self.matrix = np.full((self.n, self.n), "o")
        for i in range(self.n):
            self.matrix[i, i] = " "

        # Separator sets
        self.separators = dict()
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self.separators[(i, j)] = list()
    
    def exist_edge(self, i, j):
        if i == j: return False
        return self.matrix[i, j] != " " and self.matrix[j, i] != " "
    
    def remove_edge(self, i, j):
        if self.exist_edge(i, j):
            self.matrix[i, j] = " "
            self.matrix[j, i] = " "
    
    def list_edges(self):
        L = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.exist_edge(i, j):
                    L.append((i, j))
        return L
    
    def list_adjacents(self, i):
        return set([j for j in range(self.n) if self.exist_edge(i, j)])
    
    def num_adjacents(self, i):
        return len(self.list_adjacents(i))

    def isIndependent(self, X, Y, Z):
        _, p = self.learner.chi2(X, Y, Z)
        return p > self.alpha

    def PC_skeleton(self, stable=False):
        d = 0
        while True:
            if all(self.num_adjacents(i) - 1 < d for i in range(self.n)):
                break
            removedEdges = set()
            for i in range(self.n):
                adjacents = self.list_adjacents(i)
                if len(adjacents) - 1 < d:
                    continue
                neighbor_list = list(adjacents)
                for j in neighbor_list:
                    if j <= i:
                        continue
                    if not self.exist_edge(i, j):
                        continue
                    if stable and ((i, j) in removedEdges or (j, i) in removedEdges):
                        continue
                    candidates = [node for node in adjacents if node != j]
                    if len(candidates) < d:
                        continue
                    for Z in itertools.combinations(candidates, d):
                        if self.isIndependent(self.variables[i], self.variables[j], [self.variables[k] for k in Z]):
                            self.separators[(i, j)] = list(Z)
                            self.separators[(j, i)] = list(Z)
                            if stable:
                                removedEdges.add((i, j))
                            else:
                                self.remove_edge(i, j)
                            break
            if stable:
                for (i, j) in removedEdges:
                    self.remove_edge(i, j)
            d += 1

    def unshielded_triple_in_order_ijk(self, i, j, k):
        return self.exist_edge(i, j) and self.exist_edge(j, k) and not self.exist_edge(i, k)
    
    def rule0(self, i, j, k):
        if self.unshielded_triple_in_order_ijk(i, j, k) and j not in self.separators[(i, k)]:
            self.matrix[i, j] = ">"
            self.matrix[k, j] = ">"

    def is_collider(self, i, j, k):
        return self.matrix[i, j] == ">" and self.matrix[k, j] == ">"
    
    def is_triangle(self, i, j, k):
        return self.exist_edge(i, j) and self.exist_edge(j, k) and self.exist_edge(i, k)

    def is_discriminating_path(self, path):
        if len(path) < 4:
            return False
        x = path[0]
        y = path[-1]
        v = path[-2]
        if self.exist_edge(x, y):
            return False
        if not self.exist_edge(v, y):
            return False
        for i in range(1, len(path) - 2):
            if not self.exist_edge(path[i], y):
                return False
            if self.matrix[path[i], y] != ">":
                return False
            if not self.is_collider(path[i - 1], path[i], path[i + 1]):
                return False
        return True
    
    def get_discriminating_paths_targeted(self, u, y, current_path, limit = 10):
        current_node = current_path[-1]
        if len(current_path) > limit: return []
        paths = []
        adjacents = self.list_adjacents(current_node)
        for node in adjacents:
            if node in current_path: continue
            if node == y:
                full_path = current_path + [y]
                if self.is_discriminating_path(full_path):
                    paths.append(full_path)
                continue
            new_path = current_path + [node]
            paths += self.get_discriminating_paths_targeted(u, y, new_path, limit=limit)
        return paths

    def list_discriminating_paths(self):
        all_results = []
        for u in range(self.n):
            for y in range(self.n):
                if u == y: continue
                if self.exist_edge(u, y): continue
                found_paths = self.get_discriminating_paths_targeted(u, y, [u])
                all_results.extend(found_paths)
        return all_results
    
    def get_possible_d_sep(self, from_node, to_node):
        queue = []
        visited = set()
        pds = set()
        adjacents = self.list_adjacents(from_node)
        for node in adjacents:
            if node != to_node:
                queue.append((node, from_node))
                visited.add((node, from_node))
                pds.add(node)
        while len(queue) > 0:
            curr, prev = queue.pop(0)
            adjacents_curr = self.list_adjacents(curr)
            for next_node in adjacents_curr:
                if next_node == prev:
                    continue
                if next_node == from_node:
                    continue
                if self.is_collider(prev, curr, next_node) or self.is_triangle(prev, curr, next_node):
                    if (next_node, curr) not in visited:
                        visited.add((next_node, curr))
                        queue.append((next_node, curr))
                        if next_node != to_node:
                            pds.add(next_node)
        return pds
    
    # def make_possible_d_sep(self):
    #     possible_d_sep = dict()
    #     for i in range(self.n):
    #         for j in range(self.n):
    #             if i != j:
    #                 possible_d_sep[(i, j)] = self.get_possible_d_sep(i, j)
    #     return possible_d_sep
    
    import itertools

    def refine_skeleton_with_pds(self, stable=False):
        edges_to_remove = set()
        current_edges = list(self.list_edges()) 
        for (i, j) in current_edges:
            if (i, j) in edges_to_remove or (j, i) in edges_to_remove:
                continue
            pds_set = self.get_possible_d_sep(i, j)
            if j in pds_set: pds_set.remove(j)
            if i in pds_set: pds_set.remove(i)
            pds_list = list(pds_set)
            pds_size = len(pds_list)
            edge_removed = False
            for d in range(pds_size + 1):
                if edge_removed: break
                for Z in itertools.combinations(pds_list, d):
                    cond_set = list(Z)
                    if self.isIndependent(self.variables[i], self.variables[j], [self.variables[k] for k in cond_set]):
                        self.separators[(i, j)] = cond_set
                        self.separators[(j, i)] = cond_set
                        if stable:
                            edges_to_remove.add((i, j))
                        else:
                            self.remove_edge(i, j)
                        edge_removed = True
                        break
        if stable:
            for (i, j) in edges_to_remove:
                self.remove_edge(i, j)
        self.matrix = np.where(self.matrix == " ", " ", "o")

    def rule1(self, v1, v2, v3):
        if self.unshielded_triple_in_order_ijk(v1, v2, v3):
            if self.matrix[v1, v2] == ">" and self.matrix[v3, v2] == "o":
                self.matrix[v3, v2] = "-"
                self.matrix[v2, v3] = ">"
    
    def rule2(self, v1, v2, v3):
        if self.is_triangle(v1, v2, v3):
            if self.matrix[v1, v2] == ">" and self.matrix[v2, v3] == ">" and (self.matrix[v2, v1] == "-" or self.matrix[v3, v2] == "-"):
                if self.matrix[v1, v3] == "o":
                    self.matrix[v1, v3] = ">"

    def rule3(self, v1, v2, v3, v4):
        if self.exist_edge(v1, v2) and self.exist_edge(v3, v2) and self.exist_edge(v4, v2) and self.exist_edge(v1, v4) and self.exist_edge(v3, v4) and not self.exist_edge(v1, v3):
            if self.matrix[v1, v2] == ">" and self.matrix[v3, v2] == ">" and self.matrix[v1, v4] == "o" and self.matrix[v3, v4] == "o" and self.matrix[v4, v2] == "o":
                self.matrix[v4, v2] = ">"

    def rule4(self, path):
        x = path[0]
        y = path[-1]
        v = path[-2]
        w = path[-3]
        if self.is_discriminating_path(path):
            if self.matrix[y, v] == "o":
                if v in self.separators[(x, y)]:
                    self.matrix[v, y] = ">"
                    self.matrix[y, v] = "-"
                else:
                    self.matrix[w, v] = ">"
                    self.matrix[v, w] = ">"
                    self.matrix[v, y] = ">"
                    self.matrix[y, v] = ">"

    def is_uncovered_path(self, path):
        for i in range(1, len(path) - 1):
            if self.exist_edge(path[i - 1], path[i + 1]):
                return False
        return True
    
    def is_circle_edge(self, i, j):
        return self.matrix[i, j] == "o" and self.matrix[j, i] == "o"
    
    def is_circle_path(self, path):
        for i in range(0, len(path) - 1):
            if not self.is_circle_edge(path[i], path[i + 1]):
                return False
        return True
    
    def is_pd_edge(self, i, j):
        return FCI.arrow_attributes.index(self.matrix[i, j]) >= FCI.arrow_attributes.index(self.matrix[j, i])
    
    def is_pd_path(self, path):
        for i in range(len(path) - 1):
            if not self.is_pd_edge(path[i], path[i + 1]):
                return False
        return True
    
    def get_uncovered_circle_paths_targeted(self, u, y, current_path, limit=10):
        current_node = current_path[-1]
        if len(current_path) > limit: return []
        adjacents = self.list_adjacents(current_node)
        paths = []
        for node in adjacents:
            if self.is_circle_edge(node, current_node):
                if node == y:
                    continue
                if node in current_path: 
                    continue
                if len(current_path) >= 2:
                    if not self.exist_edge(node, current_path[-2]):
                        if self.exist_edge(node, y):
                            if self.is_circle_edge(node, y) and not self.exist_edge(y, current_path[-1]): paths.append(current_path + [node] + [y])
                        else:
                            new_path = current_path + [node]
                            paths += self.get_uncovered_circle_paths_targeted(u, y, new_path, limit=limit)
                else:
                    if self.exist_edge(node, y):
                        if self.is_circle_edge(node, y) and not self.exist_edge(y, current_path[-1]): paths.append(current_path + [node] + [y])
                    else:
                        new_path = current_path + [node]
                        paths += self.get_uncovered_circle_paths_targeted(u, y, new_path, limit=limit)
        return paths
    
    def get_uncovered_pd_paths_targeted(self, u, y, current_path, limit=10):
        current_node = current_path[-1]
        if len(current_path) > limit: return []
        adjacents = self.list_adjacents(current_node)
        paths = []
        for node in adjacents:
            if self.is_pd_edge(node, current_node):
                if node == y:
                    continue
                if node in current_path: 
                    continue
                if len(current_path) >= 2:
                    if not self.exist_edge(node, current_path[-2]):
                        if self.exist_edge(node, y):
                            if self.is_pd_edge(node, y) and not self.exist_edge(y, current_path[-1]): paths.append(current_path + [node] + [y])
                        else:
                            new_path = current_path + [node]
                            paths += self.get_uncovered_pd_paths_targeted(u, y, new_path, limit=limit)
                else:
                    if self.exist_edge(node, y):
                        if self.is_pd_edge(node, y) and not self.exist_edge(y, current_path[-1]): paths.append(current_path + [node] + [y])
                    else:
                        new_path = current_path + [node]
                        paths += self.get_uncovered_pd_paths_targeted(u, y, new_path, limit=limit)
        return paths
    
    def rule5(self, x, y):
        if self.is_circle_edge(x, y):
            paths = self.get_uncovered_circle_paths_targeted(x, y, [x])
            for p in paths:
                if len(p) >= 4:
                    if not (self.exist_edge(x, p[-2]) or self.exist_edge(y, p[1])):
                        self.matrix[x, y] = "-"
                        self.matrix[y, x] = "-"
                        for i in range(0, len(p) - 1):
                            self.matrix[p[i], p[i + 1]] = "-"
                            self.matrix[p[i + 1], p[i]] = "-"
                    return

    def rule6(self, v1, v2, v3):
        if self.exist_edge(v1, v2) and self.exist_edge(v2, v3):
            if self.matrix[v1, v2] == "-" and self.matrix[v2, v1] == "-" and self.matrix[v3, v2] == "o":
                self.matrix[v3, v2] = "-"

    def rule7(self, v1, v2, v3):
        if self.exist_edge(v1, v2) and self.exist_edge(v2, v3) and not self.exist_edge(v1, v3):
            if self.matrix[v1, v2] == "o" and self.matrix[v2, v1] == "-" and self.matrix[v3, v2] == "o":
                self.matrix[v3, v2] = "-"

    def rule8(self, v1, v2, v3):
        if self.exist_edge(v1, v2) and self.exist_edge(v2, v3) and self.exist_edge(v1, v3):
            if self.matrix[v2, v1] == "-" and (self.matrix[v1, v2] == "o" or self.matrix[v1, v2] == ">") and self.matrix[v3, v2] == "-" and self.matrix[v2, v3] == ">" and self.matrix[v1, v3] == ">" and self.matrix[v3, v1] == "o":
                self.matrix[v3, v1] = "-"

    def rule9(self, v1, v2):
        if self.exist_edge(v1, v2):
            if self.matrix[v1, v2] == ">" and self.matrix[v2, v1] == "o":
                paths = self.get_uncovered_pd_paths_targeted(v1, v2, [v1])
                for p in paths:
                    if len(p) >= 4:
                        if not self.exist_edge(p[1], v2):
                            self.matrix[v2, v1] = "-"
                            return

    def rule10(self, alpha, gamma, beta, theta):
        if self.exist_edge(alpha, gamma) and self.exist_edge(beta, gamma) and self.exist_edge(theta, gamma):
            if self.matrix[alpha, gamma] == ">" and self.matrix[gamma, alpha] == "o" and self.matrix[beta, gamma] == ">" and self.matrix[gamma, beta] == "-" and self.matrix[theta, gamma] == ">" and self.matrix[gamma, theta] == "-":
                paths1 = self.get_uncovered_pd_paths_targeted(alpha, beta, [alpha], limit=self.n)
                paths2 = self.get_uncovered_pd_paths_targeted(alpha, theta, [alpha], limit=self.n)
                for p1 in paths1:
                    for p2 in paths2:
                        if p1[1] != p2[1] and not self.exist_edge(p1[1], p2[1]):
                            self.matrix[gamma, alpha] = "-"
                            return
                        
    def find_minimal_sepset(self, i, j, superset_indices):
        candidate_nodes = list(superset_indices)
        n_candidates = len(candidate_nodes)
        for r in range(n_candidates + 1):
            for Z in itertools.combinations(candidate_nodes, r):
                Z_list = list(Z)
                var_i = self.variables[i]
                var_j = self.variables[j]
                var_k = [self.variables[k] for k in Z_list]
                if self.isIndependent(var_i, var_j, var_k):
                    return set(Z_list)
        return set()
    
    def really_fast_v_orientation(self):
        L = []
        triplets = list(itertools.permutations(range(self.n), r=3))
        M = [(t[0], t[1], t[2]) for t in triplets if self.unshielded_triple_in_order_ijk(t[0], t[1], t[2])]
        while len(M) > 0:
            (i, j, k) = M.pop(0)
            if not (self.exist_edge(i, j) and self.exist_edge(j, k)):
                continue
            sep_ik = self.separators.get((i, k), set())
            cond_set_indices = sep_ik - {j}
            cond_set_variables = [self.variables[k] for k in cond_set_indices]
            is_dep_ij = self.isIndependent(self.variables[i], self.variables[j], cond_set_variables)
            is_dep_jk = self.isIndependent(self.variables[j], self.variables[k], cond_set_variables)
            if not (is_dep_ij or is_dep_jk):
                if (i, j, k) not in L:
                    L.append((i, j, k))
            else:
                pairs_to_check = []
                if not is_dep_ij:
                    pairs_to_check.append((i, j))
                if not is_dep_jk:
                    pairs_to_check.append((k, j))
                for (r, q) in pairs_to_check:
                    Y = self.find_minimal_sepset(r, q, cond_set_indices)
                    if len(Y) > 0:
                        self.separators[(r, q)] = Y
                        self.separators[(q, r)] = Y
                        adjacents_r = self.list_adjacents(r)
                        adjacents_q = self.list_adjacents(q)
                        common_adjacents = adjacents_r.intersection(adjacents_q)
                        for w in common_adjacents:
                            triple_new = (r, w, q)
                            if triple_new not in M: 
                                M.append(triple_new)
                        M = [t for t in M if not({r, q} < set(t))]
                        L = [t for t in L if not({r, q} < set(t))]
                        self.remove_edge(r, q)
        for (i, j, k) in L:
            if self.exist_edge(i, j) and self.exist_edge(j, k):
                sep_ik = self.separators[(i, k)]
                if j not in sep_ik:
                    self.matrix[i, j] = ">"
                    self.matrix[k, j] = ">"

    def rule4_rfci(self, path):
        if self.is_discriminating_path(path):
            u = path[0]
            y = path[-1]
            sepset_ik = self.separators.get((u, y), set())
            edge_removed_in_path = False
            for idx in range(len(path) - 1):
                r = path[idx]
                q = path[idx + 1]
                base_set = sepset_ik - {r, q}
                candidate_vars_indices = list(base_set)
                limit_l = len(candidate_vars_indices)
                found_indep = False
                found_Y = None
                for l in range(limit_l + 1):
                    for Z in itertools.combinations(candidate_vars_indices, l):
                        var_r = self.variables[r]
                        var_q = self.variables[q]
                        var_Z = [self.variables[zz] for zz in Z]
                        if self.isIndependent(var_r, var_q, var_Z):
                            found_indep = True
                            found_Y = set(Z)
                            break
                    if found_indep:
                        break
                if found_indep:
                    self.separators[(r, q)] = found_Y
                    self.separators[(q, r)] = found_Y
                    self.remove_edge(r, q)
                    self.really_fast_v_orientation()
                    edge_removed_in_path = True
                    break
            if not edge_removed_in_path:
                v = path[-2]
                if v in sepset_ik:
                    if self.matrix[v, y] != ">":
                        self.matrix[v, y] = ">"
                        self.matrix[y, v] = "-"
                else:
                    w = path[-3]
                    if self.matrix[w, v] != ">" or self.matrix[v, w] != ">":
                        self.matrix[w, v] = ">"
                        self.matrix[v, w] = ">"
                    if self.matrix[v, y] != ">" or self.matrix[y, v] != ">":
                        self.matrix[v, y] = ">"
                        self.matrix[y, v] = ">"


    def triangle_for_rfci(self, l, j, k):
        return self.is_triangle(l, j, k) and self.matrix[k, j] == "o" and self.matrix[j, l] == ">" and self.matrix[l, k] == ">" and self.matrix[k, l] == "-"

    def return_PDAG(self):
        edges = set()
        arcs = set()
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.exist_edge(i, j):
                    if self.matrix[i, j] == self.matrix[j, i]:
                        edges.add((i, j))
                    elif FCI.arrow_attributes.index(self.matrix[i, j]) > FCI.arrow_attributes.index(self.matrix[j, i]):
                        arcs.add((i, j))
                    else:
                        arcs.add((j, i))
        gum_graph = gum.PDAG()
        for i in range(self.n):
            gum_graph.addNodeWithId(i)
        for (i, j) in edges:
            gum_graph.addEdge(i, j)
        for (i, j) in arcs:
            gum_graph.addArc(i, j)
        return gum_graph
    
    def return_UndiMG(self):
        edges = set()
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.exist_edge(i, j):
                    if self.matrix[i, j] == self.matrix[j, i]:
                        edges.add((i, j))
        gum_graph = gum.MixedGraph()
        for i in range(self.n):
            gum_graph.addNodeWithId(i)
        for (i, j) in edges:
            gum_graph.addEdge(i, j)
        return gum_graph


# FCI FAMILY

def FCI_PC_pyagrum(df, alpha=0.05, stable=False):
    fci = FCI(df, alpha=alpha)
    fci.PC_skeleton(stable=stable)
    triplets = list(itertools.permutations(range(fci.n), r=3))
    for t in triplets:
        fci.rule0(t[0], t[1], t[2])
    graph = fci.matrix
    old_graph = np.full((fci.n, fci.n), "")
    while not np.array_equal(old_graph, graph):
        old_graph = graph.copy()
        fci.refine_skeleton_with_pds()
        for t in triplets:
            fci.rule0(t[0], t[1], t[2])
        graph = fci.matrix
    return fci

def FCI_Zhang_pyagrum(df, alpha=0.05, stable=False):
    fci = FCI(df, alpha=alpha)
    fci.PC_skeleton(stable = stable)
    triplets = list(itertools.permutations(range(fci.n), r=3))
    for t in triplets:
        fci.rule0(t[0], t[1], t[2])
    graph = fci.matrix
    old_graph = np.full((fci.n, fci.n), "")
    while not np.array_equal(old_graph, graph):
        old_graph = graph.copy()
        for t in triplets:
            fci.rule1(t[0], t[1], t[2])
        for t in triplets:
            fci.rule2(t[0], t[1], t[2])
        for t in triplets:
            for new_elem in range(fci.n):
                if new_elem not in t:
                    fci.rule3(t[0], t[1], t[2], new_elem)
        paths = fci.list_discriminating_paths()
        for path in paths:
            fci.rule4(path)
        graph = fci.matrix
    return fci

def FCI_ETHZ_pyagrum(df, alpha=0.05, stable=False):
    fci = FCI(df, alpha=alpha)
    fci.PC_skeleton(stable=stable)
    triplets = list(itertools.permutations(range(fci.n), r=3))
    for t in triplets:
        fci.rule0(t[0], t[1], t[2])
    fci.refine_skeleton_with_pds(stable=stable)
    for t in triplets:
        fci.rule0(t[0], t[1], t[2])
    graph = fci.matrix
    old_graph = np.full((fci.n, fci.n), "")
    while not np.array_equal(old_graph, graph):
        old_graph = graph.copy()
        for t in triplets:
            fci.rule1(t[0], t[1], t[2])
        for t in triplets:
            fci.rule2(t[0], t[1], t[2])
        for t in triplets:
            for new_elem in range(fci.n):
                if new_elem not in t:
                    fci.rule3(t[0], t[1], t[2], new_elem)
        paths = fci.list_discriminating_paths()
        for path in paths:
            fci.rule4(path)
        graph = fci.matrix
    couples = list(itertools.permutations(range(fci.n), r=2))
    quadruplets = list(itertools.permutations(range(fci.n), r=4))
    graph = fci.matrix
    old_graph = np.full((fci.n, fci.n), "")
    while not np.array_equal(old_graph, graph):
        for (x, y) in couples:
            fci.rule5(x, y)
        for (x, y, z) in triplets:
            fci.rule6(x, y, z)
        for (x, y, z) in triplets:
            fci.rule7(x, y, z)
        old_graph = graph.copy()
        graph = fci.matrix
    old_graph = np.full((fci.n, fci.n), "")
    while not np.array_equal(old_graph, graph):
        for (x, y, z) in triplets:
            fci.rule8(x, y, z)
        for (x, y) in couples:
            fci.rule9(x, y)
        for (x, y, z, t) in quadruplets:
            fci.rule10(x, y, z, t)
        old_graph = graph.copy()
        graph = fci.matrix
    return fci

def raw_skeleton(df, alpha = 0.05, stable=False):
    fci = FCI(df, alpha=alpha)
    fci.PC_skeleton(stable = stable)
    triplets = list(itertools.permutations(range(fci.n), r=3))
    for t in triplets:
        fci.rule0(t[0], t[1], t[2])
    fci.refine_skeleton_with_pds()
    for t in triplets:
        fci.rule0(t[0], t[1], t[2])
    return fci

def AFCI_Zhang_pyagrum(df, alpha = 0.05, stable=False):
    fci = FCI_Zhang_pyagrum(df, alpha, stable=stable)
    couples = list(itertools.permutations(range(fci.n), r=2))
    triplets = list(itertools.permutations(range(fci.n), r=3))
    quadruplets = list(itertools.permutations(range(fci.n), r=4))
    graph = fci.matrix
    old_graph = np.full((fci.n, fci.n), "")
    while not np.array_equal(old_graph, graph):
        for (x, y) in couples:
            fci.rule5(x, y)
        for (x, y, z) in triplets:
            fci.rule6(x, y, z)
        for (x, y, z) in triplets:
            fci.rule7(x, y, z)
        old_graph = graph.copy()
        graph = fci.matrix
    old_graph = np.full((fci.n, fci.n), "")
    while not np.array_equal(old_graph, graph):
        for (x, y, z) in triplets:
            fci.rule8(x, y, z)
        for (x, y) in couples:
            fci.rule9(x, y)
        for (x, y, z, t) in quadruplets:
            fci.rule10(x, y, z, t)
        old_graph = graph.copy()
        graph = fci.matrix
    return fci

def RFCI(df, alpha = 0.05, stable=False):
    fci = FCI(df, alpha=alpha, stable=stable)
    fci.PC_skeleton()
    fci.really_fast_v_orientation()
    couples = list(itertools.permutations(range(fci.n), r=2))
    triplets = list(itertools.permutations(range(fci.n), r=3))
    quadruplets = list(itertools.permutations(range(fci.n), r=4))
    graph = fci.matrix
    old_graph = np.full((fci.n, fci.n), "")
    while not np.array_equal(old_graph, graph):
        old_graph = graph.copy()
        for t in triplets:
            fci.rule1(t[0], t[1], t[2])
        for t in triplets:
            fci.rule2(t[0], t[1], t[2])
        for t in triplets:
            for new_elem in range(fci.n):
                if new_elem not in t:
                    fci.rule3(t[0], t[1], t[2], new_elem)
        potential_paths = fci.list_discriminating_paths()
        potential_paths.sort(key=len)
        for path in potential_paths:
            fci.rule4_rfci(path)
        for (x, y) in couples:
            fci.rule5(x, y)
        for (x, y, z) in triplets:
            fci.rule6(x, y, z)
        for (x, y, z) in triplets:
            fci.rule7(x, y, z)
        for (x, y, z) in triplets:
            fci.rule8(x, y, z)
        for (x, y) in couples:
            fci.rule9(x, y)
        for (x, y, z, t) in quadruplets:
            fci.rule10(x, y, z, t)
        graph = fci.matrix
    return fci
