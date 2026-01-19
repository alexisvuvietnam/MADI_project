import itertools
import numpy as np
import pandas as pd
import pingouin as pg

# SKELETON

def create_raw_graph(nodes):
    graph = np.full((nodes, nodes), "o")
    for i in range(nodes):
        graph[i, i] = " "
    return graph

def exist_edge(graph, i, j):
    if i == j: return False
    return graph[i, j] != " " and graph[j, i] != " "

def list_edges(graph):
    _, n = graph.shape
    L = []
    for i in range(n):
        for j in range(i + 1, n):
            if exist_edge(graph, i, j):
                L.append((i, j))
    return L

def list_neighbors(graph, v):
    _, n = graph.shape
    return set([i for i in range(n) if exist_edge(graph, v, i)])

def PC_skeleton(df, alpha):
    variables = list(df.columns)
    _, n = df.shape
    graph = create_raw_graph(n)
    d = 0
    separator_set = dict()
    for i in range(n):
        for j in range(n):
            if i != j:
                separator_set[(i, j)] = set()
    cont = True
    while cont:
        cont = False
        edges = list_edges(graph)
        for (i, j) in edges:
            if exist_edge(graph, i, j):
                neighbors_i = list_neighbors(graph, i)
                neighbors_j = list_neighbors(graph, j)
                check_list = [(j, neighbors_i), (i, neighbors_j)]
                for (x, neighbors) in check_list:
                    if len(neighbors) >= d:
                        cont = True
                        Z_list = [z for z in itertools.combinations([k for k in neighbors if k != x], d)]
                        for Z in Z_list:
                            res = pg.partial_corr(data = df, x = variables[i], y = variables[j], covar = [variables[k] for k in Z])
                            if res["p-val"].values[0] > alpha:
                                graph[i, j] = " "
                                graph[j, i] = " "
                                separator_set[(i, j)] = separator_set[(i, j)].union(Z)
                                separator_set[(j, i)] = separator_set[(j, i)].union(Z)
                                break
        d += 1
        #if all(len(list_neighbors_from_init(graph, node)) <= d for node in range(n)): break
    return graph, separator_set

# BASIC RULES

def unshielded_triple_in_order_123(graph, v1, v2, v3):
    return exist_edge(graph, v1, v2) and exist_edge(graph, v2, v3) and not exist_edge(graph, v1, v3)

def rule0(graph, v1, v2, v3, separator_set):
    if unshielded_triple_in_order_123(graph, v1, v2, v3) and v2 not in separator_set[(v1, v3)]:
        graph[v1, v2] = ">"
        graph[v3, v2] = ">"

def is_collider(graph, x, y, z):
    return graph[x, y] == ">" and graph[z, y] == ">"

def is_triangle(graph, x, y, z):
    return exist_edge(graph, x, y) and exist_edge(graph, y, z) and exist_edge(graph, x, z)

def is_discriminating_path(graph, path):
    if len(path) < 4:
        return False
    x = path[0]
    y = path[-1]
    v = path[-2]
    if exist_edge(graph, x, y):
        return False
    if not exist_edge(graph, v, y):
        return False
    for i in range(1, len(path) - 2):
        if not exist_edge(graph, path[i], y):
            return False
        if graph[path[i], y] != ">":
            return False
        if not is_collider(graph, path[i - 1], path[i], path[i + 1]):
            return False
    return True

def get_discriminating_paths_targeted(graph, u, y, current_path, limit = 10):
    current_node = current_path[-1]
    if len(current_path) > limit: return []
    paths = []
    neighbors = list_neighbors(graph, current_node)
    for node in neighbors:
        if node in current_path: continue
        if node == y:
            full_path = current_path + [y]
            if is_discriminating_path(graph, full_path):
                paths.append(full_path)
            continue
        new_path = current_path + [node]
        paths += get_discriminating_paths_targeted(graph, u, y, new_path, limit=limit)
    return paths

def list_discriminating_paths(graph):
    _, n = graph.shape
    all_results = []
    for u in range(n):
        for y in range(n):
            if u == y: continue
            if exist_edge(graph, u, y): continue
            found_paths = get_discriminating_paths_targeted(graph, u, y, [u])
            all_results.extend(found_paths)
    return all_results

# def list_discriminating_paths_brute_force(graph, previous):
#     paths = []
#     if len(previous) == 0:
#         n = graph.shape
#         for i in range(n):
#             new_previous = [i]
#             paths += list_discriminating_path_brute_force(graph, new_previous)
#     else:
#         neighbors = list_neighbors(graph, previous[-1])
#         for j in neighbors:
#             if j not in previous:
#                 new_previous = previous + [j]
#                 paths += list_discriminating_path_brute_force(graph, new_previous)
#                 if is_discriminating_path(graph, new_previous):
#                     paths.append(new_previous)
#     return paths

def get_possible_d_sep(graph, from_node, to_node):
    queue = []
    visited = set()
    pds = set()
    neighbors = list_neighbors(graph, from_node)
    for node in neighbors:
        if node != to_node:
            queue.append((node, from_node))
            visited.add((node, from_node))
            pds.add(node)
    while len(queue) > 0:
        curr, prev = queue.pop(0)
        neighbors_curr = list_neighbors(graph, curr)
        for next_node in neighbors_curr:
            if next_node == prev:
                continue
            if next_node == from_node:
                continue
            if is_collider(graph, prev, curr, next_node) or is_triangle(graph, prev, curr, next_node):
                if (next_node, curr) not in visited:
                    visited.add((next_node, curr))
                    queue.append((next_node, curr))
                    if next_node != to_node:
                        pds.add(next_node)
    return pds

def make_possible_d_sep(graph):
    _, n = graph.shape
    possible_d_sep = dict()
    for i in range(n):
        for j in range(n):
            if i != j:
                possible_d_sep[(i, j)] = get_possible_d_sep(graph, i, j)
    return possible_d_sep

def refine_skeleton_with_pds(df, graph, separator_set, possible_d_sep, alpha = 0.05):
    variables = list(df.columns)
    edges = list_edges(graph)
    for (i, j) in edges:
        if exist_edge(graph, i, j):
            pds = possible_d_sep[(i, j)]
            length = len(pds)
            for d in range(length - length // 2):
                Z_list = [z for z in itertools.combinations([k for k in pds], d)]
                for Z in Z_list:
                    res = pg.partial_corr(data = df, x = variables[i], y = variables[j], covar = [variables[k] for k in Z])
                    if res["p-val"].values[0] > alpha:
                        graph[i, j] = " "
                        graph[j, i] = " "
                        separator_set[(i, j)] = separator_set[(i, j)].union(Z)
                        separator_set[(j, i)] = separator_set[(j, i)].union(Z)
                        break
            
def rule1(graph, v1, v2, v3):
    if unshielded_triple_in_order_123(graph, v1, v2, v3):
        if graph[v1, v2] == ">" and graph[v3, v2] == "o":
            graph[v3, v2] = "-"
            graph[v2, v3] = ">"

def rule2(graph, v1, v2, v3):
    if is_triangle(graph, v1, v2, v3):
        if graph[v1, v2] == ">" and graph[v2, v3] == ">" and (graph[v2, v1] == "-" or graph[v3, v2] == "-"):
            if graph[v1, v3] == "o":
                graph[v1, v3] = ">"

def rule3(graph, v1, v2, v3, v4):
    if exist_edge(graph, v1, v2) and exist_edge(graph, v3, v2) and exist_edge(graph, v4, v2) and exist_edge(graph, v1, v4) and exist_edge(graph, v3, v4) and not exist_edge(graph, v1, v3):
        if graph[v1, v2] == ">" and graph[v3, v2] == ">" and graph[v1, v4] == "o" and graph[v3, v4] == "o" and graph[v4, v2] == "o":
            graph[v4, v2] = ">"

def rule4(graph, path, separator_set):
    x = path[0]
    y = path[-1]
    v = path[-2]
    w = path[-3]
    if is_discriminating_path(graph, path):
        if graph[y, v] == "o":
            if v in separator_set[(x, y)]:
                graph[v, y] = ">"
                graph[y, v] = "-"
            else:
                graph[w, v] = ">"
                graph[v, w] = ">"
                graph[v, y] = ">"
                graph[y, v] = ">"

# ADVANCED RULES

def is_uncovered_path(graph, path):
    for i in range(1, len(path) - 1):
        if exist_edge(graph, path[i - 1], path[i + 1]):
            return False
    return True

def is_circle_edge(graph, i, j):
    return graph[i, j] == "o" and graph[j, i] == "o"

def is_circle_path(graph, path):
    for i in range(0, len(path) - 1):
        if is_circle_edge(graph, path[i], path[i + 1]):
            return False
    return True

def is_pd_edge(graph, i, j):
    attributes = ["-", "o", ">"]
    return attributes.index(graph[i, j]) >= attributes.index(graph[j, i])

def is_potentially_directed_path(graph, path):
    for i in range(len(path) - 1):
        if graph[path[i], path[i + 1]] == "-" or graph[path[i + 1], path[i]] == ">":
            return False
    return True
                
def get_uncovered_circle_paths_targeted(graph, u, y, current_path, limit=10):
    current_node = current_path[-1]
    #if len(current_path) > limit: return []
    neighbors = list_neighbors(graph, current_node)
    paths = []
    for node in neighbors:
        if is_circle_edge(graph, node, current_node):
            if node == y:
                continue
            if node in current_path: 
                continue
            if len(current_path) >= 2:
                if not exist_edge(graph, node, current_path[-2]):
                    if exist_edge(graph, node, y):
                        if is_circle_edge(graph, node, y) and not exist_edge(graph, y, current_path[-1]): paths.append(current_path + [node] + [y])
                    else:
                        new_path = current_path + [node]
                        paths += get_uncovered_circle_paths_targeted(graph, u, y, new_path, limit=limit)
            else:
                if exist_edge(graph, node, y):
                    if is_circle_edge(graph, node, y) and not exist_edge(graph, y, current_path[-1]): paths.append(current_path + [node] + [y])
                else:
                    new_path = current_path + [node]
                    paths += get_uncovered_circle_paths_targeted(graph, u, y, new_path, limit=limit)
    return paths

def get_uncovered_pd_paths_targeted(graph, u, y, current_path, limit=10):
    current_node = current_path[-1]
    #if len(current_path) > limit: return []
    neighbors = list_neighbors(graph, current_node)
    paths = []
    for node in neighbors:
        if is_pd_edge(graph, current_node, node):
            if node == y:
                continue
            if node in current_path: 
                continue
            if len(current_path) >= 2:
                if not exist_edge(graph, node, current_path[-2]):
                    if exist_edge(graph, node, y):
                        if is_pd_edge(graph, node, y) and not exist_edge(graph, y, current_path[-1]): paths.append(current_path + [node] + [y])
                    else:
                        new_path = current_path + [node]
                        paths += get_uncovered_pd_paths_targeted(graph, u, y, new_path, limit=limit)
            else:
                if exist_edge(graph, node, y):
                    if is_pd_edge(graph, node, y) and not exist_edge(graph, y, current_path[-1]): paths.append(current_path + [node] + [y])
                else:
                    new_path = current_path + [node]
                    paths += get_uncovered_pd_paths_targeted(graph, u, y, new_path, limit=limit)
    return paths

def rule5(graph, x, y):
    if is_circle_edge(graph, x, y):
        paths = get_uncovered_circle_paths_targeted(graph, x, y, [x])
        for p in paths:
            if len(p) >= 4:
                if not (exist_edge(graph, x, p[-2]) or exist_edge(graph, y, p[1])):
                    graph[x, y] = "-"
                    graph[y, x] = "-"
                    for i in range(0, len(p) - 1):
                        graph[p[i], p[i + 1]] = "-"
                        graph[p[i + 1], p[i]] = "-"
                    return

def rule6(graph, v1, v2, v3):
    if exist_edge(graph, v1, v2) and exist_edge(graph, v2, v3):
        if graph[v1, v2] == "-" and graph[v2, v1] == "-" and graph[v3, v2] == "o":
            graph[v3, v2] = "-"

def rule7(graph, v1, v2, v3):
    if exist_edge(graph, v1, v2) and exist_edge(graph, v2, v3) and not exist_edge(graph, v1, v3):
        if graph[v1, v2] == "o" and graph[v2, v1] == "-" and graph[v3, v2] == "o":
            graph[v3, v2] = "-"

def rule8(graph, v1, v2, v3):
    if exist_edge(graph, v1, v2) and exist_edge(graph, v2, v3) and exist_edge(graph, v1, v3):
        if graph[v2, v1] == "-" and (graph[v1, v2] == "o" or graph[v1, v2] == ">") and graph[v3, v2] == "-" and graph[v2, v3] == ">" and graph[v1, v3] == ">" and graph[v3, v1] == "o":
            graph[v3, v1] = "-"

def rule9(graph, v1, v2):
    if exist_edge(graph, v1, v2):
        if graph[v1, v2] == ">" and graph[v2, v1] == "o":
            paths = get_uncovered_pd_paths_targeted(graph, v1, v2, [v1])
            for p in paths:
                if len(p) >= 4:
                    if not exist_edge(graph, p[1], v2):
                        graph[v2, v1] = "-"
                        return

def rule10(graph, alpha, gamma, beta, theta):
    _, limit = graph.shape
    if exist_edge(graph, alpha, gamma) and exist_edge(graph, beta, gamma) and exist_edge(graph, theta, gamma):
        if graph[alpha, gamma] == ">" and graph[gamma, alpha] == "o" and graph[beta, gamma] == ">" and graph[gamma, beta] == "-" and graph[theta, gamma] == ">" and graph[gamma, theta] == "-":
            paths1 = get_uncovered_pd_paths_targeted(graph, alpha, beta, [alpha], limit=limit)
            paths2 = get_uncovered_pd_paths_targeted(graph, alpha, theta, [alpha], limit=limit)
            for p1 in paths1:
                for p2 in paths2:
                    if p1[1] != p2[1] and not exist_edge(graph, p1[1], p2[1]):
                        graph[gamma, alpha] = "-"
                        return

# FCI FAMILY

def FCI_PC(df, alpha=0.05):
    _, n = df.shape
    graph, separator_set = PC_skeleton(df, alpha=alpha)
    triplets = list(itertools.permutations(range(n), r=3))
    for t in triplets:
        rule0(graph, t[0], t[1], t[2], separator_set)
    old_graph = np.full((n, n), "")
    while not np.array_equal(old_graph, graph):
        old_graph = graph.copy()
        possible_d_sep = make_possible_d_sep(graph)
        for t in triplets:
            rule0(graph, t[0], t[1], t[2], separator_set)
        refine_skeleton_with_pds(df, graph, separator_set, possible_d_sep, alpha=alpha)
    return graph
    
def FCI_Zhang(df, alpha=0.05):
    _, n = df.shape
    graph, separator_set = PC_skeleton(df, alpha=alpha)
    triplets = list(itertools.permutations(range(n), r=3))
    for t in triplets:
        rule0(graph, t[0], t[1], t[2], separator_set)
    old_graph = np.full((n, n), "")
    while not np.array_equal(old_graph, graph):
        old_graph = graph.copy()
        for t in triplets:
            rule1(graph, t[0], t[1], t[2])
        for t in triplets:
            rule2(graph, t[0], t[1], t[2])
        for t in triplets:
            for new_elem in range(n):
                if new_elem not in t:
                    rule3(graph, t[0], t[1], t[2], new_elem)
        paths = list_discriminating_paths(graph)
        #paths = list_discriminating_paths_brute_force(graph, previous)
        for path in paths:
            rule4(graph, path, separator_set)
    return graph
    
def FCI_ETHZ(df, alpha=0.05):
    _, n = df.shape
    graph, separator_set = PC_skeleton(df, alpha=alpha)
    triplets = list(itertools.permutations(range(n), r=3))
    for t in triplets:
        rule0(graph, t[0], t[1], t[2], separator_set)
    possible_d_sep = make_possible_d_sep(graph)
    for t in triplets:
        rule0(graph, t[0], t[1], t[2], separator_set)
    refine_skeleton_with_pds(df, graph, separator_set, possible_d_sep, alpha=alpha)
    old_graph = np.full((n, n), "")
    while not np.array_equal(old_graph, graph):
        old_graph = graph.copy()
        for t in triplets:
            rule1(graph, t[0], t[1], t[2])
        for t in triplets:
            rule2(graph, t[0], t[1], t[2])
        for t in triplets:
            for new_elem in range(n):
                if new_elem not in t:
                    rule3(graph, t[0], t[1], t[2], new_elem)
        paths = list_discriminating_paths(graph)
        #paths = list_discriminating_paths_brute_force(graph, previous)
        for path in paths:
            rule4(graph, path, separator_set)
    return graph

def AFCI(df, alpha = 0.05, FCI_Func=FCI_Zhang):
    _, n = df.shape
    graph = FCI_Func(df, alpha)
    couples = list(itertools.permutations(range(n), r=2))
    triplets = list(itertools.permutations(range(n), r=3))
    quadruplets = list(itertools.permutations(range(n), r=4))
    old_graph = np.full((n, n), "")
    while not np.array_equal(old_graph, graph):
        for (x, y) in couples:
            rule5(graph, x, y)
        for (x, y, z) in triplets:
            rule6(graph, x, y, z)
        for (x, y, z) in triplets:
            rule7(graph, x, y, z)
        old_graph = graph.copy()
    old_graph = np.full((n, n), "")
    while not np.array_equal(old_graph, graph):
        for (x, y, z) in triplets:
            rule8(graph, x, y, z)
        for (x, y) in couples:
            rule9(graph, x, y)
        for (x, y, z, t) in quadruplets:
            rule10(graph, x, y, z, t)
        old_graph = graph.copy()