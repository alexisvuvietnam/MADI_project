import pyagrum as gum
import time
from FCI import FCI
from graphviz import Digraph, Source
from operator import xor


def generate_test_data(bn, n_samples = 1000):
    '''Génère des données à partir d'un réseau bayésien.'''
    return gum.generateSample(bn, n_samples, with_labels=True, random_order=False)

def measure_execution_time(func, *args, **kwargs):
    '''Mesure le temps d'exécution d'une fonction.'''
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed

def get_BNlabel_nodes(bn, lantent_var=None):
    if lantent_var is None:
        return {n : bn.variable(n).name() for n in bn.nodes() } 
    res = {}
    index = lantent_var[0]
    for n in bn.nodes():
        if n < index:
            res[n] = bn.variable(n).name()
        elif n > index : 
            res[n-1] = bn.variable(n).name()
    return res

def get_fci_structure(fci):
    '''
    Use for the structure analysis comparison. 
    Returns:
        - skeleton edges set() labelized
        - oriented edges (X*->Y)
        - totally oriented edges (X->Y or X<->Y)
        - number of circles tails (Xo-*Y)
        - certainty score 
    '''

    edges = set(fci.list_edges())

    num_oriented = 0
    num_totally_oriented = 0
    num_circles = 0
    
    for (i, j) in edges:
        if xor(fci.matrix[i, j] == ">", fci.matrix[j, i] == ">"):
            # oriented.add((i, j))
            num_oriented += 1
            if fci.matrix[i, j] == "-" or fci.matrix[j, i] == "-":
                # totally_oriented.add((i,j))
                num_totally_oriented += 1
        elif fci.matrix[i, j] == ">" and fci.matrix[j, i] == ">":
            # oriented.add((j, i))
            # totally_oriented.add((j,i))
            num_oriented += 1
            num_totally_oriented += 1

        if xor(fci.matrix[i,j] == 'o', fci.matrix[j,i]=='o'):
            num_circles += 1
        elif fci.matrix[i,j]=='o' and fci.matrix[j,i]=='o':
            num_circles += 2
        
    certainty_score = num_circles / (2 * len(edges)) if len(edges) > 0 else 0

    labels = fci.get_variables()
    # print('FCI', labels)
    edges_labeled = {tuple(sorted([labels[i], labels[j]])) for (i, j) in edges}


    return {
        'edges' : edges_labeled, 
        'num_oriented' : num_oriented, 
        'num_totally_oriented' : num_totally_oriented,
        'num_circles' : num_circles,
        'certainty_score' : certainty_score
    }


def get_miic_structure(miic, bn, latent_variable=None):
    '''
    Returns:
        - skeleton
        - edges
        - arcs
    '''
    edges = miic.edges()
    arcs = miic.arcs()
    
    
    labels = get_BNlabel_nodes(bn, latent_variable)
    # print('BN', labels)
    arcs_labeled = {tuple(sorted([labels[i], labels[j]])) for (i, j) in arcs}
    edges_labeled = {tuple(sorted([labels[i], labels[j]])) for (i, j) in edges}


    return  {
        'skeleton' : edges_labeled.union(arcs_labeled),
        'num_edges': len(edges),
        'num_arcs' : len(arcs)
    }

def compare_structures(fci, miic, bn, fci_time, miic_time, latent_variable=None, verbose=True):
    '''
    Comparison of structure between fci and miic.
    It prints the information.
    '''
    fci_struct = get_fci_structure(fci)
    miic_struct = get_miic_structure(miic, bn, latent_variable)

    common_edges = len(fci_struct['edges'] & miic_struct['skeleton'])
    
    total_fci = len(fci_struct['edges'])
    total_miic = len(miic_struct['skeleton'])
    jaccard_similarity = common_edges / (total_fci + total_miic - common_edges) if (total_fci + total_miic - common_edges) > 0 else 0
    t_ratio = fci_time/miic_time

    # Print results
    if verbose :
        print("-" * 60)
        print("COMPARAISON FCI vs MIIC")
        print("-" * 60)
        print(f"Structure:")
        print(f"  FCI  - Total arêtes du squelette: {total_fci}")
        print(f"       - Orientées: {fci_struct['num_oriented']}")
        print(f"       - Taux de certitude: {fci_struct['certainty_score']*100:.0f}% ({fci_struct['num_circles']}/{2*total_fci})")
        print(f"  MIIC - Total arêtes du squelette : {total_miic}")
        print(f"       - Arcs: {miic_struct['num_arcs']}")
        print(f"\nSimilarité:")
        print(f"  Arêtes communes: {common_edges}")
        print(f"  Jaccard similarity sur squelette: {jaccard_similarity:.3f}")
        print(f"\nTemps d'exécution:")
        print(f"  FCI:  {fci_time:.3f}s")
        print(f"  MIIC: {miic_time:.3f}s")
        print(f"  Ratio (FCI/MIIC): {t_ratio:.2f}x")

    return {
        'fci_edges' : fci_struct['edges'], 
        'fci_total' : total_fci,
        'fci_num_oriented' : fci_struct['num_oriented'], 
        'fci_num_totally_oriented' : fci_struct['num_totally_oriented'],
        'fci_num_circles' : fci_struct['num_circles'],
        'fci_certainty_score' : fci_struct['certainty_score'],
        'miic_skeleton' : miic_struct['skeleton'], 
        'miic_total' : total_miic,
        'miic_num_edges' : miic_struct['num_edges'], 
        'miic_num_arcs' : miic_struct['num_arcs'], 
        'common_edges': common_edges,
        'jaccard_similarity' : jaccard_similarity,
        't_ratio' : t_ratio
    }

def labelize_pdag_nodes(pdag, struct, latent_var=None):
    '''
    Change id to label in dot display
    :param struct : BN or FCI
    '''
    dot = Digraph("Output PDAG", node_attr={"shape": "oval", "fillcolor": "#333333", "textcolor": "#eeeeee"})
    dot.attr("node")
    if isinstance(struct, gum.BayesNet):
        labels = get_BNlabel_nodes(struct, latent_var)
        # print('BN', labels)
    elif isinstance(struct, FCI):
        labels = struct.get_variables()
        # print('FCI', labels)

    for n in pdag.nodes():
        dot.node(labels[n], label=f"{labels[n]}")

    for (i, j) in pdag.edges():
        dot.edge(labels[i], labels[j], dir="none")
    for (i,j) in pdag.arcs():
        dot.edge(labels[i], labels[j], dir="forward")
    return dot