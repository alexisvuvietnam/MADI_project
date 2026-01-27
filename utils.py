import pyagrum as gum
import time
from FCI import FCI
from graphviz import Digraph, Source



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


def labelize_edge(mapp, edge):
    i,j = edge
    return mapp[i],mapp[j]

# def compare_structure_pdag(fci, miic_pdag, bn, opt=2):
#     '''
#     Comparison of structure between fci and miic
    
#     :param FCI fci: 
#     :param miic_pdag: resulting of MIIC
#     '''

#     fci_pdag = fci.return_PDAG(opt=opt)

#     fci_edges = fci_pdag.edges()
#     fci_arcs = fci_pdag.arcs()
    
#     miic_edges = miic_pdag.edges()
#     miic_arcs = miic_pdag.arcs()

#     labels = fci.get_variables()
#     fci_edges = set(map(lambda edge : labelize_edge(labels, edge), fci_edges))
#     labels = get_BNlabel_nodes(bn)
#     miic_arcs = set(map(lambda edge : labelize_edge(labels, edge), miic_arcs))
#     miic_edges = set(map(lambda edge : labelize_edge(labels, edge), miic_edges))

#     common_edges = len(fci_arcs & miic_arcs) + len(fci_edges & miic_edges)
#     total_fci = len(fci_edges) + len(fci_arcs)
#     total_miic = len(miic_edges) + len(miic_arcs)
#     jaccard_similarity = common_edges / (total_fci + total_miic - common_edges) if (total_fci + total_miic - common_edges) > 0 else 0

#     return {
#         'fci_edges' : len(fci_edges),
#         'fci_arcs' : len(fci_arcs),
#         'miic_edges' : len(miic_edges),
#         'miic_arcs' : len(miic_arcs),
#         'common_edges' : common_edges,
#         'jaccard_similarity' : jaccard_similarity
#     }

def compare_structures(fci, miic_pdag, bn):
    '''
    Comparison of structure between fci and miic
    
    :param FCI fci: 
    :param miic_pdag: resulting of MIIC
    '''
    fci_edges = set(fci.list_edges())

    # On consière que l'arête est orienté si elle est de la forme X*->Y
    fci_oriented = set()
    fci_unoriented = set()
    fci_circle = 0
    
    for (i, j) in fci_edges:
        if fci.matrix[i, j] == ">" or fci.matrix[j, i] == ">":
            fci_oriented.add((i, j))
        elif fci.matrix[i, j] == ">" and fci.matrix[j, i] == ">":
            fci_oriented.add((j, i))
        else :
            fci_unoriented.add(tuple(sorted([i, j])))
        if fci.matrix[i,j] == 'o' ^ fci.matrix[j,i]=='o':
            fci_circle += 1
        elif fci.matrix[i,j]=='o' and fci.matrix[j,i]=='o':
            fci_circle += 2

    
    miic_edges = miic_pdag.edges()
    miic_arcs = miic_pdag.arcs()
    
    labels = fci.get_variables()
    print('FCI', labels)
    fci_edges = set(map(lambda edge : labelize_edge(labels, edge), fci_edges))
    labels = get_BNlabel_nodes(bn)
    print('BN', labels)
    miic_arcs = set(map(lambda edge : labelize_edge(labels, edge), miic_arcs))
    miic_edges = set(map(lambda edge : labelize_edge(labels, edge), miic_edges))
    
    common_edges = len(fci_edges & (miic_edges | miic_arcs))
    
    total_fci = len(fci_edges)
    total_miic = len(miic_edges) + len(miic_arcs)
    jaccard_similarity = common_edges / (total_fci + total_miic - common_edges) if (total_fci + total_miic - common_edges) > 0 else 0

    return {
        'fci_edges' : len(fci_edges),
        'fci_unoriented' : len(fci_unoriented),
        'fci_oriented' : len(fci_oriented),
        'miic_edges' : len(miic_edges),
        'miic_arcs' : len(miic_arcs),
        'common_edges' : common_edges,
        'jaccard_similarity' : jaccard_similarity
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
        dot.edge(labels[i], labels[j], dir="foward")
    return dot

def print_comparison_results(comparison, fci_time, miic_time):
    '''Display comparison between FCI and MIIC.'''
    print("-" * 60)
    print("COMPARAISON FCI vs MIIC")
    print("-" * 60)
    print(f"Structure:")
    print(f"  FCI  - Total arêtes squelette: {comparison['fci_edges']}")
    print(f"       - Orientées: {comparison['fci_oriented']}")
    print(f"       - Non-orientées: {comparison['fci_unoriented']}")
    print(f"  MIIC - Total arêtes squelette : {comparison['miic_edges']+comparison['miic_arcs']}")
    print(f"       - Arcs: {comparison['miic_arcs']}")
    print(f"\nSimilarité:")
    print(f"  Arêtes communes: {comparison['common_edges']}")
    print(f"  Jaccard similarity: {comparison['jaccard_similarity']:.3f}")
    print(f"\nTemps d'exécution:")
    print(f"  FCI:  {fci_time:.3f}s")
    print(f"  MIIC: {miic_time:.3f}s")
    print(f"  Ratio (FCI/MIIC): {fci_time/miic_time:.2f}x")

if __name__=='__main__':
    # Création d'un réseau bayésien simple avec une structure connue
    print("📝 Test 1: Réseau simple (4 variables)")
    print("-" * 60)

    bn_simple = gum.BayesNet("Simple_Network")

    # Ajout des variables
    a = bn_simple.add(gum.LabelizedVariable('A', 'Variable A', 2))
    b = bn_simple.add(gum.LabelizedVariable('B', 'Variable B', 2))
    c = bn_simple.add(gum.LabelizedVariable('C', 'Variable C', 2))
    d = bn_simple.add(gum.LabelizedVariable('D', 'Variable D', 2))
    e = bn_simple.add(gum.LabelizedVariable('E', 'Variable E', 2))


    # Ajout des arcs
    bn_simple.addArc(a, b)
    bn_simple.addArc(a, d)
    bn_simple.addArc(b, e)
    bn_simple.addArc(c, b)
    bn_simple.addArc(c, e)
    bn_simple.addArc(d, c)


    # Génération des CPTs aléatoires
    bn_simple.generateCPTs()

    print(f"Réseau créé: {bn_simple.size()} variables, {bn_simple.sizeArcs()} arcs")

    df_simple,_ = generate_test_data(bn_simple, n_samples=1000)

    learner_simple = gum.BNLearner(df_simple, bn_simple)

    def run_miic(learner):
        learner.useMIIC()
        return learner.learnPDAG()

    miic_simple, miic_time_simple = measure_execution_time(run_miic, learner_simple)
    print(f"MIIC terminé en {miic_time_simple:.3f}s")

    # print(labelize_pdag_nodes(miic_simple,bn_simple))

    labels = get_BNlabel_nodes(bn_simple)
    print(labels)
    edges = miic_simple.edges()
    print(edges)
    edges = map(lambda edge : labelize_edge(labels, edge), edges)
    print(list(edges))

    # Visualisation du résultat MIIC
    # print("\nPDAG résultant (MIIC):")
    # print(miic_simple.toDot())
    # print(utils.get_variablesBN(bn_simple))
    # Source(miic_simple.toDot())
