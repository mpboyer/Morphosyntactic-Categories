import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gudhi.clustering.tomato import Tomato
import gudhi
from gudhi.wasserstein import wasserstein_distance
from gudhi.wasserstein.barycenter import lagrangian_barycenter

import contextlib
import joblib
import argparse
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import itertools

import csv_proximity
from visualise import get_data_set
from csv_proximity import get_matrix_csv

# import plotly.express as px

parser = argparse.ArgumentParser()
parser.add_argument("--f1")
parser.add_argument("--f2")
parser.add_argument("-i", "--interactive")
parser.add_argument("-mode", "--mode", default="")
files = parser.parse_args()

if not files.interactive:
    matplotlib.use("pgf")
    preamble = r"\usepackage{xcolor}\definecolor{vulm}{HTML}{7d1dd3}\definecolor{yulm}{HTML}{ffe500}"
    matplotlib.rc("pgf", texsystem="pdflatex", preamble=preamble)

UDDIR = "ud-treebanks-v2.14"
MODE = files.mode
VECTOR_DIR = f"../{MODE}_Case_RelDep_Matches" if MODE else "../Case_RelDep_Matches"
SAVE_DIR = "Figures/Visualisations"


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def is_prefix(s1: str, s2: str) -> bool:
    """
    Looks at len(s1) first chars in b to check if s1 is a prefix of s2.
    :rtype: bool
    :param s1: pattern
    :param s2: text
    :return: True iff s1 is a prefix of s2.
    """
    if len(s1) > len(s2):
        return False
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            return False
    return True


def fronce(s):
    match s:
        case ("Nouns"):
            return "Noms"
        case ("Pronouns"):
            return "Pronoms"
        case (_):
            raise ValueError("Calice!")


def plot_tomato(tomat):
    l = tomat.max_weight_per_cc_.min()
    r = tomat.max_weight_per_cc_.max()
    if tomat.diagram_.size > 0:
        plt.plot(tomat.diagram_[:, 0], tomat.diagram_[:, 1], "o", color="#7d1dd3")
        l = min(l, tomat.diagram_[:, 1].min())
        r = max(r, tomat.diagram_[:, 0].max())
    if l == r:
        if l > 0:
            l, r = 0.9 * l, 1.1 * r
        elif l < 0:
            l, r = 1.1 * l, 0.9 * r
        else:
            l, r = -1.0, 1.0
    plt.plot([l, r], [l, r])
    plt.plot(
        tomat.max_weight_per_cc_, np.full(tomat.max_weight_per_cc_.shape, 1.1 * l - 0.1 * r), "o", color="#ffe500"
    )


def tomato(case1, case2):
    case_data_set = get_data_set(case1, case2).values
    data = case_data_set[:, 5:].copy()
    for i in range(len(data)):
        data[i] /= case_data_set[i][4]
    t = Tomato()
    t.fit(data)
    plot_tomato(t)
    # plt.scatter(data[:,0], data[:, 1], marker='.', s=1, c=t.labels_)
    plt.title(
        r"\Large Tomato Algorithm Clusters for \textcolor{vulm}{" + case1 + r"}, \textcolor{yulm!80!black}{" + case2 + "}"
    )
    savepath = f"{SAVE_DIR}/tomato_{case1}_{case2}.pdf" if not MODE else f"{SAVE_DIR}/tomato_{case1}_{case2}_{MODE}.pdf"
    plt.savefig(savepath)


def data_from_case(case):
    case_data_set = pd.read_csv(f"{VECTOR_DIR}/RelDep_matching_Case={case}.csv")
    case_data_set.insert(1, "Case", case, True)
    case_data_set["Treebank"].map(lambda n: n + f"_{case}")

    for column in case_data_set.columns:
        case_data_set.replace(
            {
                column: np.nan}, 0., inplace=True
        )
    data = case_data_set.values[:, 5:]
    for i in range(len(data)):
        data[i] /= case_data_set.values[i, 4]
    return data


def witness_complex(case):
    number_of_landmarks = 42
    witnesses = data_from_case(case)
    landmarks = witnesses[:number_of_landmarks]
    witness_complex = gudhi.EuclideanStrongWitnessComplex(witnesses=witnesses, landmarks=landmarks)
    simplex_tree = witness_complex.create_simplex_tree(max_alpha_square=2., limit_dimension=1)
    # print(f"Number of Simplices = {simplex_tree.num_simplices()}")
    # diag = simplex_tree.persistence()

    print("Betti Numbers")
    print(simplex_tree.betti_numbers())

    # PTDR ÇA SEGFAULT
    # gudhi.persistence_graphical_tools.plot_persistence_diagram(diag)
    # plt.title(r"\Large Persistent Homology Diagram for Witness Complex on \textcolor{vulm}{" + case + "}")
    # savepath = f"{SAVE_DIR}/wc_{case}.pdf" if not MODE else f"{SAVE_DIR}/wc_{case}_{MODE}.pdf"
    # plt.savefig(savepath)


def alpha_complex(case):
    points = data_from_case(case)[:200]
    ac = gudhi.AlphaComplex(points=points)
    simplex_tree = ac.create_simplex_tree(max_dimension=1)
    diag = simplex_tree.persistence()
    gudhi.persistence_graphical_tools.plot_persistence_diagram(diag)
    plt.show()


def rips_complex(case):
    points = data_from_case(case)[:200]
    ac = gudhi.rips_complex.RipsComplex(points=points)
    simplex_tree = ac.create_simplex_tree(max_dimension=2)
    diag = simplex_tree.persistence()
    print(simplex_tree.betti_numbers())
    gudhi.persistence_graphical_tools.plot_persistence_diagram(diag)
    plt.title(r"\Large Persistent Homology Diagram for Rips Complex on \textcolor{vulm}{" + case + "}")
    savepath = f"{SAVE_DIR}/rc_{case}.pdf" if not MODE else f"{SAVE_DIR}/rc_{case}_{MODE}.pdf"
    plt.savefig(savepath)


def cubical_complex(case):
    points = data_from_case(case)
    cc = gudhi.cubical_complex.CubicalComplex(top_dimensional_cells=points)
    diag = cc.persistence()
    cc.compute_persistence()
    print(cc.betti_numbers())
    gudhi.persistence_graphical_tools.plot_persistence_diagram(diag)
    plt.title(r"\Large Persistent Homology Diagram for Cubical Complex on \textcolor{vulm}{" + case + "}")
    savepath = f"{SAVE_DIR}/cc_{case}.pdf" if not MODE else f"{SAVE_DIR}/cc_{case}_{MODE}.pdf"
    plt.savefig(savepath)


def cubical_persistence(case1):
    p1 = data_from_case(case1)
    c1 = gudhi.cubical_complex.CubicalComplex(top_dimensional_cells=p1)
    return np.array([list(c[1]) for c in c1.persistence()])


def rips_persistence(case1):
    p1 = data_from_case(case1)
    c1 = gudhi.rips_complex.RipsComplex(points=p1)
    simplex_tree = c1.create_simplex_tree(max_dimension=2)
    diag = simplex_tree.persistence()
    return np.array([list(c[1]) for c in diag])


def wasserstein_cc(case1, case2):
    d1 = cubical_persistence(case1)
    d2 = cubical_persistence(case2)
    cost = wasserstein_distance(d1, d2, order=2., internal_p=2)
    print(f"Wasserstein distance value between {case1} and {case2} = {cost:.2f}")

    # dgm1_to_diagonal = matchings[matchings[:, 1] == -1, 0]
    # dgm2_to_diagonal = matchings[matchings[:, 0] == -1, 1]
    # off_diagonal_match = np.delete(matchings, np.where(matchings == -1)[0], axis=0)

    # for i, j in off_diagonal_match:
    #     print(f"point {i} in dgm1 is matched to point {j} in dgm2")
    # for i in dgm1_to_diagonal:
    #     print(f"point {i} in dgm1 is matched to the diagonal")
    # for j in dgm2_to_diagonal:
    #     print(f"point {j} in dgm2 is matched to the diagonal")


def wasserstein_rc(case1, case2):
    d1 = rips_persistence(case1)
    d2 = rips_persistence(case2)
    cost = wasserstein_distance(d1, d2, order=2., internal_p=2)
    print(f"Wasserstein distance value between {case1} and {case2} = {cost:.2f}")


def lagrange_barycenter(case_list):
    pdiagset = [cubical_persistence(c) for c in case_list]
    bary, log = lagrangian_barycenter(pdiagset, verbose=True, init=1)
    print(f"Energy = {log['energy']:.3f}")
    print(bary)
    gudhi.persistence_graphical_tools.plot_persistence_diagram(bary)
    plt.show()


def get_matrix_list(treebanks):
    basis = csv_proximity.overall_basis_csv()
    case_space = []
    for csv in sorted(filter(lambda t: t[-4:] == ".csv", os.listdir(f"{VECTOR_DIR}"))):
        with open(f"{VECTOR_DIR}/{csv}", "r") as csv_file:
            attributes = next(csv_file).rstrip().split(",")[4:]
            for tree in csv_file:
                parsed_tree = tree.rstrip().split(",")
                if parsed_tree[0] in treebanks:
                    coordinates = dict(zip(attributes, map(csv_proximity.void_to_zero, parsed_tree[4:])))
                    coordinates["Total"] = csv_proximity.void_to_zero(parsed_tree[3])
                    case_space.append(coordinates)

    matrix = np.array([[0. for _ in case_space] for _ in basis])
    for row, b in enumerate(basis):
        for column, vector in enumerate(case_space):
            total = vector["Total"]
            if total != 0.:
                matrix[row, column] = vector.get(b, 0.) / total
    return matrix


def cc_manifold_bank_list(treebanks):
    points = get_matrix_list(treebanks)
    cc = gudhi.cubical_complex.CubicalComplex(top_dimensional_cells=points)
    diag = cc.persistence()
    cc.compute_persistence()
    print(cc.betti_numbers())
    gudhi.persistence_graphical_tools.plot_persistence_diagram(diag)
    plt.title(
        r"\Large Persistent Homology Diagram for Cubical Complex on" + "\n" + r"\textcolor{vulm}{" + " ".join(
            treebanks
            ) + "}"
        )
    savepath = f"{SAVE_DIR}/cc_" + "_".join(treebanks) + ".pdf" if not MODE else f"{SAVE_DIR}/cc_" + "_".join(
        treebanks
        ) + f"_{MODE}.pdf"
    plt.savefig(savepath)


studied_languages = ['tr_boun-ud-train', 'sk_snk-ud-train', 'ab_abnc-ud-test', 'eu_bdt-ud-train', 'fi_ftb-ud-train',
                     'hit_hittb-ud-test', 'ta_ttb-ud-train', 'wbp_ufal-ud-test']

if __name__ == '__main__':
    for s in studied_languages:
        cc_manifold_bank_list([s])
