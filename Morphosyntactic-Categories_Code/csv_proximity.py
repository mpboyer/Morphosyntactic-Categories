import joblib
import itertools
import pandas
from tqdm import tqdm
import numpy as np
from operator import itemgetter
import os
import contextlib
import graphviz
from linalg import distance
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--f1")
parser.add_argument("--f2")
parser.add_argument("-mode", "--mode", default="")
files = parser.parse_args()

UDDIR = "../ud-treebanks-v2.14"
MODE = files.mode
VECTOR_DIR = f"../{MODE}_RelDep_Matches" if MODE else "../Case_RelDep_Matches"
SAVE_DIR = f"../{MODE}_Proximities" if MODE else "../Case_Proximities"


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


def void_to_zero(s):
    if s == '':
        return 0.
    return float(s)


def get_all_banks():
    abank = []
    treebanks = os.listdir(UDDIR)
    for treebank in treebanks:
        content = os.listdir(f"{UDDIR}/{treebank}")
        for c in list(filter(lambda t: t[-7:] == ".conllu", content)):
            abank.append(c[:-7])
    return abank


def get_all_features():
    cases = list(filter(lambda t: t[-4:] == ".csv", os.listdir(VECTOR_DIR)))
    return [(lambda t: tuple(t.split('_')[-1].split('.')[0].split('=')))(c) for c in cases]


def overall_basis_csv():
    basis = set()
    for csv in filter(lambda t: t[-4:] == ".csv", os.listdir(f"{VECTOR_DIR}")):
        with open(f"{VECTOR_DIR}/{csv}", "r") as csv_file:
            attributes = next(csv_file).rstrip().split(",")[4:]
        basis |= set(attributes)
    return basis


def get_matrix_csv(treebank):
    basis = overall_basis_csv()
    case_space = []
    features = []
    for csv in sorted(filter(lambda t: t[-4:] == ".csv", os.listdir(f"{VECTOR_DIR}"))):
        with open(f"{VECTOR_DIR}/{csv}", "r") as csv_file:
            attributes = next(csv_file).rstrip().split(",")[4:]
            for tree in csv_file:
                parsed_tree = tree.rstrip().split(",")
                if parsed_tree[0] == treebank:
                    coordinates = dict(zip(attributes, map(void_to_zero, parsed_tree[4:])))
                    coordinates["Total"] = void_to_zero(parsed_tree[3])
                    case_space.append(coordinates)
                    features.append((lambda t: tuple(t.split('_')[-1].split('.')[0].split('=')))(csv))

    matrix = np.array([[0. for _ in case_space] for _ in basis])
    for row, b in enumerate(basis):
        for column, vector in enumerate(case_space):
            total = vector["Total"]
            if total != 0.:
                matrix[row, column] = vector.get(b, 0.) / total
    return features, matrix


def closest(treebank1, treebank2):
    case1, matrix1 = get_matrix_csv(treebank1)
    case2, matrix2 = get_matrix_csv(treebank2)
    m1_dicts = {}
    m2_dicts = {}

    distance_matrix = [["" for _ in case2] for _ in case1]
    for row in range(len(case1)):
        for col in range(len(case2)):
            distance_matrix[row][col] = str(distance(matrix1[:, row], matrix2[:, col]))[:5]
    # print(str.join("\t", tuple(case1)))
    # print(str.join("\t", tuple(case2)))
    # for d in distance_matrix:
    #     print(str.join("\t", tuple(d)))

    for col, case in enumerate(case1):
        m1_dicts[case] = dict(
            [(c, distance(matrix1[:, col], matrix2[:, i])) for i, c in enumerate(case2)]
        )

    for col, case in enumerate(case2):
        m2_dicts[case] = dict(
            [(c, distance(matrix2[:, col], matrix1[:, i])) for i, c in enumerate(case1)]
        )

    for m in m1_dicts:
        mk = min(m1_dicts[m], key=m1_dicts[m].get)
        m1_dicts[m] = mk, float(m1_dicts[m][mk])
    for m in m2_dicts:
        mk = min(m2_dicts[m], key=m2_dicts[m].get)
        m2_dicts[m] = mk, float(m2_dicts[m][mk])
    return treebank1, m1_dicts, treebank2, m2_dicts


thresh = 3


def closest_list(treebanks):
    matrices = list(get_matrix_csv(t) for t in treebanks)
    result = []
    for (i1, m1), (i2, m2) in tqdm(
            itertools.combinations(enumerate(matrices), 2), desc="Computing Distances", colour="#7d1dd3",
            total=len(matrices) * (len(matrices) - 1) / 2
            ):

        distances = np.array([[0. for _ in m2[0]] for _ in m1[0]])
        t1 = treebanks[i1]
        t2 = treebanks[i2]
        for i in range(len(m1[0])):
            for j in range(len(m2[0])):
                distances[i, j] = distance(m1[1][:, i], m2[1][:, j])

        pair_result_dict = {}
        for row, c in enumerate(m1[0]):
            d = min(enumerate(distances[row, :]), key=itemgetter(1))
            pair_result_dict[c] = m2[0][d[0]], d[1]
        result.append((t1, t2, pair_result_dict))

        pair_result_dict = {}
        for column, c in enumerate(m2[0]):
            d = min(enumerate(distances[:, column]), key=itemgetter(1))
            pair_result_dict[c] = m1[0][d[0]], d[1]
        result.append((t2, t1, pair_result_dict))

    return result


def closest_graph_list(treebanks):
    edges = closest_list(treebanks)
    graphical = graphviz.Digraph(
        f"Graph of Nearest Neighbours for {MODE} in " + ",".join(
            treebanks
        ) if MODE else f"Graph of Nearest Neighbours for {MODE} in " + ",".join(treebanks)
    )
    # graphical.graph_attr['ratio'] = '0.1'
    graphical.graph_attr['engine'] = 'circo'
    for treebank1, treebank2, distances in tqdm(edges, desc="Reporting Edges to the Graph"):
        treebank1 = treebank1.split("-")[0]
        treebank2 = treebank2.split("-")[0]
        for n1, (n2, length) in distances.items():
            cbar = ['plum', 'purple', 'orangered', 'orange', 'goldenrod', 'lawngreen', 'forestgreen', 'springgreen', 'turquoise', 'deepskyblue']
            c = str(cbar[-int(np.floor(5*length))])
            graphical.edge(f"{treebank1}_{n1}", f"{treebank2}_{n2}", label=f"{length:.3}", color=c, penwidth='2.0')

    # graphical = graphical.unflatten(stagger=3)
    if MODE:
        graphical.render(f"Figures/GNN/gnn_{MODE}_Only_" + "_".join(treebanks), format="pdf")
    else:
        graphical.render("Figures/GNN/gnn_" + "_".join(treebanks), format="pdf")


def format_tuple_dict(d):
    for key, value in d.items():
        print(f'{key} : {value[0]}, Distance = {value[1]:.5f}')


def format_dict(d):
    for key, value in d.items():
        print(f'{key} : {value}')


def sample_size(treebank):
    samples = {}
    for csv in filter(lambda t: t[-4:] == ".csv", os.listdir(f"{VECTOR_DIR}")):
        with open(f"{VECTOR_DIR}/{csv}", "r") as csv_file:
            for tree in csv_file:
                parsed_tree = tree.rstrip().split(",")
                if parsed_tree[0] == treebank:
                    samples[csv[-7:-4]] = int(parsed_tree[3])
    return samples


studied_languages = ['tr_boun-ud-train', 'sk_snk-ud-train', 'ab_abnc-ud-test', 'eu_bdt-ud-train', 'fi_ftb-ud-train',
                     'hit_hittb-ud-test', 'ta_ttb-ud-train', 'wbp_ufal-ud-test']

if __name__ == "__main__":
    for l in itertools.combinations(studied_languages, 3):
        closest_graph_list(list(l))
    # compute_angles_csv()
    # compute_distances_csv()
    # tabulize_angle_pairs_csv()
    # print(len(get_all_cases()), len(overall_basis_csv()))
    # t1, d1, t2, d2 = closest(files.f1, files.f2)
    # print(f"Distances for {t1}")
    # format_tuple_dict(d1)
    # print(f"Distances for {t2}")
    # format_tuple_dict(d2)
    # d1 = sample_size(files.f1)
    # print(f"Sample Sizes for {files.f1}")
    # format_dict(d1)
    # d2 = sample_size(files.f2)
    # print(f"Sample Sizes for {files.f2}")
    # format_dict(d2)
    # closest_graph(files.f1, files.f2)
